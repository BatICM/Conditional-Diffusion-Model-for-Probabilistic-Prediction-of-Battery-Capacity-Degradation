import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from pathlib import Path

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Config:
    total_steps = 120
    input_steps = 8
    pred_steps = 112
    num_features = 10
    num_targets = 1
    hidden_size = 128
    num_layers = 2
    batch_size = 16
    epochs = 1000
    lr = 1e-3
    data_dir = "../finalData"
    result_dir = "./seq8" 
    model_save_dir = "fold_models"
    scaler_save_dir = "scalers"
    metrics_save_path = "all_metrics_results.csv"

config = Config()

Path(config.data_dir).mkdir(parents=True, exist_ok=True)
Path(config.result_dir).mkdir(parents=True, exist_ok=True)
Path(os.path.join(config.result_dir, config.model_save_dir)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(config.result_dir, config.scaler_save_dir)).mkdir(parents=True, exist_ok=True)


class TimeSeriesDataset(Dataset):
    def __init__(self, files, scaler_X=None, scaler_y=None):
        self.files = files
        self.X, self.y, self.masks, self.original_lengths = [], [], [], []
        
        for file in files:
            df = pd.read_csv(file)
            original_length = len(df)
            self.original_lengths.append(original_length)
            
            data = df.iloc[:config.total_steps, :config.num_features].values
            padded_features = np.zeros((config.total_steps, config.num_features))
            padded_targets = np.zeros((config.total_steps, 1))
            mask = np.zeros(config.total_steps)
            
            if original_length >= config.total_steps:
                padded_features[:] = data[:config.total_steps]
                padded_targets[:] = df.iloc[:config.total_steps, config.num_features - 1].values.reshape(-1, 1)
                mask[:] = 1
            else:
                padded_features[:original_length] = data
                padded_targets[:original_length] = df.iloc[:original_length, config.num_features - 1].values.reshape(-1, 1)
                mask[:original_length] = 1
                
            self.X.append(padded_features[:config.input_steps])
            self.y.append(padded_targets)
            self.masks.append(mask)
        
        if scaler_X is None:
            all_valid_X = np.vstack(self.X)
            self.scaler_X = MinMaxScaler().fit(all_valid_X)
        else:
            self.scaler_X = scaler_X
            
        if scaler_y is None:
            all_valid_y = np.vstack([y[mask.astype(bool)] for y, mask in zip(self.y, self.masks)])
            self.scaler_y = MinMaxScaler().fit(all_valid_y)
        else:
            self.scaler_y = scaler_y
            
        for i in range(len(self.X)):
            self.X[i] = self.scaler_X.transform(self.X[i])
            valid_idx = self.masks[i].astype(bool)
            if np.any(valid_idx):
                self.y[i][valid_idx] = self.scaler_y.transform(self.y[i][valid_idx])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor(self.y[idx]),
            torch.FloatTensor(self.masks[idx]),
            self.original_lengths[idx]
        )

class ProbSeq2SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=1, 
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_size, 2) 
        
    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        decoder_input = x[:, -1:, -1:].repeat(1, config.pred_steps, 1)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        params = self.fc(decoder_output)
        mu_seq = params[:, :, 0:1]
        log_sigma_seq = params[:, :, 1:2]
        sigma_seq = torch.exp(log_sigma_seq)
        
        dist_output = torch.distributions.Normal(mu_seq, sigma_seq)
        return dist_output

def test_and_visualize_prob(models, test_dataset, test_group_label, scaler_X, scaler_y, save_plot=True):
    
    test_dataset.scaler_X = scaler_X
    test_dataset.scaler_y = scaler_y
    
    all_preds, all_sigmas = [], []
    for model in models:
        model.eval()
        preds, sigmas = [], []
        with torch.no_grad():
            for x, _, _, _ in DataLoader(test_dataset, batch_size=config.batch_size):
                x = x.to(device)
                pred_dist = model(x)
                pred = pred_dist.mean.cpu().numpy()
                sigma = pred_dist.stddev.cpu().numpy()
                
                pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).reshape(-1, config.pred_steps)
               
                if scaler_y.scale_[0] != 0:
                    sigma = sigma / scaler_y.scale_[0]

                preds.append(pred)
                sigmas.append(sigma)
                
        all_preds.append(np.concatenate(preds, axis=0))
        all_sigmas.append(np.concatenate(sigmas, axis=0))
    
    avg_preds = np.mean(np.stack(all_preds, axis=0), axis=0)
    avg_sigmas = np.mean(np.stack(all_sigmas, axis=0), axis=0)
    
    all_metrics = []
    inference_results = []
    num_samples = len(test_dataset)
    
    num_plots = min(num_samples, 4)
    if save_plot:
        plt.figure(figsize=(20, 5 * num_plots))
    
    for idx in range(num_samples):
        true_y = scaler_y.inverse_transform(test_dataset.y[idx]).flatten()
        original_len = test_dataset.original_lengths[idx]
        effective_pred_steps = min(config.pred_steps, original_len - config.input_steps)
        
        mse, rmse, mae, avg_ci_width, coverage_rate = [np.nan] * 5
        
        if effective_pred_steps > 0:
            true_pred = true_y[config.input_steps : config.input_steps + effective_pred_steps]
            pred_pred = avg_preds[idx][:effective_pred_steps].flatten()
            sigma_pred = avg_sigmas[idx][:effective_pred_steps].flatten()
            
            pred_lower = pred_pred - 1.96 * sigma_pred
            pred_upper = pred_pred + 1.96 * sigma_pred
            
            mse = mean_squared_error(true_pred, pred_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_pred, pred_pred)
            avg_ci_width = np.mean(pred_upper - pred_lower)
            
            covered = (true_pred >= pred_lower) & (true_pred <= pred_upper)
            coverage_rate = np.mean(covered)

            for t in range(effective_pred_steps):
                inference_results.append({
                    "sample_id": idx,
                    "time_step": t + config.input_steps,
                    "true": true_pred[t],
                    "pred": pred_pred[t],
                    "lower": pred_lower[t],
                    "upper": pred_upper[t],
                    "covered": covered[t]
                })

            if save_plot and idx < num_plots:
                plt.subplot(num_plots, 1, idx + 1)
                input_part = true_y[:config.input_steps]
                complete_true = np.concatenate([input_part, true_pred])
                x_total = np.arange(len(complete_true))
                
                plt.plot(x_total, complete_true, label='True', color='green', linewidth=1.5)
                plt.plot(np.arange(config.input_steps, config.input_steps + effective_pred_steps), pred_pred,
                         label='Predicted', color='red', linestyle='--', marker='o', markersize=4, linewidth=1.2)
                plt.fill_between(np.arange(config.input_steps, config.input_steps + effective_pred_steps),
                                 pred_lower, pred_upper, color='orange', alpha=0.3, label='95% Confidence Interval')
                plt.axvline(x=config.input_steps, color='black', linestyle='--', label='Prediction Start')
                plt.title(f"Sample {idx+1} (RMSE: {rmse:.2f}, MAE: {mae:.2f}, CI Width: {avg_ci_width:.2f})", fontsize=10)
                plt.xlabel("Time Step", fontsize=9)
                plt.ylabel("Value", fontsize=9)
                plt.legend(fontsize=8, loc="upper right")
        
        all_metrics.append({
            "Test_Group": test_group_label,
            "Sample": f"{test_group_label}_{idx+1}",
            "MSE": mse, "RMSE": rmse, "MAE": mae,
            "Avg_CI_Width": avg_ci_width, "Coverage": coverage_rate
        })

    if save_plot:
        plt.tight_layout(pad=3.0)
        plot_path = os.path.join(config.result_dir, f"{test_group_label}_results_prob.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Test {test_group_label} results plot saved to {plot_path}")
    infer_csv_path = os.path.join(config.result_dir, f"{test_group_label}_inference_results.csv")
    pd.DataFrame(inference_results).to_csv(infer_csv_path, index=False)
    print(f"Detailed inference results saved to {infer_csv_path}")

    return all_metrics

def rolling_cross_validation():
    all_metrics = []
    val_loss_logs = []
    
    all_files = [os.path.join(config.data_dir, f"data{i}.csv") for i in range(1, 21)]
    
    groups_of_files = [all_files[i:i + 4] for i in range(0, len(all_files), 4)]
    
    for i in range(len(groups_of_files)):
        test_files = groups_of_files[i]
        train_val_files = [f for j, g in enumerate(groups_of_files) if i != j for f in g]
        group_label = f"Group{i+1}"
        
        print(f"\n=== Processing: Test set = {group_label} ({len(test_files)} files) ===")
        print(f"   Training/Validation set = {len(train_val_files)} files")

        full_train_dataset = TimeSeriesDataset(train_val_files)
        scaler_X, scaler_y = full_train_dataset.scaler_X, full_train_dataset.scaler_y
        
        scaler_path = os.path.join(config.result_dir, config.scaler_save_dir, f"{group_label}_scalers.pkl")
        joblib.dump((scaler_X, scaler_y), scaler_path)
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_files)):
            print(f"   Fold {fold+1}/4")
            fold_train_files = [train_val_files[i] for i in train_idx]
            fold_val_files = [train_val_files[i] for i in val_idx]
            
            train_dataset = TimeSeriesDataset(fold_train_files, scaler_X=scaler_X, scaler_y=scaler_y)
            val_dataset = TimeSeriesDataset(fold_val_files, scaler_X=scaler_X, scaler_y=scaler_y)
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
            
            model = ProbSeq2SeqModel().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
            
            best_val_loss = float('inf')
            
            for epoch in range(config.epochs):
                model.train()
                for batch_x, batch_y, batch_mask, _ in train_loader:
                    batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                    
                    target_slice = slice(config.input_steps, config.input_steps + config.pred_steps)
                    batch_y_pred = batch_y[:, target_slice]
                    mask_tensor = batch_mask[:, target_slice].unsqueeze(-1)
                
                    optimizer.zero_grad()
                    pred_dist = model(batch_x)
                    nll_loss = -pred_dist.log_prob(batch_y_pred)
                    loss = (nll_loss * mask_tensor).sum() / mask_tensor.sum()
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_x, val_y, val_mask, _ in val_loader:
                        val_x, val_y, val_mask = val_x.to(device), val_y.to(device), val_mask.to(device)
                        val_y_pred = val_y[:, target_slice]
                        mask_tensor = val_mask[:, target_slice].unsqueeze(-1)
                        
                        pred_dist = model(val_x)
                        nll_loss = -pred_dist.log_prob(val_y_pred)
                        loss = (nll_loss * mask_tensor).sum() / mask_tensor.sum()
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_loss_logs.append({ "Group": group_label, "Fold": fold+1, "Epoch": epoch, "Val_Loss": avg_val_loss })
                lr_scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_save_path = os.path.join(config.result_dir, config.model_save_dir, f"{group_label}_fold{fold+1}_best.pth")
                    torch.save(model.state_dict(), model_save_path)
            best_model = ProbSeq2SeqModel().to(device)
            best_model.load_state_dict(torch.load(model_save_path))
            fold_models.append(best_model)
        test_dataset = TimeSeriesDataset(test_files, scaler_X=scaler_X, scaler_y=scaler_y)
        group_metrics = test_and_visualize_prob(fold_models, test_dataset, group_label, scaler_X, scaler_y)
        all_metrics.extend(group_metrics)
    metrics_df = pd.DataFrame(all_metrics).dropna()
    metrics_df.to_csv(os.path.join(config.result_dir, config.metrics_save_path), index=False)
    
    print("\n" + "="*50)
    print(f"Final Average Results Across All {len(metrics_df)} Test Samples")
    print("="*50)
    if not metrics_df.empty:
        print(f"Average RMSE: {metrics_df['RMSE'].mean():.4f}")
        print(f"Average MAE: {metrics_df['MAE'].mean():.4f}")
        print(f"Average CI Width: {metrics_df['Avg_CI_Width'].mean():.4f}")
        print(f"Average Coverage: {metrics_df['Coverage'].mean():.4f}")
    else:
        print("No valid metrics were generated.")
    print("="*50)
    
    pd.DataFrame(val_loss_logs).to_csv(os.path.join(config.result_dir, "val_loss_log.csv"), index=False)

if __name__ == "__main__":
    rolling_cross_validation()