import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class Config:
    total_steps = 120
    input_steps = 24
    pred_steps = 96
    num_features = 10
    num_targets = 1
    hidden_size = 128
    num_layers = 2
    batch_size = 16 
    epochs = 1000
    lr = 1e-3
    data_dir = "../finalData"
    result_dir = "./lstm24" 
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
            valid_indices = self.masks[i].astype(bool)
            self.y[i][valid_indices] = self.scaler_y.transform(self.y[i][valid_indices])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor(self.y[idx]),
            torch.FloatTensor(self.masks[idx]),
            self.original_lengths[idx]
        )

class ProbLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.2 if config.num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, config.pred_steps * 2 * config.num_targets)
        )

    def forward(self, x):
        h0 = torch.zeros(config.num_layers, x.size(0), config.hidden_size).to(device)
        c0 = torch.zeros(config.num_layers, x.size(0), config.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = out.view(-1, config.pred_steps, 2 * config.num_targets)
        mu = out[:, :, 0:config.num_targets]
        log_sigma = out[:, :, config.num_targets:]
        sigma = torch.exp(log_sigma)
        dist_output = torch.distributions.Normal(mu, sigma)
        return dist_output

def test_and_visualize_prob(models, test_dataset, test_group_label, scaler_X, scaler_y, save_plot=True):
    all_preds, all_sigmas = [], []
    test_dataset.scaler_X, test_dataset.scaler_y = scaler_X, scaler_y

    for model in models:
        model.eval()
        preds, sigmas = [], []
        with torch.no_grad():
            for x, y, mask, _ in DataLoader(test_dataset, batch_size=config.batch_size):
                x = x.to(device)
                pred_dist = model(x)
                pred = pred_dist.mean.cpu().numpy()
                sigma = pred_dist.stddev.cpu().numpy()
                pred = test_dataset.scaler_y.inverse_transform(pred.reshape(-1, 1)).reshape(-1, config.pred_steps)
                
                scale_value = (test_dataset.scaler_y.data_max_ - test_dataset.scaler_y.data_min_)
                sigma = sigma * scale_value[0]
                
                preds.append(pred)
                sigmas.append(sigma)
        all_preds.append(np.concatenate(preds, axis=0))
        all_sigmas.append(np.concatenate(sigmas, axis=0))

    avg_preds = np.mean(np.stack(all_preds, axis=0), axis=0)
    avg_sigmas = np.mean(np.stack(all_sigmas, axis=0), axis=0)

    mse_values, rmse_values, mae_values, ci_width_values, coverage_values = [], [], [], [], []
    inference_results = []
    num_samples = len(test_dataset)

    for idx in range(num_samples):
        true_y = test_dataset.scaler_y.inverse_transform(test_dataset.y[idx]).flatten()
        effective_pred_steps = min(config.pred_steps, test_dataset.original_lengths[idx] - config.input_steps)
        if effective_pred_steps <= 0: continue
        true_pred = true_y[config.input_steps:config.input_steps+effective_pred_steps]
        pred_pred = avg_preds[idx][:effective_pred_steps].flatten()
        sigma_pred = avg_sigmas[idx][:effective_pred_steps].flatten()

        mse = mean_squared_error(true_pred, pred_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_pred, pred_pred)
        mse_values.append(mse)
        rmse_values.append(rmse)
        mae_values.append(mae)

        pred_lower = pred_pred - 1.96 * sigma_pred
        pred_upper = pred_pred + 1.96 * sigma_pred
        avg_ci_width = np.mean(pred_upper - pred_lower)
        ci_width_values.append(avg_ci_width)
        
        covered = (true_pred >= pred_lower) & (true_pred <= pred_upper)
        coverage_rate = np.mean(covered) if len(covered) > 0 else 0
        coverage_values.append(coverage_rate)

        for t in range(effective_pred_steps):
            inference_results.append({
                "sample_id": idx,
                "time_step": t + config.input_steps,
                "true": true_y[t + config.input_steps],
                "pred": pred_pred[t],
                "lower": pred_lower[t],
                "upper": pred_upper[t],
                "covered": covered[t]
            })

    infer_csv_path = os.path.join(config.result_dir, f"{test_group_label}_inference_results.csv")
    pd.DataFrame(inference_results).to_csv(infer_csv_path, index=False)

    if save_plot:
        plt.figure(figsize=(20, 30))
        for idx in range(min(10, num_samples)):
            plt.subplot(5, 2, idx+1)
            true_y = test_dataset.scaler_y.inverse_transform(test_dataset.y[idx]).flatten()
            effective_pred_steps = min(config.pred_steps, test_dataset.original_lengths[idx] - config.input_steps)
            if effective_pred_steps <= 0: continue
            
            input_part = true_y[:config.input_steps]
            true_pred_part = true_y[config.input_steps:config.input_steps+effective_pred_steps]
            pred_pred_part = avg_preds[idx][:effective_pred_steps].flatten()
            sigma_pred_part = avg_sigmas[idx][:effective_pred_steps].flatten()
            pred_lower = pred_pred_part - 3 * sigma_pred_part
            pred_upper = pred_pred_part + 3 * sigma_pred_part

            rmse = np.sqrt(mean_squared_error(true_pred_part, pred_pred_part))
            mae = mean_absolute_error(true_pred_part, pred_pred_part)
            avg_ci_width = np.mean(pred_upper - pred_lower)
            
            x_total = np.arange(config.input_steps + effective_pred_steps)
            complete_true = np.concatenate([input_part, true_pred_part])

            plt.plot(x_total, complete_true, label='True', color='green', linewidth=1.5)
            plt.plot(np.arange(config.input_steps, config.input_steps+effective_pred_steps), pred_pred_part, label='Predicted', color='red', linestyle='--', marker='o', markersize=4, linewidth=1.2)
            plt.fill_between(np.arange(config.input_steps, config.input_steps+effective_pred_steps), pred_lower, pred_upper, color='orange', alpha=0.3, label='Confidence Interval (6σ)')
            plt.axvline(x=config.input_steps, color='black', linestyle='--', label='Prediction Start')
            plt.title(f'Sample {idx+1} (RMSE: {rmse:.2f}, MAE: {mae:.2f}, CI Width: {avg_ci_width:.2f})', fontsize=10)
            plt.xlabel('Time Step', fontsize=9)
            plt.ylabel('Value', fontsize=9)
            plt.legend(fontsize=8, loc='upper right')

        plt.tight_layout(pad=3.0)
        plot_path = os.path.join(config.result_dir, f"{test_group_label}_results_prob.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()

    metrics = []
    for i in range(len(mse_values)): 
        metrics.append({
            "Test_Group": test_group_label,
            "Sample": f"{test_group_label}_{i+1}",
            "MSE": mse_values[i],
            "RMSE": rmse_values[i],
            "MAE": mae_values[i],
            "Avg_CI_Width": ci_width_values[i],
            "Coverage": coverage_values[i]
        })
    return metrics

def rolling_cross_validation():
    
    all_metrics = []
    val_loss_logs = []
    
    
    all_files = [os.path.join(config.data_dir, f"data{i}.csv") for i in range(1, 21)]
    
    groups_of_files = [all_files[i:i + 4] for i in range(0, len(all_files), 4)]
    
    for i in range(len(groups_of_files)):
        test_files = groups_of_files[i]
        train_val_files = [f for g in groups_of_files if g is not test_files for f in g]
        group_label = f"Group{i+1}"
        
        print("\n" + "="*60)
        print(f"Processing: Test set = {group_label} ({len(test_files)} files), Training set = {len(train_val_files)} files")
        print("="*60)

        full_train_dataset = TimeSeriesDataset(train_val_files)
        scaler_X, scaler_y = full_train_dataset.scaler_X, full_train_dataset.scaler_y
        scaler_path = os.path.join(config.result_dir, config.scaler_save_dir, f"{group_label}_scalers.pkl")
        joblib.dump((scaler_X, scaler_y), scaler_path)
        
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_files)):
            print(f"\n--- {group_label}, Fold {fold+1}/4 ---")
            fold_train_files = [train_val_files[i] for i in train_idx]
            fold_val_files = [train_val_files[i] for i in val_idx]
            
            train_dataset = TimeSeriesDataset(fold_train_files, scaler_X=scaler_X, scaler_y=scaler_y)
            val_dataset = TimeSeriesDataset(fold_val_files, scaler_X=scaler_X, scaler_y=scaler_y)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
            
            model = ProbLSTMModel().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
            
            best_val_loss = float('inf')
            for epoch in range(config.epochs):
                model.train()
                for batch_x, batch_y, batch_mask, _ in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y_pred = batch_y[:, config.input_steps:config.input_steps+config.pred_steps].to(device)
                    mask_tensor = batch_mask[:, config.input_steps:config.input_steps+config.pred_steps].to(device).unsqueeze(-1)
                    optimizer.zero_grad()
                    pred_distribution = model(batch_x)
                    nll_loss = -pred_distribution.log_prob(batch_y_pred)
                    loss = (nll_loss * mask_tensor).sum() / mask_tensor.sum()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_x, val_y, val_mask, _ in val_loader:
                        val_x, val_y_pred = val_x.to(device), val_y[:, config.input_steps:config.input_steps+config.pred_steps].to(device)
                        mask_tensor = val_mask[:, config.input_steps:config.input_steps+config.pred_steps].to(device).unsqueeze(-1)
                        pred_distribution = model(val_x)
                        nll_loss = -pred_distribution.log_prob(val_y_pred)
                        loss = (nll_loss * mask_tensor).sum() / mask_tensor.sum()
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                
                val_loss_logs.append({"Group": group_label, "Fold": fold+1, "Epoch": epoch, "Val_Loss": avg_val_loss})
                lr_scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_save_path = os.path.join(config.result_dir, config.model_save_dir, f"{group_label}_fold{fold+1}_best_prob.pth")
                    torch.save(model.state_dict(), model_save_path)

            best_model = ProbLSTMModel().to(device)
            best_model.load_state_dict(torch.load(model_save_path))
            fold_models.append(best_model)
        
        test_dataset = TimeSeriesDataset(test_files, scaler_X=scaler_X, scaler_y=scaler_y)
        group_metrics = test_and_visualize_prob(fold_models, test_dataset, group_label, scaler_X, scaler_y)
        all_metrics.extend(group_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(config.result_dir, config.metrics_save_path), index=False)
    
    if not metrics_df.empty:
        avg_rmse = metrics_df['RMSE'].mean()
        avg_mae = metrics_df['MAE'].mean()
        avg_ci = metrics_df['Avg_CI_Width'].mean()
        avg_coverage = metrics_df['Coverage'].mean()

        print("\n" + "="*50)
        print("Final Average Metrics Across All 5 Test Groups")
        print(f"Final Average RMSE: {avg_rmse:.4f}")
        print(f"Final Average MAE: {avg_mae:.4f}")
        print(f"Final Average CI Width: {avg_ci:.4f}")
        print(f"Final Average Coverage: {avg_coverage:.4f}")
        print("="*50)
    else:
        print("No metrics were generated. Check data and configuration.")

    pd.DataFrame(val_loss_logs).to_csv(os.path.join(config.result_dir, "val_loss_log_prob.csv"), index=False)


def load_models_and_test(group_label, test_files=None):
    
    scaler_path = os.path.join(config.result_dir, config.scaler_save_dir, f"{group_label}_scalers.pkl")
    if not os.path.exists(scaler_path): 
        print(f"Scaler not found at {scaler_path}")
        return

    scaler_X, scaler_y = joblib.load(scaler_path)
    
    if test_files is None:
        group_num = int(group_label.replace("Group", "")) 
        
        start_idx = (group_num - 1) * 4
        test_files = [os.path.join(config.data_dir, f"data{i}.csv") for i in range(start_idx + 1, start_idx + 5)]
    
    models = []
    for fold in range(1, 5): 
        model_path = os.path.join(config.result_dir, config.model_save_dir, f"{group_label}_fold{fold}_best_prob.pth")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, skipping.")
            continue
        
        model = ProbLSTMModel().to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    
    if not models:
        print(f"No models found for {group_label}. Aborting test.")
        return
    
    print(f"Testing {group_label} with {len(models)} loaded models.")
    test_dataset = TimeSeriesDataset(test_files, scaler_X=scaler_X, scaler_y=scaler_y)
    metrics = test_and_visualize_prob(models, test_dataset, group_label, scaler_X, scaler_y)
    
    if not metrics:
        print("Metric calculation returned empty.")
        return
        
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df[['Sample', 'RMSE', 'MAE', 'Avg_CI_Width', 'Coverage']])
    
    avg_rmse = metrics_df['RMSE'].mean()
    avg_mae = metrics_df['MAE'].mean()
    avg_ci = metrics_df['Avg_CI_Width'].mean()
    avg_coverage = metrics_df['Coverage'].mean()
    
    print("\n" + "="*50)
    print(f"Group {group_label} Average RMSE: {avg_rmse:.4f}")
    print(f"Group {group_label} Average MAE: {avg_mae:.4f}")
    print(f"Group {group_label} Average CI Width: {avg_ci:.4f}")
    print(f"Group {group_label} Average Coverage: {avg_coverage:.4f}")
    print("="*50)
    
    return metrics

if __name__ == "__main__":
    rolling_cross_validation()