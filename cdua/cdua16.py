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
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)
fixed_generator = torch.Generator()
fixed_generator.manual_seed(SEED)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Config:
    total_steps = 120
    input_steps = 16
    pred_steps = 104
    num_features = 10
    num_targets = 1
    timesteps = 700
    beta_schedule = 'linear'
    batch_size = 16
    epochs = 1000
    lr = 1e-3
    min_lr = 1e-5
    gradient_clip_val = 1.0
    data_dir = "../finalData"
    result_dir = "./cdua16"
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

        for file_path in files:
            df = pd.read_csv(file_path)
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

class BetaScheduler:
    def __init__(self, timesteps, schedule='linear'):
        self.timesteps = timesteps
        if schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif schedule == 'cosine':
            s = 0.008
            steps = torch.arange(timesteps + 1, dtype=torch.float32)
            f = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
            self.betas = torch.clip(1 - f[1:] / f[:-1], 0, 0.999)
        else:
            raise ValueError(f"Unsupported beta schedule: {schedule}")

        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)
        self._move_to_device(device)

    def _move_to_device(self, dev):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(dev))

    def add_noise(self, original, t):
        noise = torch.randn_like(original, device=original.device)
        t = t.to(original.device)
        sqrt_alpha_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_t * original + sqrt_one_minus_alpha_t * noise, noise

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1) if half_dim > 1 else 0
        emb = torch.exp(-emb_factor * torch.arange(half_dim, device=t.device).float())
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        if self.is_res and not self.same_channels:
            self.shortcut_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut_conv = None

    def forward(self, x):
        if self.is_res:
            x_res = x
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if not self.same_channels and self.shortcut_conv is not None:
                x_res = self.shortcut_conv(x_res)
            out = x_res + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels, is_res=True),
            ResidualConvBlock(out_channels, out_channels, is_res=True)
        )
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv_block(x)
        x_down = self.downsample(x)
        return x_down, x

class UnetUp(nn.Module):
    def __init__(self, in_channels_skip, in_channels_up, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels_up, out_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            ResidualConvBlock(out_channels + in_channels_skip, out_channels, is_res=True),
            ResidualConvBlock(out_channels, out_channels, is_res=True),
        )

    def forward(self, x_up, x_skip):
        x_up = self.upsample(x_up)
        if x_up.shape[-1] != x_skip.shape[-1]:
            x_up = F.interpolate(x_up, size=x_skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat((x_up, x_skip), dim=1)
        x = self.conv_block(x)
        return x

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = 2 * math.ceil(channels / 2)
        self.inv_freq = 1.0 / torch.pow(10000, torch.arange(0, self.channels, 2).float() / self.channels)

    def get_emb(self, sin_inp):
        emb = torch.stack((torch.sin(sin_inp), torch.cos(sin_inp)), dim=-1)
        return emb.view(*emb.shape[:-2], -1)

    def forward(self, x_input_tensor):
        if len(x_input_tensor.shape) != 3:
            raise RuntimeError("Input tensor must be 3D (Batch, Channels, SeqLen)")
        batch_size, _, seq_len = x_input_tensor.shape
        pos = torch.arange(seq_len, dtype=torch.float32, device=x_input_tensor.device)
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq.to(x_input_tensor.device))
        emb = self.get_emb(sin_inp)
        penc = emb.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
        return penc

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        num_groups = 8 if channels >= 8 and channels % 8 == 0 else max(1, channels // 4)
        self.group_norm = nn.GroupNorm(num_groups, channels)
        self.mha = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        residual = x
        x_norm = self.group_norm(x)
        x_norm = x_norm.permute(0, 2, 1)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        attn_out = attn_out.permute(0, 2, 1)
        return residual + attn_out

class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads,
                                         kdim=context_dim, vdim=context_dim, batch_first=True)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x_query, context):
        residual = x_query
        x_query_norm = self.norm(x_query.permute(0, 2, 1)).permute(0, 2, 1)
        attn_out, _ = self.mha(x_query.permute(0, 2, 1), context.permute(0, 2, 1), context.permute(0, 2, 1))
        return residual + attn_out.permute(0, 2, 1)

class ContextUnet(nn.Module):
    def __init__(self, in_channels_x, n_feat=64, pos_enc_dim=16):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.pos_encoder = PositionalEncoding1D(channels=self.pos_enc_dim)
        self.init_conv = ResidualConvBlock(in_channels_x + self.pos_enc_dim, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.attn1 = SelfAttentionBlock(n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.attn2 = SelfAttentionBlock(2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)
        self.attn3 = SelfAttentionBlock(4 * n_feat)
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.GELU())
        up0_kernel_size = max(1, config.input_steps // 8)
        self.up0_transpose = nn.ConvTranspose1d(4 * n_feat, 4 * n_feat, kernel_size=up0_kernel_size)
        self.up0_resblock = ResidualConvBlock(4 * n_feat, 4*n_feat, is_res=True)
        self.up1 = UnetUp(in_channels_skip=4 * n_feat, in_channels_up=4 * n_feat, out_channels=2 * n_feat)
        self.up2 = UnetUp(in_channels_skip=2 * n_feat, in_channels_up=2 * n_feat, out_channels=n_feat)
        self.up3 = UnetUp(in_channels_skip=n_feat, in_channels_up=n_feat, out_channels=n_feat)
        num_groups_out = 8 if n_feat >= 8 and n_feat % 8 == 0 else max(1, n_feat // 4)
        self.out_conv = nn.Conv1d(n_feat + (in_channels_x + self.pos_enc_dim), n_feat, kernel_size=3, padding=1)
        self.out_norm_act = nn.Sequential(nn.GroupNorm(num_groups_out, n_feat), nn.GELU())
        self.final_conv = nn.Conv1d(n_feat, n_feat, kernel_size=1)

    def forward(self, x_cond):
        pos_enc = self.pos_encoder(x_cond)
        x_with_pos = torch.cat([x_cond, pos_enc], dim=1)
        x0 = self.init_conv(x_with_pos)

        d1_down, d1_skip = self.down1(x0)
        d1_attn = self.attn1(d1_down)
        d2_down, d2_skip = self.down2(d1_attn)
        d2_attn = self.attn2(d2_down)
        d3_down, d3_skip = self.down3(d2_attn)
        d3_attn = self.attn3(d3_down)
        
        bottleneck_vec = self.to_vec(d3_attn)
        up0_t = self.up0_transpose(bottleneck_vec)
        target_len_d3 = d3_skip.shape[-1]
        if up0_t.shape[-1] != target_len_d3:
            up0_t = F.interpolate(up0_t, size=target_len_d3, mode='linear', align_corners=False)
        up0 = self.up0_resblock(up0_t)

        u1 = self.up1(up0, d3_skip)
        target_len_d2 = d2_skip.shape[-1]
        if u1.shape[-1] != target_len_d2:
            u1 = F.interpolate(u1, size=target_len_d2, mode='linear', align_corners=False)
        u2 = self.up2(u1, d2_skip)
        
        target_len_d1 = d1_skip.shape[-1]
        if u2.shape[-1] != target_len_d1:
            u2 = F.interpolate(u2, size=target_len_d1, mode='linear', align_corners=False)
        u3 = self.up3(u2, d1_skip)

        if u3.shape[-1] != x_with_pos.shape[-1]:
            u3 = F.interpolate(u3, size=x_with_pos.shape[-1], mode='linear', align_corners=False)

        out = torch.cat((u3, x_with_pos), dim=1)
        out = self.out_conv(out)
        out = self.out_norm_act(out)
        out = self.final_conv(out)
        return out

class NoisePredictor(nn.Module):
    def __init__(self, noisy_signal_channels=1, time_embed_dim=64,
                 hidden_channels=128, out_channels=1, num_res_blocks=4, context_feature_dim=64):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.context_feature_dim = context_feature_dim
        self.initial_conv = nn.Conv1d(self.context_feature_dim + noisy_signal_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([
            ResidualConvBlock(hidden_channels, hidden_channels, is_res=True)
            for _ in range(num_res_blocks)
        ])
        self.cross_attn = CrossAttentionBlock(query_dim=hidden_channels, context_dim=self.context_feature_dim)
        self.final_conv = nn.Conv1d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, noisy_y_seq, final_context_features, time_embedding_original):
        combined_input = torch.cat([final_context_features, noisy_y_seq], dim=1)
        x = self.initial_conv(combined_input)
        for block in self.blocks:
            x = block(x)
        x = self.cross_attn(x, final_context_features)
        return self.final_conv(x)

class ConditionalDiffusion(nn.Module):
    def __init__(self, context_unet_out_features=64, time_embedding_dim=64):
        super().__init__()
        self.context_unet_out_features = context_unet_out_features
        self.time_embedding_dim = time_embedding_dim
        self.context_unet = ContextUnet(in_channels_x=config.num_features, n_feat=self.context_unet_out_features, pos_enc_dim=16)
        self.time_embed = TimeEmbedding(self.time_embedding_dim)
        if self.time_embedding_dim != self.context_unet_out_features:
            self.time_feature_projection = nn.Linear(self.time_embedding_dim, self.context_unet_out_features)
        else:
            self.time_feature_projection = nn.Identity()
        self.noise_pred = NoisePredictor(
            noisy_signal_channels=1,
            time_embed_dim=self.time_embedding_dim,
            hidden_channels=128,
            out_channels=1,
            num_res_blocks=4,
            context_feature_dim=self.context_unet_out_features
        )

    def forward(self, noisy_y, t, cond_x):
        cond_x_permuted = cond_x.to(device).permute(0, 2, 1)
        context_features_map = self.context_unet(cond_x_permuted)
        t_embed = self.time_embed(t)
        context_features_interpolated = F.interpolate(context_features_map, size=config.total_steps, mode='linear', align_corners=False)
        projected_t_embed = self.time_feature_projection(t_embed)
        projected_t_embed_expanded = projected_t_embed.unsqueeze(-1).expand(-1, -1, config.total_steps)
        final_context_features = context_features_interpolated + projected_t_embed_expanded
        noisy_y_permuted = noisy_y.to(device).permute(0, 2, 1)
        out = self.noise_pred(noisy_y_permuted, final_context_features, t_embed)
        return out.permute(0, 2, 1)

def sample(model, cond_x, mask, beta_scheduler_instance):
    model.eval()
    batch_size = cond_x.shape[0]
    y = torch.randn((batch_size, config.total_steps, 1), device=device)

    for t_val in reversed(range(config.timesteps)):
        t_tensor = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
        with torch.no_grad():
            pred_noise = model(y, t_tensor, cond_x)
        alpha_t = beta_scheduler_instance.alphas[t_val].view(-1, 1, 1)
        beta_t = beta_scheduler_instance.betas[t_val].view(-1, 1, 1)
        sqrt_recip_alpha_t = torch.rsqrt(alpha_t)
        sqrt_one_minus_alpha_cumprod_t = beta_scheduler_instance.sqrt_one_minus_alpha_cumprod[t_val].view(-1,1,1)
        y = sqrt_recip_alpha_t * (y - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise)
        if t_val > 0:
            noise_z = torch.randn_like(y)
            y += torch.sqrt(beta_t) * noise_z
    return y.cpu()

def multi_sample(model, cond_x, mask, beta_scheduler_instance, n_times=20):
    samples_list = []
    for _ in range(n_times):
        pred = sample(model, cond_x, mask, beta_scheduler_instance)
        samples_list.append(pred.numpy())
    samples_np = np.stack(samples_list, axis=0)
    avg_samples = np.mean(samples_np, axis=0)
    std_samples = np.std(samples_np, axis=0)
    return avg_samples, std_samples

def test_and_visualize(models_ensemble, test_dataset, test_group_label, beta_scheduler_instance, scaler_X, scaler_y, smooth_predictions=False, smoothing_window=5, avg_over_steps=3):
    test_dataset.scaler_X = scaler_X
    test_dataset.scaler_y = scaler_y
    
    all_fold_avg_preds_list, all_fold_std_preds_list = [], []
    
    for model_fold in models_ensemble:
        model_fold.eval()
        current_fold_avg_preds, current_fold_std_preds = [], []
        temp_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        with torch.no_grad():
            for x, _, mask, _ in temp_loader:
                x = x.to(device)
                avg_pred_batch, std_pred_batch = multi_sample(model_fold, x, mask, beta_scheduler_instance, n_times=40)
                
                num_samples_in_batch = avg_pred_batch.shape[0]
                avg_pred_batch_inv = scaler_y.inverse_transform(avg_pred_batch.reshape(-1, 1)).reshape(num_samples_in_batch, config.total_steps, 1)
                
                scale_factor_y = 1 / scaler_y.scale_[0] if scaler_y.scale_[0] != 0 else 1
                std_pred_batch_orig_scale = std_pred_batch * scale_factor_y

                current_fold_avg_preds.append(avg_pred_batch_inv)
                current_fold_std_preds.append(std_pred_batch_orig_scale)
        
        all_fold_avg_preds_list.append(np.concatenate(current_fold_avg_preds, axis=0))
        all_fold_std_preds_list.append(np.concatenate(current_fold_std_preds, axis=0))

    stacked_avg_preds = np.stack(all_fold_avg_preds_list, axis=0)
    stacked_std_preds = np.stack(all_fold_std_preds_list, axis=0)
    ensemble_avg = np.mean(stacked_avg_preds, axis=0)
    mean_of_variances = np.mean(np.square(stacked_std_preds), axis=0)
    variance_of_means = np.var(stacked_avg_preds, axis=0)
    ensemble_std = np.sqrt(mean_of_variances + variance_of_means)

    results_for_df, mse_values, rmse_values, mae_values, ci_width_values, coverage_values = [], [], [], [], [], []
    z_score = 1.96
    num_samples = len(test_dataset)
    
    num_plots = min(num_samples, 4)
    plt.figure(figsize=(20, 6 * num_plots))
    
    for idx in range(num_samples):
        mse, rmse, mae, ci_width, coverage = [np.nan] * 5

        true_y_scaled = test_dataset.y[idx]
        mask_sample = test_dataset.masks[idx].astype(bool)
        true_y_inv = np.full((config.total_steps, 1), np.nan)
        if np.any(mask_sample):
            true_y_inv[mask_sample] = scaler_y.inverse_transform(true_y_scaled[mask_sample])
        true_y_inv_full = true_y_inv.squeeze()
        
        pred_y_sample = ensemble_avg[idx].squeeze()
        std_y_sample = ensemble_std[idx].squeeze()
        
        original_sample_len = test_dataset.original_lengths[idx]
        pred_start_idx = config.input_steps
        pred_end_idx = min(original_sample_len, config.total_steps)
        
        if smooth_predictions and (pred_end_idx - pred_start_idx > smoothing_window):
            smoothing_start_idx = max(pred_start_idx, pred_end_idx - smoothing_window)
            for t_step in range(smoothing_start_idx, pred_end_idx):
                avg_start_idx = max(pred_start_idx, t_step - avg_over_steps)
                if avg_start_idx < t_step:
                    std_y_sample[t_step] = np.mean(std_y_sample[avg_start_idx:t_step])

        lower_bound = pred_y_sample - z_score * std_y_sample
        upper_bound = pred_y_sample + z_score * std_y_sample
        
        if pred_start_idx < pred_end_idx:
            true_valid = true_y_inv_full[pred_start_idx:pred_end_idx]
            pred_valid = pred_y_sample[pred_start_idx:pred_end_idx]
            std_valid = std_y_sample[pred_start_idx:pred_end_idx]
            valid_indices = ~np.isnan(true_valid)
            true_valid = true_valid[valid_indices]
            pred_valid = pred_valid[valid_indices]
            std_valid = std_valid[valid_indices]

            if len(true_valid) > 0:
                mse = mean_squared_error(true_valid, pred_valid)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(true_valid, pred_valid)
                
                lower_ci = pred_valid - z_score * std_valid
                upper_ci = pred_valid + z_score * std_valid
                ci_width = np.mean(upper_ci - lower_ci)
                coverage = np.mean((true_valid >= lower_ci) & (true_valid <= upper_ci))
        
        mse_values.append(mse)
        rmse_values.append(rmse)
        mae_values.append(mae)
        ci_width_values.append(ci_width)
        coverage_values.append(coverage)

        for t_step in range(config.total_steps):
            results_for_df.append({
                'sample_id': idx, 'time_step': t_step,
                'true': true_y_inv_full[t_step], 'pred': pred_y_sample[t_step],
                'lower': lower_bound[t_step], 'upper': upper_bound[t_step]
            })
        
        if idx < num_plots:
            plt.subplot(num_plots, 1, idx + 1)
            plt.plot(np.arange(original_sample_len), true_y_inv_full[:original_sample_len], label='True Value', color='blue')
            
            if pred_start_idx < pred_end_idx:
                time_axis_pred = np.arange(pred_start_idx, pred_end_idx)
                plt.plot(time_axis_pred, pred_y_sample[pred_start_idx:pred_end_idx], '--', label='Prediction', color='red')
                ci_label = 'Smoothed 95% CI' if smooth_predictions else '95% CI'
                plt.fill_between(time_axis_pred, 
                                 lower_bound[pred_start_idx:pred_end_idx], 
                                 upper_bound[pred_start_idx:pred_end_idx], 
                                 alpha=0.3, label=ci_label, color='orange')
            
            plt.axvline(x=config.input_steps, linestyle='--', color='gray', label='Prediction Start')
            title_str = f'Sample {idx}'
            if not np.isnan(rmse):
                title_str += f' | RMSE: {rmse:.2f}, MAE: {mae:.2f}, CI: {ci_width:.2f}, Cov: {coverage:.2%}'
            plt.title(title_str)
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(config.result_dir, f"{test_group_label}_results_smoothed_CI.png" if smooth_predictions else f"{test_group_label}_results.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Visualization plots saved to: {plot_path}")

    infer_csv_path = os.path.join(config.result_dir, f"{test_group_label}_inference_results.csv")
    pd.DataFrame(results_for_df).to_csv(infer_csv_path, index=False)
    print(f"Inference results saved to: {infer_csv_path}")

    metrics = [{'Test_Group': test_group_label, 'Sample': i, 'MSE': mse_values[i], 'RMSE': rmse_values[i], 'MAE': mae_values[i], 'CI_Width': ci_width_values[i], 'Coverage': coverage_values[i]} for i in range(num_samples)]
    return metrics

def train_group(group, group_label, beta_scheduler_instance):
    print(f"\n=== Training {group_label} ===")
    fold_models = []
    
    full_train_dataset = TimeSeriesDataset(group['train_val'])
    scaler_X, scaler_y = full_train_dataset.scaler_X, full_train_dataset.scaler_y
    
    scaler_path = os.path.join(config.result_dir, config.scaler_save_dir, f"{group_label}_scalers.pkl")
    joblib.dump((scaler_X, scaler_y), scaler_path)
    print(f"Scalers saved to {scaler_path}")
    
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(group['train_val'])):
        print(f"  Fold {fold+1}/4")
        fold_train_files = [group['train_val'][i] for i in train_idx]
        fold_val_files = [group['train_val'][i] for i in val_idx]
        
        train_dataset = TimeSeriesDataset(fold_train_files, scaler_X=scaler_X, scaler_y=scaler_y)
        val_dataset = TimeSeriesDataset(fold_val_files, scaler_X=scaler_X, scaler_y=scaler_y)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=fixed_generator)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        
        model = ConditionalDiffusion().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=config.min_lr)
        
        best_val_loss = float('inf')
        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            for batch_x, batch_y, batch_mask, _ in train_loader:
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                t = torch.randint(0, config.timesteps, (batch_y.size(0),), device=device)
                noisy_y, noise = beta_scheduler_instance.add_noise(batch_y, t)
                
                optimizer.zero_grad()
                pred_noise = model(noisy_y, t, batch_x)
                mask_expanded = batch_mask.unsqueeze(-1)
                loss = F.mse_loss(pred_noise * mask_expanded, noise * mask_expanded)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_x, val_y, val_mask, _ in val_loader:
                    val_x, val_y, val_mask = val_x.to(device), val_y.to(device), val_mask.to(device)
                    t_val = torch.randint(0, config.timesteps, (val_y.size(0),), device=device)
                    noisy_y_val, noise_val = beta_scheduler_instance.add_noise(val_y, t_val)
                    pred_noise_val = model(noisy_y_val, t_val, val_x)
                    mask_val_expanded = val_mask.unsqueeze(-1)
                    loss_val = F.mse_loss(pred_noise_val * mask_val_expanded, noise_val * mask_val_expanded)
                    val_loss += loss_val.item()
            
            avg_val_loss = val_loss / len(val_loader)
            lr_scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = os.path.join(config.result_dir, config.model_save_dir, f"{group_label}_fold{fold+1}_best.pth")
                torch.save(model.state_dict(), model_save_path)
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Best Val: {best_val_loss:.6f} (Saved)")

        best_model = ConditionalDiffusion().to(device)
        best_model.load_state_dict(torch.load(model_save_path, map_location=device))
        fold_models.append(best_model)
    
    return fold_models, scaler_X, scaler_y

def rolling_cross_validation(beta_scheduler_instance):
    all_metrics = []
    all_files = [os.path.join(config.data_dir, f"data{i}.csv") for i in range(1, 21)]
    groups_of_files = [all_files[i:i + 4] for i in range(0, len(all_files), 4)]
    
    for i in range(len(groups_of_files)):
        test_files = groups_of_files[i]
        train_val_files = [f for j, g in enumerate(groups_of_files) if i != j for f in g]
        group_label = f"Group{i+1}"
        group_definition = {"train_val": train_val_files, "test": test_files, "label": group_label}
        
        fold_models, scaler_X, scaler_y = train_group(group_definition, group_label, beta_scheduler_instance)
        
        test_dataset = TimeSeriesDataset(group_definition['test'], scaler_X=scaler_X, scaler_y=scaler_y)
        group_metrics = test_and_visualize(fold_models, test_dataset, group_label, beta_scheduler_instance, scaler_X, scaler_y, smooth_predictions=True)
        all_metrics.extend(group_metrics)
    
    metrics_df = pd.DataFrame(all_metrics).dropna()
    metrics_path = os.path.join(config.result_dir, config.metrics_save_path)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nAll metrics from training run saved to {metrics_path}")
    
    if not metrics_df.empty:
        avg_rmse = metrics_df['RMSE'].mean()
        avg_mae = metrics_df['MAE'].mean()
        avg_ci = metrics_df['CI_Width'].mean()
        avg_cov = metrics_df['Coverage'].mean()
        
        print("\n" + "="*50)
        print(f"Final Avg Results Across {len(metrics_df)} Test Samples")
        print("="*50)
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average MAE: {avg_mae:.4f}")
        print(f"Average CI Width: {avg_ci:.3f}")
        print(f"Average Coverage: {avg_cov:.3f}")
        print("="*50)
    else:
        print("\nNo valid metrics were generated.")

def load_models_and_test(group_label, test_files=None, beta_scheduler_instance=None):
    scaler_path = os.path.join(config.result_dir, config.scaler_save_dir, f"{group_label}_scalers.pkl")
    if not os.path.exists(scaler_path):
        print(f"Error: Scalers not found for {group_label} at {scaler_path}")
        return None
    
    scaler_X, scaler_y = joblib.load(scaler_path)
    print(f"Loaded scalers for {group_label}")
    
    if test_files is None:
        group_num = int(group_label.replace("Group", ""))
        start_idx = (group_num - 1) * 4
        test_files = [os.path.join(config.data_dir, f"data{i}.csv") for i in range(start_idx + 1, start_idx + 5)]
    
    models = []
    for fold in range(1, 5):
        model_path = os.path.join(config.result_dir, config.model_save_dir, f"{group_label}_fold{fold}_best.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model not found for {group_label} fold {fold}")
            continue
        
        model = ConditionalDiffusion().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"Loaded model: {model_path}")
    
    if not models:
        print("Error: No models loaded for this group.")
        return None
      
    test_dataset = TimeSeriesDataset(test_files, scaler_X=scaler_X, scaler_y=scaler_y)
    
    if beta_scheduler_instance is None:
        beta_scheduler_instance = BetaScheduler(config.timesteps, config.beta_schedule)
    
    metrics = test_and_visualize(models, test_dataset, group_label, beta_scheduler_instance, scaler_X, scaler_y, smooth_predictions=True)
    metrics_df = pd.DataFrame(metrics).dropna()
    
    print(f"\nTest results for {group_label}:")
    print(metrics_df[['Sample', 'RMSE', 'MAE', 'CI_Width', 'Coverage']])
    
    if not metrics_df.empty:
        print("\n" + "="*50)
        print(f"Group {group_label} Summary:")
        print(f"Average RMSE: {metrics_df['RMSE'].mean():.4f}")
        print(f"Average MAE: {metrics_df['MAE'].mean():.4f}")
        print(f"Average CI Width: {metrics_df['CI_Width'].mean():.3f}")
        print(f"Average Coverage: {metrics_df['Coverage'].mean():.3f}")
        print("="*50)

    return metrics

if __name__ == "__main__":
    print("\n--- Starting Rolling Cross-Validation Training and Evaluation ---")
    scheduler_instance = BetaScheduler(config.timesteps, config.beta_schedule)
    rolling_cross_validation(scheduler_instance)
    print("\n--- Rolling Cross-Validation Complete ---")