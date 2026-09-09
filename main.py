"""Methodology-complete FiLM--TimesNet volatility forecasting model.

This implementation follows Draft v3.0: raw-input decomposition, a fixed
Legendre memory recurrence followed by a frequency-enhanced layer (FEL),
top-k multi-period TimesNet modelling, cross-scale guidance (CSG), full-map
scale attention (SA), and the MSE--MADL hybrid objective.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================== Configuration ==============================
FILE_PATH = r"E:\论文\Fund_Dataset-20251025T124416Z-1-001\纳斯达克100.xlsx"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "film_timesnet_csg_sa_madl_output.csv")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
SEQ_LEN = 10
PRED_LEN = 1
BATCH_SIZE = 16
D_MODEL = 64
NUM_LAYERS = 2
DROPOUT = 0.1
LR = 1e-4
EPOCHS = 100
PATIENCE = 15
LAMBDA_MADL = 0.5
DECOMP_KERNEL = 5
NUM_LEGENDRE = 64
FEL_MODES = 3
TIMES_TOP_K = 2
NUM_INCEPTION_KERNELS = 3
SA_HIDDEN_DIM = 32
MADL_TEMPERATURE = 10.0
ANNUALIZATION = 252.0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================= Data preparation ============================
def load_and_process_data(filepath: str, annualization: float = ANNUALIZATION) -> pd.DataFrame:
    """Load prices and construct the two input variables used in the paper."""
    df = pd.read_excel(filepath, sheet_name="Sheet1")
    required_cols = ["Trddt", "Clsidx"]
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Trddt"] = pd.to_datetime(df["Trddt"])
    df = df.sort_values("Trddt").reset_index(drop=True)
    eps = 1e-8
    df["log_ret"] = np.log(df["Clsidx"] + eps).diff()
    df["rv"] = df["log_ret"].rolling(window=7).var() * annualization
    df = df.loc[df["rv"] > eps].copy()

    df["log_rv"] = np.log(df["rv"])
    # Volatility is sqrt(RV); therefore log-volatility is 0.5 * log(RV).
    df["log_vol"] = np.log(np.sqrt(df["rv"]))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df[["Trddt", "log_rv", "log_vol"]].reset_index(drop=True)


def create_sequences(data: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, ...]:
    """Create one-step targets and retain their row indices for chronological splits."""
    features = data[["log_rv", "log_vol"]].to_numpy(dtype=np.float32)
    targets = data["log_rv"].to_numpy(dtype=np.float32)
    dates = data["Trddt"].to_numpy()
    xs, ys, target_dates, target_indices = [], [], [], []
    for target_index in range(seq_len, len(data)):
        xs.append(features[target_index - seq_len : target_index])
        ys.append(targets[target_index])
        target_dates.append(dates[target_index])
        target_indices.append(target_index)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(target_dates),
        np.asarray(target_indices, dtype=np.int64),
    )


def prepare_splits(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[Dict[str, Tuple[np.ndarray, ...]], StandardScaler, Tuple[int, int]]:
    """Fit scaling on training rows only and split sequences by target date.

    Validation and test inputs may use preceding observations, while their
    targets remain strictly inside the corresponding chronological partition.
    """
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one")

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    if train_end <= seq_len or val_end <= train_end or val_end >= n_rows:
        raise ValueError("Dataset is too short for the requested sequence length and splits")

    feature_cols = ["log_rv", "log_vol"]
    scaler = StandardScaler().fit(df.loc[: train_end - 1, feature_cols])
    scaled_df = df.copy()
    scaled_df[feature_cols] = scaler.transform(df[feature_cols])
    x_all, y_all, dates_all, indices = create_sequences(scaled_df, seq_len)

    masks = {
        "train": indices < train_end,
        "val": (indices >= train_end) & (indices < val_end),
        "test": indices >= val_end,
    }
    splits = {
        name: (x_all[mask], y_all[mask], dates_all[mask])
        for name, mask in masks.items()
    }
    return splits, scaler, (train_end, val_end)


# ============================== Model modules ===============================
class MovingAverage(nn.Module):
    """Average pooling with replicated endpoint padding."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("decomposition kernel size must be odd")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.avg(padded.transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """X_trend = AvgPool(Padding(X)); X_seasonal = X - X_trend."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class FrequencyEnhancedLayer(nn.Module):
    """Learn a complex transfer function on low-frequency memory modes."""

    def __init__(self, num_legendre: int, modes: int, dropout: float) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / max(1, num_legendre)
        self.weight_real = nn.Parameter(scale * torch.randn(modes, num_legendre))
        self.weight_imag = nn.Parameter(scale * torch.randn(modes, num_legendre))
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        # memory: [batch, length, d_model, num_legendre]
        length = memory.size(1)
        spectrum = torch.fft.rfft(memory, dim=1)
        enhanced = torch.zeros_like(spectrum)
        modes = min(self.modes, spectrum.size(1))
        weight = torch.complex(
            self.weight_real[:modes], self.weight_imag[:modes]
        ).unsqueeze(0).unsqueeze(2)
        enhanced[:, :modes] = spectrum[:, :modes] * weight
        output = torch.fft.irfft(enhanced, n=length, dim=1)
        return self.dropout(output)


class FiLMEncoder(nn.Module):
    """Fixed Legendre recurrence followed by the methodology's FEL readout."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_legendre: int = NUM_LEGENDRE,
        fel_modes: int = FEL_MODES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.num_legendre = num_legendre
        transition = torch.zeros(num_legendre, num_legendre)
        input_vector = torch.zeros(num_legendre)
        for n in range(num_legendre):
            input_vector[n] = (2 * n + 1) * ((-1) ** n)
            for k in range(num_legendre):
                transition[n, k] = (2 * n + 1) * (
                    ((-1) ** (n - k)) if k <= n else 1
                )

        # Fixed numerical scaling retains the matrix pattern while keeping the
        # short discrete recurrence stable. These tensors are not trainable.
        self.register_buffer("A", transition.float() / num_legendre)
        self.register_buffer("B", input_vector.float() / num_legendre)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.fel = FrequencyEnhancedLayer(num_legendre, fel_modes, dropout)
        self.memory_readout = nn.Linear(num_legendre, 1)
        self.output_projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(x)
        batch, length, d_model = embedded.shape
        state = embedded.new_zeros(batch, d_model, self.num_legendre)
        states = []
        for t in range(length):
            state = torch.matmul(state, self.A.transpose(0, 1))
            state = state + embedded[:, t, :, None] * self.B[None, None, :]
            states.append(state)
        memory = torch.stack(states, dim=1)
        enhanced = self.fel(memory)
        encoded = self.memory_readout(enhanced).squeeze(-1)
        return self.norm(self.output_projection(encoded))


class InceptionBlock2D(nn.Module):
    """Parallel 2-D kernels used on each period-folded representation."""

    def __init__(self, d_model: int, num_kernels: int = NUM_INCEPTION_KERNELS) -> None:
        super().__init__()
        kernel_sizes = [2 * index + 1 for index in range(num_kernels)]
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(
                    d_model,
                    d_model,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
                for kernel in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([conv(x) for conv in self.convolutions], dim=-1).mean(dim=-1)


class TimesBlock(nn.Module):
    """FFT top-k period discovery, 2-D modelling and spectral aggregation."""

    def __init__(self, d_model: int, top_k: int, num_kernels: int, dropout: float) -> None:
        super().__init__()
        self.top_k = top_k
        self.inception = nn.Sequential(
            InceptionBlock2D(d_model, num_kernels),
            nn.GELU(),
            InceptionBlock2D(d_model, num_kernels),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = x.shape
        spectrum = torch.fft.rfft(x, dim=1)
        amplitude = spectrum.abs().mean(dim=-1)  # [batch, frequency]
        if amplitude.size(1) <= 1:
            return self.norm(x)

        global_amplitude = amplitude.mean(dim=0).clone()
        global_amplitude[0] = -torch.inf
        top_k = min(self.top_k, global_amplitude.numel() - 1)
        frequency_indices = torch.topk(global_amplitude, top_k).indices
        period_outputs = []
        for frequency_index in frequency_indices.tolist():
            period = max(1, length // frequency_index)
            padded_length = ((length + period - 1) // period) * period
            if padded_length > length:
                padded = F.pad(x, (0, 0, 0, padded_length - length))
            else:
                padded = x
            folded = padded.reshape(batch, padded_length // period, period, d_model)
            folded = folded.permute(0, 3, 1, 2).contiguous()
            processed = self.inception(folded)
            restored = processed.permute(0, 2, 3, 1).reshape(batch, padded_length, d_model)
            period_outputs.append(restored[:, :length])

        stacked = torch.stack(period_outputs, dim=-1)
        selected_amplitude = amplitude[:, frequency_indices]
        weights = torch.softmax(selected_amplitude, dim=-1)[:, None, None, :]
        aggregated = (stacked * weights).sum(dim=-1)
        return self.norm(self.dropout(aggregated))


class TimesNetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        top_k: int,
        num_kernels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [TimesBlock(d_model, top_k, num_kernels, dropout) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(x)
        for block in self.blocks:
            encoded = block(encoded)
        return self.output_projection(encoded)


class CrossScaleGuidance(nn.Module):
    """G = sigmoid(Linear(H_trend)); H_seasonal_guided = H_seasonal * G."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(d_model, d_model)

    def forward(
        self, trend: torch.Tensor, seasonal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate_projection(trend))
        return seasonal * gate, gate


class ScaleAttention(nn.Module):
    """Compute two scale weights from complete L x D feature maps."""

    def __init__(self, seq_len: int, d_model: int, hidden_dim: int = SA_HIDDEN_DIM) -> None:
        super().__init__()
        flattened_dim = seq_len * d_model
        self.attention = nn.Sequential(
            nn.Linear(2 * flattened_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, trend: torch.Tensor, guided_seasonal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = trend.size(0)
        combined = torch.cat(
            [trend.reshape(batch, -1), guided_seasonal.reshape(batch, -1)], dim=-1
        )
        weights = self.attention(combined)
        fused = (
            weights[:, 0, None, None] * trend
            + weights[:, 1, None, None] * guided_seasonal
        )
        return fused, weights


class VolatilityMICN(nn.Module):
    """Complete architecture described in the Methodology section."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int = PRED_LEN,
        input_dim: int = 2,
        d_model: int = D_MODEL,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        decomp_kernel: int = DECOMP_KERNEL,
        num_legendre: int = NUM_LEGENDRE,
        fel_modes: int = FEL_MODES,
        times_top_k: int = TIMES_TOP_K,
        num_inception_kernels: int = NUM_INCEPTION_KERNELS,
        sa_hidden_dim: int = SA_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.decomposition = SeriesDecomposition(decomp_kernel)
        self.trend_encoder = FiLMEncoder(
            input_dim, d_model, num_legendre, fel_modes, dropout
        )
        self.seasonal_encoder = TimesNetEncoder(
            input_dim,
            d_model,
            num_layers,
            times_top_k,
            num_inception_kernels,
            dropout,
        )
        self.csg = CrossScaleGuidance(d_model)
        self.scale_attention = ScaleAttention(seq_len, d_model, sa_hidden_dim)
        self.prediction_head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        seasonal_raw, trend_raw = self.decomposition(x)
        trend = self.trend_encoder(trend_raw)
        seasonal = self.seasonal_encoder(seasonal_raw)
        guided_seasonal, gate = self.csg(trend, seasonal)
        fused, scale_weights = self.scale_attention(trend, guided_seasonal)
        prediction = self.prediction_head(fused[:, -1, :])
        if not return_aux:
            return prediction
        return prediction, {
            "trend": trend,
            "seasonal": seasonal,
            "csg_gate": gate,
            "guided_seasonal": guided_seasonal,
            "scale_weights": scale_weights,
            "fused": fused,
        }


# ============================ Loss and metrics ==============================
class MADLLoss(nn.Module):
    """Exact MADL forward value with a straight-through direction gradient."""

    def __init__(self, temperature: float = MADL_TEMPERATURE) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        hard_sign = torch.sign(y_pred)
        soft_sign = torch.tanh(self.temperature * y_pred)
        pred_sign = hard_sign.detach() - soft_sign.detach() + soft_sign
        direction_match = torch.sign(y_true) * pred_sign
        return torch.mean(-direction_match * torch.abs(y_true))


class HybridLoss(nn.Module):
    def __init__(self, lambd: float = LAMBDA_MADL) -> None:
        super().__init__()
        self.lambd = lambd
        self.mse = nn.MSELoss()
        self.madl = MADLLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return (1 - self.lambd) * self.mse(y_pred, y_true) + self.lambd * self.madl(
            y_pred, y_true
        )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {name: float("nan") for name in ["MSE", "RMSE", "MAE", "R2", "MAPE", "DA", "MADL"]}
    mse = mean_squared_error(y_true, y_pred)
    nonzero = y_true != 0
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan"),
        "DA": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "MADL": float(np.mean(-np.sign(y_true * y_pred) * np.abs(y_true))),
    }


def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def validation_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb).squeeze(-1), yb)
            total += loss.item() * xb.size(0)
            count += xb.size(0)
    return total / max(1, count)


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(f"[{label}]")
    print(
        f"  MSE={metrics['MSE']:.6f} | RMSE={metrics['RMSE']:.6f} | "
        f"MAE={metrics['MAE']:.6f} | R2={metrics['R2']:.6f}"
    )
    print(
        f"  MAPE={metrics['MAPE']:.2f}% | DA={metrics['DA']:.4f} | "
        f"MADL={metrics['MADL']:.6f}"
    )


# =================================== Main ===================================
if __name__ == "__main__":
    print(f"Using device: {device}")
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Data file does not exist: {FILE_PATH}")

    df_full = load_and_process_data(FILE_PATH)
    splits, scaler, boundaries = prepare_splits(df_full)
    train_end, val_end = boundaries
    print(
        f"Data range: {df_full['Trddt'].min():%Y-%m-%d} -- "
        f"{df_full['Trddt'].max():%Y-%m-%d} ({len(df_full)} rows)"
    )
    print(
        f"Chronological targets: train={len(splits['train'][1])}, "
        f"validation={len(splits['val'][1])}, test={len(splits['test'][1])}"
    )
    print(
        f"Boundaries: validation starts {df_full.iloc[train_end]['Trddt']:%Y-%m-%d}; "
        f"test starts {df_full.iloc[val_end]['Trddt']:%Y-%m-%d}"
    )

    x_train, y_train, _ = splits["train"]
    x_val, y_val, _ = splits["val"]
    x_test, y_test, test_dates = splits["test"]
    train_loader = make_loader(x_train, y_train, shuffle=True)
    val_loader = make_loader(x_val, y_val, shuffle=False)

    model = VolatilityMICN(seq_len=SEQ_LEN).to(device)
    criterion = HybridLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_validation = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        training_total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            training_total += loss.item() * xb.size(0)
        scheduler.step()

        current_validation = validation_loss(model, val_loader, criterion)
        if current_validation < best_validation - 1e-8:
            best_validation = current_validation
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            training_mean = training_total / len(x_train)
            print(
                f"Epoch {epoch:03d}/{EPOCHS}: train={training_mean:.6f}, "
                f"validation={current_validation:.6f}"
            )
        if stale_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best validation={best_validation:.6f}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction_scaled, auxiliary = model(
            torch.from_numpy(x_test).to(device), return_aux=True
        )
    prediction_scaled = prediction_scaled.cpu().numpy().reshape(-1)
    prediction_scaled = np.nan_to_num(prediction_scaled)

    rv_scale = scaler.scale_[0] if scaler.scale_[0] != 0 else 1.0
    rv_mean = scaler.mean_[0]
    pred_log_rv = prediction_scaled * rv_scale + rv_mean
    true_log_rv = y_test * rv_scale + rv_mean
    pred_rv = np.exp(pred_log_rv)
    true_rv = np.exp(true_log_rv)

    print("=" * 72)
    print_metrics("Log-RV scale", calculate_metrics(true_log_rv, pred_log_rv))
    print_metrics("Original RV scale", calculate_metrics(true_rv, pred_rv))
    print("=" * 72)

    scale_weights = auxiliary["scale_weights"].cpu().numpy()
    gate_mean = auxiliary["csg_gate"].mean(dim=(1, 2)).cpu().numpy()
    results = pd.DataFrame(
        {
            "Pred_Date": test_dates,
            "True_Log_RV": true_log_rv,
            "Pred_Log_RV": pred_log_rv,
            "True_RV": true_rv,
            "Pred_RV": pred_rv,
            "SA_Trend_Weight": scale_weights[:, 0],
            "SA_Seasonal_Weight": scale_weights[:, 1],
            "CSG_Gate_Mean": gate_mean,
        }
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Predictions and RQ3 diagnostics saved to: {OUTPUT_FILE}")
