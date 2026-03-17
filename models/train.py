"""Offline training script for LSTM and Transformer alpha models.

Usage:
    uv run python -m models.train --symbols BTC/USDT ETH/USDT --days 90
    uv run python -m models.train --parquet-dir /path/to/parquet --model-type transformer
    uv run python -m models.train --synthetic --symbols BTC/USDT --device auto

Pipeline:
    1. Load OHLCV data (parquet / ccxt REST / synthetic GBM)
    2. Compute features using FeatureExtractor
    3. Build (sequence, label) pairs — label = forward return
    4. Train LSTM or Transformer with MSE loss
    5. Export best model to ONNX
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import OHLCV
from features.extractor import FeatureExtractor
from models.lstm_model import LSTMAlphaModel

from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

# Resolve project root from this file's location so paths work regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Wandb helpers ──────────────────────────────────────────────────────────

_wandb_run = None  # module-level handle


def _init_wandb(args: argparse.Namespace, extra_config: Optional[dict] = None) -> None:
    """Initialize a wandb run if --wandb is set."""
    global _wandb_run
    if not getattr(args, "wandb", False):
        return

    import wandb

    config = {
        "symbols": args.symbols,
        "data_source": "synthetic" if args.synthetic else "exchange",
        "n_candles": args.n_candles if args.synthetic else None,
        "days": args.days if not args.synthetic else None,
        "seq_len": args.seq_len,
        "forward_window": args.forward_window,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "amp": args.amp,
        "compile": getattr(args, "compile", False),
        "seed": args.seed,
        "resample_minutes": args.resample_minutes,
        "loss": "log_weighted_mse",
    }
    if extra_config:
        config.update(extra_config)

    # Resolve device tag
    device_tag = args.device
    if device_tag == "auto":
        device_tag = "cuda" if torch.cuda.is_available() else "cpu"

    # Flags string
    flags = []
    if args.amp:
        flags.append("amp")
    if getattr(args, "compile", False):
        flags.append("compile")
    flag_str = f"_{'_'.join(flags)}" if flags else ""

    # Candle count label
    candle_k = args.n_candles // 1000 if args.synthetic else args.days
    unit = "k" if args.synthetic else "d"

    date_tag = datetime.now().strftime("%m%d")

    # Group: device_epochs_candles_flags_date (one group per day per experiment)
    group = args.wandb_group
    if not group:
        group = f"{device_tag}_e{args.epochs}_{candle_k}{unit}{flag_str}_{date_tag}"

    # Run name: group_HHMMSS (nested under the group)
    name = args.wandb_name
    if not name:
        ts = datetime.now().strftime("%H%M%S")
        name = f"{group}_{ts}"

    # Default tag to "test" if none provided
    tags = args.wandb_tags.split(",") if args.wandb_tags else ["test"]

    _wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=group,
        name=name,
        tags=tags,
        config=config,
    )
    logger.info("wandb run: %s (group=%s)", _wandb_run.url, group)


def _log_wandb(metrics: dict, step: Optional[int] = None) -> None:
    """Log metrics to wandb if active."""
    if _wandb_run is not None:
        _wandb_run.log(metrics, step=step)


def _finish_wandb(summary: Optional[dict] = None) -> None:
    """Finalize wandb run."""
    global _wandb_run
    if _wandb_run is not None:
        if summary:
            for k, v in summary.items():
                _wandb_run.summary[k] = v
        _wandb_run.finish()
        _wandb_run = None


def setup_train_logging() -> None:
    """Configure console + rotating file logging for training runs."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{ts}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)

    logger.info("Training log: %s", log_file)

HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ── Data Fetching ───────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str = "1m", days: int = 90) -> List[list]:
    """Fetch historical OHLCV from exchange via ccxt, with CSV caching."""
    import ccxt

    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = symbol.replace("/", "_") + f"_{interval}.csv"
    csv_path = HISTORICAL_DIR / csv_name
    log_path = HISTORICAL_DIR / "fetch_log.json"

    # Check cache
    fetch_log = {}
    if log_path.exists():
        fetch_log = json.loads(log_path.read_text())

    cached_data = []
    last_ts = 0

    if csv_path.exists() and symbol in fetch_log:
        logger.info("Loading cached data from %s", csv_path)
        import csv
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                cached_data.append([
                    int(row[0]),    # timestamp ms
                    float(row[1]),  # open
                    float(row[2]),  # high
                    float(row[3]),  # low
                    float(row[4]),  # close
                    float(row[5]),  # volume
                ])
        if cached_data:
            last_ts = cached_data[-1][0]
            logger.info("Cache has %d candles, last ts=%d", len(cached_data), last_ts)

    # Fetch new data
    exchange = ccxt.binance({"enableRateLimit": True})
    since = last_ts + 60000 if last_ts else int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    now_ms = int(datetime.utcnow().timestamp() * 1000)

    all_new = []
    logger.info("Fetching %s from %d to %d", symbol, since, now_ms)

    while since < now_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, interval, since=since, limit=1000)
        except Exception as e:
            logger.error("Fetch error: %s", e)
            break

        if not batch:
            break

        all_new.extend(batch)
        since = batch[-1][0] + 60000  # next minute
        logger.info("  fetched %d candles, total new=%d", len(batch), len(all_new))

    # Merge and save
    all_data = cached_data + all_new
    logger.info("Total candles: %d (cached=%d, new=%d)", len(all_data), len(cached_data), len(all_new))

    if all_new:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for row in all_data:
                writer.writerow(row)

        fetch_log[symbol] = {
            "last_timestamp": all_data[-1][0],
            "count": len(all_data),
            "updated_at": datetime.utcnow().isoformat(),
        }
        log_path.write_text(json.dumps(fetch_log, indent=2))
        logger.info("Saved %d candles to %s", len(all_data), csv_path)

    return all_data


def raw_to_ohlcv(raw_data: List[list], symbol: str) -> List[OHLCV]:
    """Convert raw ccxt OHLCV lists to OHLCV model objects."""
    candles = []
    for row in raw_data:
        candles.append(OHLCV(
            symbol=symbol,
            open=row[1],
            high=row[2],
            low=row[3],
            close=row[4],
            volume=row[5],
            timestamp=datetime.utcfromtimestamp(row[0] / 1000),
            is_closed=True,
        ))
    return candles


def resample_candles(candles: List[OHLCV], minutes: int) -> List[OHLCV]:
    """Aggregate 1-min candles into N-min candles.

    Groups by floor(timestamp, minutes). Discards incomplete trailing bars.
    OHLCV aggregation: open=first, high=max, low=min, close=last, volume=sum.
    """
    if minutes <= 1:
        return candles
    if not candles:
        return []

    result: List[OHLCV] = []
    bucket: List[OHLCV] = []
    symbol = candles[0].symbol

    def _floor_ts(ts: datetime) -> datetime:
        """Floor timestamp to nearest N-minute boundary."""
        total_min = ts.hour * 60 + ts.minute
        floored_min = (total_min // minutes) * minutes
        return ts.replace(hour=floored_min // 60, minute=floored_min % 60, second=0, microsecond=0)

    current_bucket_ts = _floor_ts(candles[0].timestamp)

    for c in candles:
        bucket_ts = _floor_ts(c.timestamp)
        if bucket_ts != current_bucket_ts:
            # Emit completed bucket
            if len(bucket) == minutes:
                result.append(OHLCV(
                    symbol=symbol,
                    open=bucket[0].open,
                    high=max(b.high for b in bucket),
                    low=min(b.low for b in bucket),
                    close=bucket[-1].close,
                    volume=sum(b.volume for b in bucket),
                    timestamp=bucket[-1].timestamp,
                    is_closed=True,
                ))
            bucket = [c]
            current_bucket_ts = bucket_ts
        else:
            bucket.append(c)

    # Emit last bucket if complete
    if len(bucket) == minutes:
        result.append(OHLCV(
            symbol=symbol,
            open=bucket[0].open,
            high=max(b.high for b in bucket),
            low=min(b.low for b in bucket),
            close=bucket[-1].close,
            volume=sum(b.volume for b in bucket),
            timestamp=bucket[-1].timestamp,
            is_closed=True,
        ))

    logger.info("Resampled %d 1-min candles → %d %d-min candles", len(candles), len(result), minutes)
    return result


def load_parquet_ohlcv(parquet_dir: str, symbols: Optional[List[str]] = None) -> dict:
    """Load OHLCV data from parquet files.

    Args:
        parquet_dir: Directory containing .parquet files
        symbols: Optional list of symbols to filter. If None, loads all.

    Returns:
        Dict mapping symbol → List[OHLCV], sorted by timestamp.
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    parquet_path = Path(parquet_dir)
    parquet_files = sorted(parquet_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    logger.info("Loading parquet files from %s (%d files)", parquet_dir, len(parquet_files))
    tables = [pq.read_table(f) for f in parquet_files]
    table = pa.concat_tables(tables)
    logger.info("Loaded %d total rows", len(table))

    # Convert to pandas for easy groupby
    df = table.to_pandas()
    available_symbols = sorted(df["symbol"].unique())
    logger.info("Available symbols: %d", len(available_symbols))

    if symbols:
        missing = set(symbols) - set(available_symbols)
        if missing:
            logger.warning("Symbols not found in parquet: %s", missing)
        df = df[df["symbol"].isin(symbols)]
    else:
        symbols = available_symbols

    result = {}
    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        n = len(sym_df)
        opens = sym_df["open"].values
        highs = sym_df["high"].values
        lows = sym_df["low"].values
        closes = sym_df["close"].values
        vols = sym_df["volume"].values
        timestamps = sym_df["timestamp"].dt.tz_localize(None).values.astype("datetime64[us]")

        candles = [
            OHLCV(symbol=sym, open=float(opens[i]), high=float(highs[i]),
                   low=float(lows[i]), close=float(closes[i]), volume=float(vols[i]),
                   timestamp=timestamps[i].item(), is_closed=True)
            for i in range(n)
        ]
        result[sym] = candles
        logger.info("  %s: %d candles", sym, len(candles))

    return result


# ── Dataset Building ────────────────────────────────────────────────────────

def _compute_all_features(candles: List[OHLCV], extractor: FeatureExtractor) -> np.ndarray:
    """Compute features for every candle using vectorized numpy.

    Returns: (N, n_features) array — [rsi, ema_fast, ema_slow, atr, momentum, volatility,
    order_book_imbalance, volume_ratio, funding_rate, taker_ratio].
    Rows before min_candles are zero-filled.
    Note: order_book_imbalance, funding_rate, taker_ratio are zero in offline training
    (only available during live trading via supplementary feeds).
    """
    n = len(candles)
    n_features = extractor.N_FEATURES
    closes = np.array([c.close for c in candles], dtype=np.float64)
    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    volumes = np.array([c.volume for c in candles], dtype=np.float64)

    features = np.zeros((n, n_features), dtype=np.float32)
    min_c = extractor.min_candles

    # ── RSI: rolling gains/losses ──
    rsi_p = extractor._rsi_period
    deltas = np.diff(closes)  # (n-1,)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi_all = np.full(n, 50.0, dtype=np.float64)
    # Rolling mean via cumsum
    cum_gains = np.concatenate([[0], np.cumsum(gains)])
    cum_losses = np.concatenate([[0], np.cumsum(losses)])
    for i in range(rsi_p, n):
        avg_gain = (cum_gains[i] - cum_gains[i - rsi_p]) / rsi_p
        avg_loss = (cum_losses[i] - cum_losses[i - rsi_p]) / rsi_p
        if avg_loss < 1e-10:
            rsi_all[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_all[i] = 100.0 - 100.0 / (1.0 + rs)

    # ── EMA: incremental computation ──
    def _ema_series(vals: np.ndarray, period: int) -> np.ndarray:
        mult = 2.0 / (period + 1)
        ema = np.zeros(n, dtype=np.float64)
        ema[0] = vals[0]
        for i in range(1, n):
            ema[i] = vals[i] * mult + ema[i - 1] * (1 - mult)
        return ema

    ema_fast_all = _ema_series(closes, extractor._ema_fast)
    ema_slow_all = _ema_series(closes, extractor._ema_slow)

    # ── ATR: rolling true range ──
    atr_p = extractor._atr_period
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    cum_tr = np.cumsum(tr)
    atr_all = np.zeros(n, dtype=np.float64)
    for i in range(atr_p, n):
        atr_all[i] = (cum_tr[i] - cum_tr[i - atr_p]) / atr_p

    # ── Momentum: rate of change (fully vectorized) ──
    mom_w = extractor._momentum_window
    mom_all = np.zeros(n, dtype=np.float64)
    safe_denom = np.where(closes == 0, 1e-10, closes)
    mom_all[mom_w:] = closes[mom_w:] / safe_denom[:n - mom_w] - 1.0

    # ── Volatility: std of log returns (rolling via cumsum trick) ──
    vol_w = extractor._volatility_window
    safe_closes = np.where(closes <= 0, 1e-10, closes)
    log_prices = np.log(safe_closes)
    log_ret = np.diff(log_prices)  # (n-1,)
    # Rolling mean and var via cumsum
    cum_lr = np.concatenate([[0], np.cumsum(log_ret)])
    cum_lr2 = np.concatenate([[0], np.cumsum(log_ret ** 2)])
    vol_all = np.zeros(n, dtype=np.float64)
    for i in range(vol_w + 1, n):
        j = i - 1  # index into log_ret (length n-1, offset by 1 from closes)
        s = cum_lr[j + 1] - cum_lr[j + 1 - vol_w]
        s2 = cum_lr2[j + 1] - cum_lr2[j + 1 - vol_w]
        var = s2 / vol_w - (s / vol_w) ** 2
        vol_all[i] = np.sqrt(max(var, 0.0))

    # Pack into features array
    features[min_c:, 0] = rsi_all[min_c:]
    features[min_c:, 1] = ema_fast_all[min_c:]
    features[min_c:, 2] = ema_slow_all[min_c:]
    features[min_c:, 3] = atr_all[min_c:]
    features[min_c:, 4] = mom_all[min_c:]
    features[min_c:, 5] = vol_all[min_c:]

    # Volume ratio: current volume / rolling average (feature index 7, vectorized)
    vol_ratio_window = 24
    if n_features > 7:
        cum_vol = np.cumsum(volumes)
        # rolling avg = (cum_vol[i-1] - cum_vol[i-1-window]) / window
        start = vol_ratio_window + 1
        if start < n:
            idx = np.arange(start, n)
            avg_vol = (cum_vol[idx - 1] - cum_vol[idx - 1 - vol_ratio_window]) / vol_ratio_window
            safe_avg = np.where(avg_vol > 1e-10, avg_vol, 1.0)
            features[start:, 7] = np.where(avg_vol > 1e-10, volumes[start:] / safe_avg, 1.0)

    # Synthetic supplementary features for training (indices 6, 8, 9)
    # In live mode these come from Binance feeds. During training we generate
    # realistic synthetic values so the model learns to use them, avoiding
    # a train/test distribution mismatch where they'd always be zero.
    if n_features > 9:
        rng = np.random.RandomState(42)
        # order_book_imbalance: centered around 1.0, range ~[0.5, 2.0]
        features[min_c:, 6] = rng.lognormal(0.0, 0.3, n - min_c).astype(np.float32)
        # funding_rate: small values centered around 0, range ~[-0.001, 0.001]
        features[min_c:, 8] = rng.normal(0.0001, 0.0003, n - min_c).astype(np.float32)
        # taker_ratio: centered around 1.0, range ~[0.6, 1.6]
        features[min_c:, 9] = rng.lognormal(0.0, 0.2, n - min_c).astype(np.float32)

    return features


def build_dataset(
    candles: List[OHLCV],
    extractor: FeatureExtractor,
    seq_len: int = 30,
    forward_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) pairs for training.

    X: (N, seq_len, n_features) — feature sequences
    y: (N, 1) — forward returns mapped to [-1, 1] via tanh scaling

    Computes all features once, then slices sliding windows — much faster
    than recomputing features per sample.
    """
    min_start = extractor.min_candles + seq_len
    max_end = len(candles) - forward_window
    n_samples = max_end - min_start

    if n_samples <= 0:
        raise ValueError(
            f"Not enough candles: need at least {min_start + forward_window}, got {len(candles)}"
        )

    logger.info("Building dataset: %d samples (vectorized)", n_samples)

    # Step 1: compute features for all candles in one pass
    t0 = time.time()
    all_features = _compute_all_features(candles, extractor)
    logger.info("Feature extraction: %.1fs", time.time() - t0)

    # Step 2: slice sliding windows + z-score normalize each window
    t0 = time.time()
    closes = np.array([c.close for c in candles], dtype=np.float64)

    X = np.zeros((n_samples, seq_len, extractor.N_FEATURES), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)

    for idx, i in enumerate(range(min_start, max_end)):
        window = all_features[i - seq_len + 1: i + 1]  # (seq_len, n_features)
        # Z-score normalize per window
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        X[idx] = (window - mean) / std

        # Label: forward return scaled by tanh
        cur = closes[i]
        fut = closes[i + forward_window]
        fwd_return = (fut - cur) / cur if cur > 0 else 0.0
        y[idx, 0] = np.tanh(fwd_return * 100.0)

    # Z-score normalize labels so MSE loss has meaningful gradients
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-8:
        y_std = 1.0
    y = (y - y_mean) / y_std

    logger.info("Window slicing: %.1fs", time.time() - t0)
    logger.info("Dataset shape: X=%s, y=%s | y_mean=%.4f, y_std=%.4f",
                X.shape, y.shape, y_mean, y_std)
    return X, y


# ── Training ────────────────────────────────────────────────────────────────

def _create_model(model_type: str, n_features: int, **kwargs) -> nn.Module:
    """Create model by type string with optional architecture overrides."""
    if model_type == "lstm":
        lstm_kwargs = {k: v for k, v in kwargs.items()
                       if k in ("hidden_size", "num_layers", "dropout") and v is not None}
        return LSTMAlphaModel(n_features=n_features, **lstm_kwargs)
    elif model_type == "transformer":
        from models.transformer_model import TransformerAlphaModel
        tf_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("d_model", "nhead", "num_layers", "d_ff", "dropout") and v is not None}
        return TransformerAlphaModel(n_features=n_features, **tf_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int = 10,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    use_amp: bool = False,
    use_compile: bool = False,
    sample_weights: Optional[np.ndarray] = None,
    model_type: str = "lstm",
    model_kwargs: Optional[dict] = None,
    early_stop_patience: int = 5,
) -> nn.Module:
    """Train LSTM or Transformer model with optional recency-weighted loss."""
    # Use GPU if available
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s (model: %s)", device, model_type)

    model = _create_model(model_type, n_features, **(model_kwargs or {})).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    if use_compile:
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.warning("torch.compile failed, using eager mode: %s", e)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    if sample_weights is not None:
        w_train = torch.from_numpy(sample_weights).float()
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), w_train)
    else:
        # Use uniform weights
        w_train = torch.ones(len(X_train), dtype=torch.float32)
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), w_train)
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("Automatic mixed precision enabled")

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    epoch_times: list[float] = []

    # Pre-training validation baseline
    model.eval()
    pre_val_loss = 0.0
    criterion_pre = nn.MSELoss()
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                pred = model(xb)
                pre_val_loss += criterion_pre(pred, yb).item() * len(xb)
    pre_val_loss /= len(val_ds)
    logger.info("Pre-train val_loss=%.4f (baseline, naive≈1.0)", pre_val_loss)

    t_start = time.time()

    for epoch in range(epochs):
        t_epoch = time.time()

        # Train
        model.train()
        train_loss = 0.0
        n_batches = len(train_dl)
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [train]",
                    leave=False, dynamic_ncols=True, mininterval=30)
        for xb, yb, wb in pbar:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                pred = model(xb)
                # Log-weighted MSE: error² × log(1 + error²)
                # Super-quadratic: gentle near zero, escalates for large errors
                error_sq = (pred - yb) ** 2
                per_sample_loss = error_sq * torch.log1p(error_sq)
                loss = (per_sample_loss * wb.unsqueeze(1)).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(xb)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        criterion_val = nn.MSELoss()
        val_example_logged = False
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast(device_type=device, enabled=use_amp):
                    pred = model(xb)
                    val_loss += criterion_val(pred, yb).item() * len(xb)
                # Log one example per epoch
                if not val_example_logged:
                    ex_pred = pred[0].item()
                    ex_true = yb[0].item()
                    ex_input = xb[0, -1, :6].cpu().numpy()
                    feat_names = ["rsi", "ema_f", "ema_s", "atr", "mom", "vol"]
                    feat_str = " ".join(f"{n}={v:+.2f}" for n, v in zip(feat_names, ex_input))
                    val_example_logged = True
        val_loss /= len(val_ds)

        # Log validation example
        logger.info(
            "  sample: pred=%.4f true=%.4f err=%.4f | %s",
            ex_pred, ex_true, abs(ex_pred - ex_true), feat_str,
        )

        scheduler.step(val_loss)

        epoch_dt = time.time() - t_epoch
        epoch_times.append(epoch_dt)
        cur_lr = optimizer.param_groups[0]["lr"]
        improved = val_loss < best_val_loss

        # Terminal output
        marker = " *" if improved else ""
        logger.info(
            "Epoch %d/%d  train=%.4f  val=%.4f  lr=%.2e  %.1fs%s",
            epoch + 1, epochs, train_loss, val_loss, cur_lr, epoch_dt, marker,
        )

        # wandb
        _log_wandb({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/lr": cur_lr,
            "train/epoch_time_s": epoch_dt,
        }, step=epoch + 1)

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch + 1, early_stop_patience)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.cpu().eval()

    total_time = time.time() - t_start
    avg_epoch = np.mean(epoch_times) if epoch_times else 0

    # Summary banner
    logger.info("─" * 60)
    logger.info("Training complete in %.1fs (%.1fs/epoch avg)", total_time, avg_epoch)
    logger.info("Best val_loss=%.6f @ epoch %d/%d", best_val_loss, best_epoch, epochs)
    logger.info("Device: %s | AMP: %s | Compile: %s",
                device, use_amp, use_compile)
    logger.info("─" * 60)

    _log_wandb({
        "summary/best_val_loss": best_val_loss,
        "summary/best_epoch": best_epoch,
        "summary/total_time_s": total_time,
        "summary/avg_epoch_time_s": avg_epoch,
    })

    return model


# ── ONNX Export ─────────────────────────────────────────────────────────────

def export_onnx(model: LSTMAlphaModel, seq_len: int, n_features: int, path: str) -> None:
    """Export trained model to ONNX format."""
    # Unwrap torch.compile's OptimizedModule if present
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.eval()
    raw_model.cpu()
    dummy_input = torch.randn(1, seq_len, n_features)

    torch.onnx.export(
        raw_model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["alpha"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "alpha": {0: "batch_size"},
        },
        opset_version=18,
    )
    logger.info("Exported ONNX model to %s", path)

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    test_input = dummy_input.numpy()
    result = sess.run(None, {"input": test_input})
    logger.info("ONNX verification — output shape: %s, value: %.4f", result[0].shape, result[0][0][0])


# ── Main ────────────────────────────────────────────────────────────────────

def generate_synthetic_ohlcv(
    symbol: str = "BTC/USDT",
    n_candles: int = 10_000,
    start_price: float = 65000.0,
    drift: float = 0.0001,
    vol: float = 0.002,
    base_volume: float = 50.0,
    seed: int = 42,
) -> List[OHLCV]:
    """Generate synthetic OHLCV candles using geometric Brownian motion.

    Useful for training when exchange APIs are unavailable.
    """
    import math
    import random as _random

    rng = _random.Random(seed)
    candles = []
    price = start_price
    dt = 1.0 / 1440.0  # 1 minute as fraction of day
    base_time = datetime(2025, 1, 1)

    for i in range(n_candles):
        dw = rng.gauss(0, 1)
        returns = drift * dt + vol * math.sqrt(dt) * dw

        open_price = price
        close_price = price * (1 + returns)

        intra_vol = abs(returns) + vol * math.sqrt(dt) * 0.5
        high_price = max(open_price, close_price) * (1 + abs(rng.gauss(0, intra_vol)))
        low_price = min(open_price, close_price) * (1 - abs(rng.gauss(0, intra_vol)))

        volume = base_volume * (1 + abs(rng.gauss(0, 0.5)))

        candles.append(OHLCV(
            symbol=symbol,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=round(volume, 4),
            timestamp=base_time + timedelta(minutes=i),
            is_closed=True,
        ))
        price = close_price

    logger.info("Generated %d synthetic candles for %s (seed=%d)", n_candles, symbol, seed)
    return candles


# Presets for synthetic data generation
SYNTHETIC_PRESETS = {
    "BTC/USDT": {"start_price": 65000.0, "drift": 0.0001, "vol": 0.002, "base_volume": 50.0},
    "ETH/USDT": {"start_price": 3500.0, "drift": 0.00015, "vol": 0.003, "base_volume": 500.0},
}


def main():
    setup_train_logging()
    parser = argparse.ArgumentParser(description="Train LSTM alpha model")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"], help="Symbols to train on")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    parser.add_argument("--interval", default="1m", help="Candle interval")
    parser.add_argument("--seq-len", type=int, default=30, help="LSTM lookback window")
    parser.add_argument("--forward-window", type=int, default=5, help="Forward return window for labels")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="auto", help="Device: cpu, cuda, auto")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Use walk-forward validation: train 0-60%%, val 60-80%%, test 80-100%%",
    )
    parser.add_argument(
        "--recency-half-life", type=float, default=0.0,
        help="Recency weighting half-life in days (0 = uniform weighting). E.g., 35 = recent data weighted 2x vs 35-day-old data",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic GBM data instead of fetching from exchange (no API needed)",
    )
    parser.add_argument("--n-candles", type=int, default=10000, help="Number of synthetic candles per symbol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--parquet-dir", default="", help="Load OHLCV from parquet directory (overrides --synthetic and ccxt)")
    parser.add_argument("--model-type", default="lstm", choices=["lstm", "transformer"], help="Model architecture")
    # Model architecture overrides
    parser.add_argument("--hidden-size", type=int, default=None, help="LSTM hidden size (default: 128)")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of LSTM/Transformer layers")
    parser.add_argument("--d-model", type=int, default=None, help="Transformer d_model dimension")
    parser.add_argument("--nhead", type=int, default=None, help="Transformer attention heads")
    parser.add_argument("--d-ff", type=int, default=None, help="Transformer feedforward dimension")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (GPU recommended)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (PyTorch 2.0+, GPU recommended)")
    parser.add_argument("--resample-minutes", type=int, default=1, help="Resample candles to N-minute bars (1=no-op)")
    parser.add_argument("--save-dataset", default="", help="Pre-build dataset and save to .npz (skip training)")
    parser.add_argument("--load-dataset", default="", help="Load pre-built dataset from .npz (skip data parsing)")
    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb experiment tracking")
    parser.add_argument("--wandb-entity", default="Base-Work-Space", help="wandb entity/team")
    parser.add_argument("--wandb-project", default="trading-lstm", help="wandb project name")
    parser.add_argument("--wandb-group", default="", help="wandb run group (auto-generated if empty)")
    parser.add_argument("--wandb-name", default="", help="wandb run name (auto-generated if empty)")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated wandb tags")
    args = parser.parse_args()

    _init_wandb(args)

    feature_config = {
        "rsi_period": 14,
        "ema_fast": 12,
        "ema_slow": 26,
        "atr_period": 14,
        "volatility_window": 20,
        "momentum_window": 10,
    }
    extractor = FeatureExtractor(feature_config)

    # ── Load or build dataset ──
    if args.load_dataset:
        logger.info("Loading pre-built dataset from %s", args.load_dataset)
        t0 = time.time()
        data = np.load(args.load_dataset)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        logger.info("Loaded in %.1fs: Train=%d, Val=%d", time.time() - t0, len(X_train), len(X_val))
    else:
        # Build from raw data
        all_X = []
        all_y = []

        if args.parquet_dir:
            symbol_filter = args.symbols if args.symbols != ["BTC/USDT"] else None
            symbol_candles = load_parquet_ohlcv(args.parquet_dir, symbols=symbol_filter)
            for symbol, candles in symbol_candles.items():
                if args.resample_minutes > 1:
                    candles = resample_candles(candles, args.resample_minutes)
                logger.info("=== Processing %s (%d candles) ===", symbol, len(candles))
                X, y = build_dataset(candles, extractor, args.seq_len, args.forward_window)
                all_X.append(X)
                all_y.append(y)
        else:
            for symbol in args.symbols:
                logger.info("=== Processing %s ===", symbol)

                if args.synthetic:
                    preset = SYNTHETIC_PRESETS.get(symbol, {})
                    candles = generate_synthetic_ohlcv(
                        symbol=symbol,
                        n_candles=args.n_candles,
                        start_price=preset.get("start_price", 1000.0),
                        drift=preset.get("drift", 0.0001),
                        vol=preset.get("vol", 0.002),
                        base_volume=preset.get("base_volume", 100.0),
                        seed=args.seed,
                    )
                else:
                    raw_data = fetch_ohlcv(symbol, args.interval, args.days)
                    candles = raw_to_ohlcv(raw_data, symbol)

                if args.resample_minutes > 1:
                    candles = resample_candles(candles, args.resample_minutes)
                logger.info("Got %d candles for %s", len(candles), symbol)

                X, y = build_dataset(candles, extractor, args.seq_len, args.forward_window)
                all_X.append(X)
                all_y.append(y)

    # Per-symbol chronological split (skip if loaded from npz)
    if not args.load_dataset:
        if args.walk_forward:
            X_trains, X_vals, X_tests = [], [], []
            y_trains, y_vals, y_tests = [], [], []
            for X_sym, y_sym in zip(all_X, all_y):
                s1 = int(len(X_sym) * 0.6)
                s2 = int(len(X_sym) * 0.8)
                X_trains.append(X_sym[:s1])
                X_vals.append(X_sym[s1:s2])
                X_tests.append(X_sym[s2:])
                y_trains.append(y_sym[:s1])
                y_vals.append(y_sym[s1:s2])
                y_tests.append(y_sym[s2:])
            X_train = np.concatenate(X_trains)
            X_val = np.concatenate(X_vals)
            X_test = np.concatenate(X_tests)
            y_train = np.concatenate(y_trains)
            y_val = np.concatenate(y_vals)
            y_test = np.concatenate(y_tests)
            logger.info("Walk-forward split: Train=%d, Val=%d, Test=%d", len(X_train), len(X_val), len(X_test))
        else:
            X_trains, X_vals = [], []
            y_trains, y_vals = [], []
            for X_sym, y_sym in zip(all_X, all_y):
                split_idx = int(len(X_sym) * (1 - args.val_split))
                X_trains.append(X_sym[:split_idx])
                X_vals.append(X_sym[split_idx:])
                y_trains.append(y_sym[:split_idx])
                y_vals.append(y_sym[split_idx:])
            X_train = np.concatenate(X_trains)
            X_val = np.concatenate(X_vals)
            y_train = np.concatenate(y_trains)
            y_val = np.concatenate(y_vals)

        # Save dataset if requested
        if args.save_dataset:
            logger.info("Saving dataset to %s", args.save_dataset)
            np.savez_compressed(args.save_dataset,
                                X_train=X_train, y_train=y_train,
                                X_val=X_val, y_val=y_val)
            logger.info("Saved: Train=%d, Val=%d", len(X_train), len(X_val))
            return

    logger.info("Train: %d samples, Val: %d samples", len(X_train), len(X_val))

    # Recency weighting: exponential decay so recent samples matter more
    sample_weights = None
    if args.recency_half_life > 0:
        candles_per_day = 1440 / args.resample_minutes
        decay_lambda = np.log(2) / (args.recency_half_life * candles_per_day)  # half-life in candle units
        # Weight increases from oldest (index 0) to newest (index N-1)
        ages = np.arange(len(X_train), 0, -1, dtype=np.float64)  # oldest=N, newest=1
        sample_weights = np.exp(-decay_lambda * ages).astype(np.float32)
        # Normalize so mean weight = 1.0
        sample_weights /= sample_weights.mean()
        logger.info(
            "Recency weighting: half-life=%.0f days, weight range=[%.3f, %.3f]",
            args.recency_half_life, sample_weights.min(), sample_weights.max(),
        )

    # Train
    # Build model architecture kwargs from CLI
    model_kwargs = {}
    if args.model_type == "lstm":
        if args.hidden_size is not None:
            model_kwargs["hidden_size"] = args.hidden_size
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
    elif args.model_type == "transformer":
        if args.d_model is not None:
            model_kwargs["d_model"] = args.d_model
        if args.nhead is not None:
            model_kwargs["nhead"] = args.nhead
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        if args.d_ff is not None:
            model_kwargs["d_ff"] = args.d_ff

    model = train_model(
        X_train, y_train, X_val, y_val,
        n_features=extractor.N_FEATURES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        use_amp=args.amp,
        use_compile=getattr(args, "compile", False),
        sample_weights=sample_weights,
        model_type=args.model_type,
        model_kwargs=model_kwargs,
    )

    # Save PyTorch checkpoint
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = str(ARTIFACTS_DIR / "model.pt")
    torch.save(model.state_dict(), pt_path)
    logger.info("Saved PyTorch model to %s", pt_path)

    # Export to ONNX
    onnx_path = str(ARTIFACTS_DIR / "model.onnx")
    export_onnx(model, args.seq_len, extractor.N_FEATURES, onnx_path)

    _finish_wandb(summary={
        "model_path": onnx_path,
    })

    logger.info("Done! Model ready at %s", onnx_path)


if __name__ == "__main__":
    main()
