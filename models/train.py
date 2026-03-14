"""Offline LSTM training script.

Usage:
    uv run python -m models.train --symbols BTC/USDT ETH/USDT --days 90

Pipeline:
    1. Fetch historical OHLCV via ccxt REST API (with CSV caching)
    2. Compute features using FeatureExtractor
    3. Build (sequence, label) pairs — label = forward return
    4. Train LSTM with MSE loss
    5. Export best model to ONNX
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import OHLCV
from features.extractor import FeatureExtractor
from models.lstm_model import LSTMAlphaModel

from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)


def setup_train_logging() -> None:
    """Configure console + rotating file logging for training runs."""
    log_dir = Path("logs")
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

HISTORICAL_DIR = Path("data/historical")
ARTIFACTS_DIR = Path("artifacts")


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


# ── Dataset Building ────────────────────────────────────────────────────────

def build_dataset(
    candles: List[OHLCV],
    extractor: FeatureExtractor,
    seq_len: int = 30,
    forward_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) pairs for training.

    X: (N, seq_len, n_features) — feature sequences
    y: (N, 1) — forward returns mapped to [-1, 1] via tanh scaling

    Args:
        candles: full history of OHLCV candles
        extractor: FeatureExtractor instance
        seq_len: lookback window for LSTM
        forward_window: how many candles ahead for the label
    """
    min_start = extractor.min_candles + seq_len
    max_end = len(candles) - forward_window

    if min_start >= max_end:
        raise ValueError(
            f"Not enough candles: need at least {min_start + forward_window}, got {len(candles)}"
        )

    X_list = []
    y_list = []

    logger.info("Building dataset: %d samples", max_end - min_start)

    for i in range(min_start, max_end):
        # Feature sequence ending at candle[i]
        window = candles[:i + 1]
        seq = extractor.extract_sequence(window, seq_len=seq_len)
        X_list.append(seq)

        # Label: forward return
        current_close = candles[i].close
        future_close = candles[i + forward_window].close
        if current_close > 0:
            fwd_return = (future_close - current_close) / current_close
        else:
            fwd_return = 0.0

        # Scale with tanh to [-1, 1], amplify small returns
        label = float(np.tanh(fwd_return * 100.0))
        y_list.append([label])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info("Dataset shape: X=%s, y=%s", X.shape, y.shape)
    return X, y


# ── Training ────────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int = 6,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    use_amp: bool = False,
    use_compile: bool = False,
) -> LSTMAlphaModel:
    """Train the LSTM model."""
    # Use GPU if available
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    model = LSTMAlphaModel(n_features=n_features).to(device)

    if use_compile:
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.warning("torch.compile failed, using eager mode: %s", e)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("Automatic mixed precision enabled")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                pred = model(xb)
                loss = criterion(pred, yb)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast(device_type=device, enabled=use_amp):
                    pred = model(xb)
                    val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)

        logger.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
            epoch + 1, epochs, train_loss, val_loss,
            optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.cpu().eval()

    logger.info("Best validation loss: %.6f", best_val_loss)
    return model


# ── ONNX Export ─────────────────────────────────────────────────────────────

def export_onnx(model: LSTMAlphaModel, seq_len: int, n_features: int, path: str) -> None:
    """Export trained model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, seq_len, n_features)

    torch.onnx.export(
        model,
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
        "--synthetic", action="store_true",
        help="Use synthetic GBM data instead of fetching from exchange (no API needed)",
    )
    parser.add_argument("--n-candles", type=int, default=10000, help="Number of synthetic candles per symbol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (GPU recommended)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (PyTorch 2.0+, GPU recommended)")
    args = parser.parse_args()

    feature_config = {
        "rsi_period": 14,
        "ema_fast": 12,
        "ema_slow": 26,
        "atr_period": 14,
        "volatility_window": 20,
        "momentum_window": 10,
    }
    extractor = FeatureExtractor(feature_config)

    # Fetch and merge data from all symbols
    all_X = []
    all_y = []

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

        logger.info("Got %d candles for %s", len(candles), symbol)

        X, y = build_dataset(candles, extractor, args.seq_len, args.forward_window)
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Chronological split (no shuffle for time-series)
    split_idx = int(len(X) * (1 - args.val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info("Train: %d samples, Val: %d samples", len(X_train), len(X_val))

    # Train
    model = train_model(
        X_train, y_train, X_val, y_val,
        n_features=extractor.N_FEATURES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # Save PyTorch checkpoint
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = str(ARTIFACTS_DIR / "model.pt")
    torch.save(model.state_dict(), pt_path)
    logger.info("Saved PyTorch model to %s", pt_path)

    # Export to ONNX
    onnx_path = str(ARTIFACTS_DIR / "model.onnx")
    export_onnx(model, args.seq_len, extractor.N_FEATURES, onnx_path)

    logger.info("Done! Model ready at %s", onnx_path)


if __name__ == "__main__":
    main()
