#!/usr/bin/env python3
"""Upload trained LSTM artifacts to Hugging Face Hub.

Uses ``LSTMAlphaModel.upload_to_hf`` — writes ``config.json`` + ``model.safetensors`` and uploads
``.onnx`` / ``README.md`` if present.

From repo root (with venv active):

  uv add huggingface_hub safetensors   # once
  huggingface-cli login                 # or HF_TOKEN in env

  python scripts/upload_model_to_hf.py --repo-id your-org/trading-lstm-v1

Architecture is inferred from ``artifacts/model.pt`` when possible; override with flags or ``--config``.

Examples::

  python scripts/upload_model_to_hf.py --repo-id user/my-model
  python scripts/upload_model_to_hf.py --repo-id user/my-model --model-dir ./my_artifacts --seq-len 60
  python scripts/upload_model_to_hf.py --repo-id user/my-model --config training_meta.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Run from trading_competition root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _infer_lstm_from_state_dict(state: dict) -> dict:
    """Derive n_features, hidden_size, num_layers from a saved LSTMAlphaModel state_dict."""
    ih_keys = [k for k in state if re.match(r"lstm\.weight_ih_l\d+$", k)]
    if not ih_keys:
        raise ValueError(
            "Cannot infer architecture: no lstm.weight_ih_l* keys. "
            "Pass --n-features, --hidden-size, --num-layers explicitly."
        )
    layer_ids: list[int] = []
    for k in ih_keys:
        m = re.search(r"l(\d+)$", k)
        if m:
            layer_ids.append(int(m.group(1)))
    if not layer_ids:
        raise ValueError("Could not parse LSTM layer indices from state dict.")
    layer_ids.sort()
    num_layers = layer_ids[-1] + 1
    w0 = state["lstm.weight_ih_l0"]
    hidden_size = w0.shape[0] // 4
    n_features = w0.shape[1]
    return {
        "n_features": n_features,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": 0.2,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo, e.g. username/my-lstm")
    parser.add_argument(
        "--model-dir",
        default="artifacts",
        type=Path,
        help="Directory with model.pt (and optional model.onnx, README.md)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON file merged into Hub config (n_features, hidden_size, num_layers, dropout, seq_len, ...)",
    )
    parser.add_argument("--n-features", type=int, help="Override inferred / default feature count")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seq-len", type=int, help="Stored in config.json for consumers (inference shape)")
    parser.add_argument(
        "--force-safetensors",
        action="store_true",
        help="Regenerate model.safetensors even if it already exists",
    )
    parser.add_argument(
        "--no-prefer-checkpoint",
        action="store_true",
        help="Use in-memory random weights instead of model.pt (almost never wanted)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    pt_path = model_dir / "model.pt"
    if not pt_path.is_file():
        print(f"Error: {pt_path} not found. Train or copy checkpoint there first.", file=sys.stderr)
        sys.exit(1)

    import torch

    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    try:
        hparams = _infer_lstm_from_state_dict(state)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    extra: dict = {}
    _ARCH = frozenset({"n_features", "hidden_size", "num_layers", "dropout"})

    def _apply_config_blob(blob: dict) -> None:
        for k, v in blob.items():
            if k in _ARCH:
                hparams[k] = v
            elif k == "seq_len":
                extra["seq_len"] = v
            elif k != "model_type":
                extra[k] = v

    if args.config and args.config.is_file():
        _apply_config_blob(_load_json(args.config))
    cfg_in_dir = model_dir / "config.json"
    if cfg_in_dir.is_file() and not args.config:
        try:
            _apply_config_blob(_load_json(cfg_in_dir))
        except json.JSONDecodeError:
            pass

    if args.n_features is not None:
        hparams["n_features"] = args.n_features
    if args.hidden_size is not None:
        hparams["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        hparams["num_layers"] = args.num_layers
    if args.dropout is not None:
        hparams["dropout"] = args.dropout
    if args.seq_len is not None:
        extra["seq_len"] = args.seq_len

    from models.lstm_model import LSTMAlphaModel

    model = LSTMAlphaModel(
        n_features=int(hparams["n_features"]),
        hidden_size=int(hparams["hidden_size"]),
        num_layers=int(hparams["num_layers"]),
        dropout=float(hparams.get("dropout", 0.2)),
    )
    # Sanity: state dict must load (architecture matches checkpoint)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(
            "Checkpoint does not match inferred architecture. "
            "Pass correct --n-features / --hidden-size / --num-layers.\n"
            f"  {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Repo:      {args.repo_id}")
    print(f"Model dir: {model_dir}")
    print(f"Config:    {model.hf_config() | extra}")
    model.upload_to_hf(
        args.repo_id,
        model_dir,
        extra_config=extra or None,
        prefer_checkpoint=not args.no_prefer_checkpoint,
        force_safetensors=args.force_safetensors,
    )


if __name__ == "__main__":
    main()
