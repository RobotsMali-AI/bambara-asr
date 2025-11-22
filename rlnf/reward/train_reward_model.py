#!/usr/bin/env python3
"""
Train Reward Model (Config-driven)
---------------------------------
All parameters are provided via a YAML config file (including the audio
preprocessor settings).

Usage
~~~~~

# Train using a config
python reward_model_train_configured.py --config config.yaml

"""
import os
import json
import argparse
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e

from rlnf.dataloaders.reward_dataset import get_dataloaders
from rlnf.reward.reward_model import RewardModel
from rlnf.reward.train_utils import fit, evaluate
from sentencepiece import SentencePieceProcessor

# -----------------------
# Tokenizer
# -----------------------

def load_tokenizer(model_path: str) -> SentencePieceProcessor:
    sp = SentencePieceProcessor()
    sp.Load(model_path)
    return sp


# -----------------------
# Config helpers
# -----------------------

def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key in config: '{key}'")
    return d[key]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Basic shape checks
    for top in ("paths", "training", "dataloader", "optimizer", "model", "preprocessor"):
        if top not in cfg:
            raise KeyError(f"Config missing top-level section '{top}'")
    return cfg


# -----------------------
# Main
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config-driven Reward Model training")
    p.add_argument("--config", type=str, help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config:
        raise SystemExit("--config is required")

    cfg = load_config(args.config)

    # Paths
    train_manifest = _require(cfg["paths"], "train_manifest")
    test_manifest = _require(cfg["paths"], "test_manifest")
    save_dir = _require(cfg["paths"], "save_dir")
    tokenizer_path = _require(cfg["paths"], "tokenizer_path")

    # Training & dataloader
    epochs = int(_require(cfg["training"], "epochs"))
    seed = int(cfg["training"].get("seed", 42))
    checkpoint_dir_name = cfg["training"].get("checkpoint_dir", "checkpoints")

    batch_size = int(_require(cfg["dataloader"], "batch_size"))
    num_workers = int(cfg["dataloader"].get("num_workers", 0))

    # Optimizer
    opt_cfg = _require(cfg, "optimizer")
    lr = float(_require(opt_cfg, "lr"))

    # Scheduler
    sch_cfg = cfg.get("scheduler", {"use": False})
    use_scheduler = bool(sch_cfg.get("use", False))
    step_size = int(sch_cfg.get("step_size", 30))
    gamma = float(sch_cfg.get("gamma", 0.8))

    # Model
    mcfg = _require(cfg, "model")
    embed_dim = int(mcfg.get("embed_dim", 128))
    hidden_dim = int(mcfg.get("hidden_dim", 256))
    lstm_layers = int(mcfg.get("lstm_layers", 1))
    audio_conv_channels = int(mcfg.get("audio_conv_channels", 128))
    audio_conv_layers = int(mcfg.get("audio_conv_layers", 3))
    head_hidden = int(mcfg.get("head_hidden", hidden_dim))
    dropout = float(mcfg.get("dropout", 0.3))

    # Preprocessor
    preprocessor_config: Dict[str, Any] = _require(cfg, "preprocessor")
    n_mel = int(preprocessor_config.get("features", 64))

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create save dirs
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, checkpoint_dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print config snapshot
    pretty = {
        "paths": cfg["paths"],
        "training": {k: cfg["training"][k] for k in cfg["training"]},
        "dataloader": cfg["dataloader"],
        "optimizer": cfg["optimizer"],
        "scheduler": cfg.get("scheduler", {}),
        "model": cfg["model"],
        "preprocessor": cfg["preprocessor"],
    }
    print("Configuration:\n" + json.dumps(pretty, indent=2))

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.GetPieceSize()
    print(f"Text Tokenizer Vocabulary size: {vocab_size}")

    # DataLoaders
    print("Preparing data loaders...")
    train_loader, test_loader = get_dataloaders(
        train_manifest,
        test_manifest,
        tokenizer_path,
        preprocessor_config=preprocessor_config,
        batch_size=batch_size,
        audio_transform=None,
        num_workers=num_workers,
    )

    # Model
    print("Instantiating model...")
    model = RewardModel(
        n_mel=n_mel,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        lstm_hidden=hidden_dim,
        lstm_layers=lstm_layers,
        audio_conv_channels=audio_conv_channels,
        audio_conv_layers=audio_conv_layers,
        head_hidden=head_hidden,
        dropout=dropout,
    )
    model.to(device)

    # Optimizer & loss
    if opt_cfg.get("type", "adam").lower() != "adam":
        print("[warning] Only Adam is supported currently; falling back to Adam.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"Using scheduler: StepLR(step_size={step_size}, gamma={gamma})")

    # Training
    history = fit(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=test_loader,
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler,
    )

    # Final model
    final_path = os.path.join(save_dir, "final_model.rw")
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    # Save training logs
    logs_path = os.path.join(save_dir, "training_logs.json")
    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training logs to {logs_path}")

    # Final evaluation
    print("Final evaluation on test set:")
    print(evaluate(model, test_loader, criterion, device))


if __name__ == '__main__':
    main()
