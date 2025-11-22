from __future__ import annotations

import os
import sys
import argparse
from typing import Any, Dict

import numpy as np
import torch

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e

# -----------------------------
# Imports
# -----------------------------
import nemo.collections.asr as nemo_asr
from rlnf.reward.reward_model import RewardModel
from rlnf.ppo.critic_network import CriticModel
from rlnf.trainer import RLNFTrainer
from sentencepiece import SentencePieceProcessor

# -----------------------------
# Example config
# -----------------------------
EXAMPLE_CFG: Dict[str, Any] = {
    "training": {
        "seed": 42,
        "device": "cuda",              # "auto" | "cpu" | "cuda"
        "precision": "fp32",           # "fp32" | "bf16"
        "clip_eps": 0.2,
        "actor_lr": 1e-5,
        "critic_lr": 1e-4,
        "epochs": 3,
        "K_updates": 4,                # PPO updates per batch
        "save_dir": "./rlnf_runs/exp01",
        "pin_memory": False
    },

    "paths": {
        "tokenizer_path": "/path/to/tokenizer.model",
        "reward_path": "/path/to/reward_model.rw"
    },

    "wandb": {
        "enable": True,
        "project": "rlnf",
        "run_name": "exp01"
    },

    "model": {
        "name": "RobotsMali/stt-bm-quartznet15x5-V0",
        "vocab": None                  # list[str] if you really want to override; else leave None
    },

    "critic": {
        "hidden_dim": 256,
        "layers": 3,
        "dropout": 0.1
    },

    "data": {
        "train_manifest": "path/to/train_rollouts.jsonl",
        "valid_manifest": "path/to/valid_rollouts.jsonl",
        "batch_size": 4,
        "num_workers": 0
    },

    "preprocessor": {
        "normalize": "per_feature",
        "window_size": 0.02,
        "sample_rate": 16000,
        "window_stride": 0.01,
        "window": "hann",
        "features": 64,
        "n_fft": 512,
        "frame_splicing": 1,
        "dither": 1e-5,
        "stft_conv": False
    }
}

# -----------------------------
# Config IO
# -----------------------------

def write_example_config(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(EXAMPLE_CFG, f, sort_keys=False)
    print(f"Wrote example config to {path}")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Light validation
    for sec in ("paths", "model", "critic", "data", "preprocessor", "training"):
        if sec not in cfg:
            raise KeyError(f"Missing top-level section: {sec}")
    return cfg

# -----------------------------
# Builders
# -----------------------------

def load_tokenizer(model_path: str) -> SentencePieceProcessor:
    sp = SentencePieceProcessor()
    ok = sp.Load(model_path)
    if not ok:
        raise RuntimeError(f"Failed to load SentencePiece model at: {model_path}")
    return sp

def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)

def build_actor(cfg: Dict[str, Any]) -> nemo_asr.models.ASRModel:
    m = cfg["model"]
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=m["name"])  # type: ignore
    vocab = m.get("vocab")
    if vocab is not None:
        if not isinstance(vocab, (list, tuple)) or not all(isinstance(s, str) for s in vocab):
            raise ValueError("model.vocab must be a list[str] or None")
        # NeMo will require lengths to match decoder dims; only do this if you know what you're doing.
        model.change_vocabulary(new_vocabulary=list(vocab))
    return model

def build_critic(cfg: Dict[str, Any], device: torch.device) -> CriticModel:
    c = cfg["critic"]
    n_mels = int(cfg["preprocessor"]["features"])
    crit = CriticModel(
        n_mel=n_mels,
        head_hidden=int(c.get("hidden_dim")),
        audio_conv_layers=int(c.get("layers")),
        dropout=float(c.get("dropout")),
    ).to(device)
    return crit

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLNF training (config-driven, RLNFTrainer)")
    p.add_argument("--config", type=str, help="Path to YAML config")
    p.add_argument("--write-example", type=str, help="Write example YAML and exit")
    return p.parse_args()

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()

    if args.write_example:
        write_example_config(args.write_example)
        return

    if not args.config:
        raise SystemExit("--config is required (or use --write-example)")

    cfg = load_config(args.config)

    # Repro / device
    trn = cfg.get("training", {})
    seed = int(trn.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = pick_device(trn.get("device", "auto"))
    use_bf16 = (str(trn.get("precision", "fp32")).lower() == "bf16")
    amp = (device.type == "cuda") and use_bf16  # pass to PPO/Trainer

    # Save dir
    save_dir = trn.get("save_dir", "./rlnf_runs/exp")
    os.makedirs(save_dir, exist_ok=True)

    # Build components
    actor = build_actor(cfg)
    critic = build_critic(cfg, device)

    tokenizer = load_tokenizer(cfg["paths"]["tokenizer_path"])

    # Reward model
    reward_path = cfg["paths"]["reward_path"]
    reward_model = RewardModel.from_pretrained(reward_path)
    
    reward_model = reward_model.to(device)  # type: ignore

    data = cfg["data"]
    wandb_cfg = cfg.get("wandb", {})
    pin_memory = bool(trn.get("pin_memory", False))

    trainer = RLNFTrainer(
        reward_model=reward_model,
        critic_model=critic,
        asr_model=actor,
        train_manifest=data["train_manifest"],
        val_manifest=data["valid_manifest"],
        audio_preprocessor_config=cfg["preprocessor"],
        batch_size=int(data.get("batch_size", 4)),
        epochs=int(trn.get("epochs", 3)),
        num_workers=int(data.get("num_workers", 0)),
        sp_tokenizer=tokenizer,
        device=device,
        wandb_logging=bool(wandb_cfg.get("enable", False)),
        wandb_project=str(wandb_cfg.get("project", "rlnf")),
        run_name=str(wandb_cfg.get("run_name", "exp")),
        K_updates=int(trn.get("K_updates", 4)),
        actor_lr=float(trn.get("actor_lr", 1e-5)),
        critic_lr=float(trn.get("critic_lr", 1e-4)),
        clip_eps=float(trn.get("clip_eps", 0.2)),
        pin_memory=pin_memory,
        amp=amp,
    )

    # Train
    trainer.train()
    print(f"Training complete. Models saved in: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
    main()
