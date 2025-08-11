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
        "precision": "fp32",           # fp32 | "bf16"
        "clip_eps": 0.2,
        "actor_lr": 1e-5,
        "critic_lr": 1e-4,
        "epochs": 3,
        "K_updates": 4,   # PPO updates per batch
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
        "name": "RobotsMali/stt-bm-quartznet15x5-V0",      # QuartzNet15x5Base-En
        "vocab": None 
    },

    "critic": {
        "hidden_dim": 256,
        "layers": 3,
        "dropout": 0.1,
    },

    "data": {
        "train_manifest": "path/to/train_rollouts.jsonl",
        "valid_manifest": "path/to/valid_rollouts.jsonl",
        "batch_size": 4,
        "num_workers": 0
    },

    "preprocessor": {
        'normalize': 'per_feature',
        'window_size': 0.02,
        'sample_rate': 16000,
        'window_stride': 0.01,
        'window': 'hann',
        'features': 64,
        'n_fft': 512,
        'frame_splicing': 1,
        'dither': 1e-05,
        'stft_conv': False
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

# Load a pre-trained Tokenizer
def load_tokenizer(model_path: str) -> SentencePieceProcessor:
    sp = SentencePieceProcessor()
    sp.Load(model_path)
    return sp

# -----------------------------
# Builders
# -----------------------------

def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def build_actor(cfg: Dict[str, Any]) -> nemo_asr.models.ASRModel:
    m = cfg["model"]
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=m["name"])  # type: ignore
    vocab = m.get("vocab")
    model.change_vocabulary(new_vocabulary=list(vocab))
    return model


def build_critic(cfg: Dict[str, Any], device: torch.device) -> CriticModel:
    c = cfg["critic"]
    n_mels = cfg["preprocessor"]["features"]

    crit = CriticModel(
        n_mel=int(n_mels),
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
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))
    device = pick_device(cfg.get("device", "auto"))

    # Save dir
    save_dir = cfg["training"].get("save_dir")
    os.makedirs(save_dir, exist_ok=True)

    # Build components
    actor = build_actor(cfg)
    critic = build_critic(cfg, device)

    tokenizer = load_tokenizer(cfg["paths"]["tokenizer_path"])
    reward_model = RewardModel.from_pretrained(cfg["paths"]["tokenizer_path"])

    trainer = RLNFTrainer(
        reward_model=reward_model,
        critic_model=critic,
        asr_model=actor,
        train_manifest=cfg["data"]["train_manifest"],
        val_manifest=cfg["data"]["valid_manifest"],
        audio_preprocessor_config=cfg["preprocessor"],
        batch_size=cfg["data"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        num_workers=cfg["data"]["num_workers"],
        sp_tokenizer=tokenizer,
        device=device,
        wandb_logging=cfg["wandb"]["enable"],
        wandb_project=cfg["wandb"]["project"],
        run_name=cfg["wandb"]["run_name"],
        K_updates = cfg["training"]["K_updates"],
        actor_lr = cfg["training"]["actor_lr"],
        critic_lr = cfg["training"]["critic_lr"],
        clip_eps = cfg["training"]["clip_eps"],
        pin_memory = cfg["training"]["pin_memory"]
    )

    # Delegate training entirely to RLNFTrainer
    trainer.train()

    print("Training complete. Saving actor and critic models in the current directory")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
    main()
