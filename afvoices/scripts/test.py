#!/usr/bin/env python3
"""
test_asr_model_best.py

Run NeMo ASR model testing using greedy_batch decoding (default)
and optionally load weights from a specific best checkpoint.

Usage:
  python test_asr_model_best.py \
      /path/to/model.nemo \
      /path/to/log_dir \
      --best_ckpt /path/to/epoch=35-step=94428.ckpt \
      --devices 1 \
      --accelerator gpu

If --best_ckpt is provided:
  - Restore .nemo model
  - Load weights from the checkpoint
  - Run test
  - Save the updated model to best_models/best-{model_name}.nemo
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr
from nemo.lightning import Trainer as NLTrainer
from omegaconf import OmegaConf


def parse_devices(devices_arg: str):
    """Parse device argument for Lightning."""
    devices_arg = str(devices_arg).strip()
    if devices_arg == "-1":
        return -1
    if "," in devices_arg:
        return [int(x) for x in devices_arg.split(",") if x.strip()]
    return [int(devices_arg)]


def load_best_checkpoint(model, ckpt_path):
    """Load weights from a specified checkpoint into the restored model."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading best checkpoint weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle common NeMo checkpoint key format
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print("✅ Checkpoint weights successfully loaded.")


def derive_model_name(nemo_path):
    """Extract base model name from .nemo archive."""
    name = Path(nemo_path).stem  # e.g., soloba-ctc-v2.5.0
    parts = name.split("-v")
    return parts[0] if parts else name


def main():
    parser = argparse.ArgumentParser(description="Evaluate a NeMo ASR model with optional best checkpoint loading.")
    parser.add_argument("restore_path", type=str, help="Path to .nemo model archive.")
    parser.add_argument("log_dir", type=str, help="Directory where logs/results will be saved.")
    parser.add_argument("--best_ckpt", type=str, default=None, help="Optional path to best checkpoint to load.")
    parser.add_argument("--devices", type=str, default="-1", help='Devices: e.g. "1" or "0,1". Use "-1" for all.')
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"], help="Accelerator type.")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar.")
    args = parser.parse_args()

    # Prepare logging directory
    run_dir = Path(args.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log both console and file
    console_log = run_dir / "console.log"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.stdout, open(console_log, "w", buffering=1))
    sys.stderr = sys.stdout

    print("=== NeMo ASR Evaluation ===")
    print(f"Restore path : {args.restore_path}")
    print(f"Best ckpt    : {args.best_ckpt}")
    print(f"Devices      : {args.devices}")
    print(f"Accelerator  : {args.accelerator}")
    print(f"Run dir      : {run_dir}")
    print()

    # Restore base model
    model = nemo_asr.models.ASRModel.restore_from(restore_path=args.restore_path)

    # Load best checkpoint weights if provided
    if args.best_ckpt:
        load_best_checkpoint(model, args.best_ckpt)

    # Ensure model in test mode
    model.eval()
    model.freeze()

    # Save test configuration snapshot
    test_cfg = model.cfg.get("test_ds", None)
    if test_cfg is None:
        raise RuntimeError("Model config missing test_ds section.")
    cfg_path = run_dir / "test_config.yaml"
    with open(cfg_path, "w") as f:
        f.write(OmegaConf.to_yaml(test_cfg))
    print(f"Saved test config to: {cfg_path}")

    manifest = getattr(test_cfg, "manifest_filepath", None)
    print(f"Manifest: {manifest}\n")

    # Prepare data & trainer
    model.setup_test_data(test_cfg)
    devices_parsed = parse_devices(args.devices)
    trainer = NLTrainer(
        devices=devices_parsed,
        accelerator=args.accelerator,
        enable_progress_bar=not args.no_progress,
    )

    # Run test
    print("Running test...")
    results = trainer.test(model)
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Save new best model
    if args.best_ckpt:
        out_dir = Path("best_models")
        out_dir.mkdir(exist_ok=True)
        model_name = derive_model_name(args.restore_path)
        out_path = out_dir / f"best-{model_name}.nemo"
        model.save_to(out_path)
        print(f"✅ Saved best model to: {out_path}")

    print(f"\nAll logs and outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()

