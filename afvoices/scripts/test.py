#!/usr/bin/env python3
"""
test_asr_model.py

Run NeMo ASR model testing with a selectable decoding strategy and logging.

Features
--------
- Choose decoding strategy: beam_search or greedy_batch
- (Hybrid RNNT+CTC) also applies decoding to aux CTC decoder
- Prints test config and the decoding strategy BEFORE testing
- Saves all outputs (config snapshot + results JSON + console log) to a run dir

Usage
-----
python test_asr_model.py \
  /path/to/model.nemo \
  --strategy beam_search \
  --beam-size 8 \
  --devices 1 \
  --accelerator gpu \
  --log-dir ./test_logs

Notes
-----
- For beam search, we set:
    decoding.strategy = "beam_search"
    decoding.beam.beam_size = <beam-size>
    decoding.beam.return_best_hypothesis = True
- For greedy, we set:
    decoding.strategy = "greedy_batch"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nemo.collections.asr as nemo_asr
from nemo.lightning import Trainer as NLTrainer
from omegaconf import OmegaConf

# Optional: handle EncDecHybridRNNTCTCBPEModel specifically
try:
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
except Exception:
    EncDecHybridRNNTCTCBPEModel = tuple()  # fallback if import not available


def parse_devices(devices_arg: str):
    """
    Accepts:
      - single int like "1" or "0"
      - comma list like "0,1"
    Returns:
      - list[int] for Lightning's devices
    """
    devices_arg = str(devices_arg).strip()
    if devices_arg == "-1":
        return -1
    if "," in devices_arg:
        return [int(x) for x in devices_arg.split(",") if x.strip() != ""]
    return [int(devices_arg)]

def apply_decoding_strategy(model, strategy: str, beam_size: int):
    """
    Apply decoding strategy to the model. If model is EncDecHybridRNNTCTCBPEModel,
    also apply to aux_ctc.decoding and reassign explicitly.
    """
    # ---- Main decoder ----
    decoding_cfg = model.cfg.decoding
    if strategy == "beam_search":
        decoding_cfg.strategy = "beam"
        if hasattr(decoding_cfg, "beam"):
            decoding_cfg.beam.beam_size = beam_size
            decoding_cfg.beam.return_best_hypothesis = True
    elif strategy == "greedy_batch":
        decoding_cfg.strategy = "greedy_batch"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply through NeMo helper (updates model-side decoding objects)
    model.change_decoding_strategy(decoding_cfg=decoding_cfg)

    # ---- Hybrid: aux CTC decoder ----
    try:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    except Exception:
        EncDecHybridRNNTCTCBPEModel = tuple()  # if not present

    if isinstance(model, EncDecHybridRNNTCTCBPEModel):
        aux_cfg = getattr(model.cfg, "aux_ctc", None)
        if aux_cfg is not None and getattr(aux_cfg, "decoding", None) is not None:
            aux_dec = aux_cfg.decoding

            if strategy == "beam_search":
                aux_dec.strategy = "beam"
                if hasattr(aux_dec, "beam"):
                    aux_dec.beam.beam_size = beam_size
                    aux_dec.beam.return_best_hypothesis = True
            else:
                aux_dec.strategy = "greedy_batch"

            # ✅ Explicitly write back to the config (clarity + safety)
            model.cfg.aux_ctc.decoding = aux_dec

            # If the model exposes a separate aux decoding object, mirror it too
            # (Some NeMo versions keep an instantiated ctc_decoding module.)
            if hasattr(model, "ctc_decoding") and model.ctc_decoding is not None:
                # Best-effort sync for common attributes
                try:
                    if strategy == "beam_search":
                        model.ctc_decoding.strategy = "beam"
                        if hasattr(model.ctc_decoding, "beam"):
                            model.ctc_decoding.beam.beam_size = beam_size
                            model.ctc_decoding.beam.return_best_hypothesis = True
                    else:
                        model.ctc_decoding.strategy = "greedy_batch"
                except Exception:
                    # Non-fatal: some versions may not expose these attrs
                    pass


def main():
    parser = argparse.ArgumentParser(description="Test a NeMo ASR model with configurable decoding.")
    parser.add_argument("restore_path", type=str, help="Path to .nemo (or checkpoint) to restore.")
    parser.add_argument("log_dir", type=str, help="Directory where run logs/results will be saved.")
    parser.add_argument("--strategy", type=str, default="greedy_batch",
                        choices=["beam_search", "greedy_batch"],
                        help="Decoding strategy to use.")
    parser.add_argument("--beam_size", type=int, default=8, help="Beam size (when using beam_search).")
    parser.add_argument("--devices", type=str, default="1",
                        help='Devices: e.g. "1" or "0,1". Use "-1" for all available.')
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="Accelerator for Lightning Trainer.")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar.")
    args = parser.parse_args()

    # Prepare run directory
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.log_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout/err to a log file as well
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

    print("=== NeMo ASR Test Runner ===")
    print(f"Restore path     : {args.restore_path}")
    print(f"Strategy         : {args.strategy}")
    if args.strategy == "beam_search":
        print(f"Beam size        : {args.beam_size}")
    print(f"Devices          : {args.devices}")
    print(f"Accelerator      : {args.accelerator}")
    print(f"Run dir          : {run_dir}")
    print()

    # Restore model
    model = nemo_asr.models.ASRModel.restore_from(restore_path=args.restore_path)

    # Snapshot test config to file and show a short preview
    test_cfg = model.cfg.test_ds if hasattr(model.cfg, "test_ds") else None
    if test_cfg is None:
        raise RuntimeError("Model config has no test_ds section.")

    # Save full test config to YAML
    cfg_path = run_dir / "test_config.yaml"
    with open(cfg_path, "w") as f:
        f.write(OmegaConf.to_yaml(test_cfg))
    # Print a concise preview
    manifest = getattr(test_cfg, "manifest_filepath", None)
    batch_size = getattr(test_cfg, "batch_size", None)
    num_workers = getattr(test_cfg, "num_workers", None)
    print("=== Test Config (preview) ===")
    print(f"manifest_filepath: {manifest}")
    print(f"batch_size       : {batch_size}")
    print(f"num_workers      : {num_workers}")
    print()

    # Apply decoding strategy (and aux_ctc if hybrid)
    apply_decoding_strategy(model, args.strategy, args.beam_size)

    # Also persist an explicit snapshot of decoding choices
    decode_snapshot = {
        "strategy": args.strategy,
        "beam_size": args.beam_size if args.strategy == "beam_search" else None,
        "hybrid_aux_ctc_adjusted": bool(isinstance(model, EncDecHybridRNNTCTCBPEModel)),
    }
    with open(run_dir / "decoding_used.json", "w") as f:
        json.dump(decode_snapshot, f, indent=2)

    # Prepare test data and trainer
    model.setup_test_data(test_cfg)

    devices_parsed = parse_devices(args.devices)
    trainer = NLTrainer(
        devices=devices_parsed,
        accelerator=args.accelerator,
        enable_progress_bar=not args.no_progress,
    )

    # Run test
    results_path = run_dir / "results.json"
    results = None
    if hasattr(model.cfg, "test_ds") and getattr(model.cfg.test_ds, "manifest_filepath", None):
        if model.prepare_test(trainer):
            print("Running test...")
            results = trainer.test(model)
            print("=== Test Results ===")
            print(results)
        else:
            print("model.prepare_test(trainer) returned False; skipping test.")
    else:
        print("No test_ds.manifest_filepath configured; skipping test.")

    # Save results if any
    if results is not None:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nLogs and outputs saved to: {run_dir}")
    # Ensure console log file closes
    for s in sys.stdout.streams:
        if s is not sys.__stdout__:
            try:
                s.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

