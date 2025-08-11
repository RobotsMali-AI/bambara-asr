#!/usr/bin/env python3
"""
Unified ASR Training Script (QuartzNet, Parakeet Hybrid, etc.)
--------------------------------------------------------------
- Loads any NeMo ASR model via the base class `ASRModel`.
- Supports first fine-tuning vs. continued training via `config.model.first_training`.
  * first_training = True  -> load with `from_pretrained()` and (optionally) change vocabulary/tokenizer
  * first_training = False -> load with `restore_from()` and DO NOT change vocabulary
- Optional CLI overrides for SpecAugment knobs and auxiliary CTC loss weight (hybrid models).

Usage
~~~~~
python unified_asr_training.py --config <config.yaml> \
    [--freq-masks 4 --freq-width 27 --time-masks 10 --time-width 0.1] \
    [--rect-freq 50 --rect-time 120 --rect-masks 10] \
    [--aux-ctc-weight 0.6]

Notes
~~~~~
- If you set SpecAugment args, they are applied only if the model exposes `model.cfg.spec_augment`.
- `--aux-ctc-weight` is applied only if `model.cfg.aux_ctc.ctc_loss_weight` exists (hybrid models).
- Tokenizer/vocabulary change on first training:
    * If `config.tokenizer.path` and `config.tokenizer.type` exist -> use `change_vocabulary(new_tokenizer_dir=..., new_tokenizer_type=...)`.
    * Else if `config.model.new_vocab` (list[str]) exists -> use `change_vocabulary(new_vocabulary=...)`.

"""
from __future__ import annotations

import argparse
import sys, os
from typing import Any, Dict

# Utils (project-specific)
from utils.python.preprocessing import check_and_convert_audio_channels
from utils.python.helpers import load_config, enable_bn_se
from utils.python.wandb import MyWandbLogger as WandbLogger

# Lightning / callbacks
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# W&B
import wandb

# NeMo
import nemo.collections.asr as nemo_asr
import nemo.lightning as nl
from nemo.lightning import AutoResume


# -----------------------------
# Helpers
# -----------------------------

def maybe_apply_spec_augment(model: nemo_asr.models.ASRModel, args: argparse.Namespace) -> None:
    """Apply SpecAugment overrides if the model exposes `cfg.spec_augment`.
    Safe no-ops if fields are missing.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None or not hasattr(cfg, "spec_augment"):
        return

    sa = cfg.spec_augment

    # Basic masks
    if args.freq_masks is not None:
        sa.freq_masks = args.freq_masks
    if args.freq_width is not None:
        sa.freq_width = args.freq_width
    if args.time_masks is not None:
        sa.time_masks = args.time_masks
    if args.time_width is not None:
        sa.time_width = args.time_width

    # Rectangular masks (for some models)
    if args.rect_freq is not None:
        setattr(sa, "rect_freq", args.rect_freq)
    if args.rect_time is not None:
        setattr(sa, "rect_time", args.rect_time)
    if args.rect_masks is not None:
        setattr(sa, "rect_masks", args.rect_masks)

    # Some models need the augmentation module re-instantiated from config
    try:
        model.spec_augmentation = model.from_config_dict(sa)
    except Exception:
        # If the model doesn't support building aug from dict, just keep cfg
        pass


def maybe_set_aux_ctc_weight(model: nemo_asr.models.ASRModel, weight: float | None) -> None:
    if weight is None:
        return
    cfg = getattr(model, "cfg", None)
    if cfg is None or not hasattr(cfg, "aux_ctc"):
        return
    aux_cfg = cfg.aux_ctc
    if hasattr(aux_cfg, "ctc_loss_weight"):
        aux_cfg.ctc_loss_weight = float(weight)


def load_model_from_config(config: Any) -> nemo_asr.models.ASRModel:
    """Load an ASR model according to config.model.first_training.
    - first_training=True: from_pretrained() + optional vocabulary/tokenizer change
    - first_training=False: restore_from() (no vocab change)
    """
    first_training = bool(getattr(config.model, "first_training", False))

    if first_training:
        # from_pretrained via base class for broad model support
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=config.model.name)

        # Optionally change vocabulary/tokenizer for first training only
        # Prefer tokenizer spec if provided; otherwise allow char-level vocab
        has_tok_dir = hasattr(config, "tokenizer") and hasattr(config.tokenizer, "path") and hasattr(config.tokenizer, "type")
        has_char_vocab = hasattr(config.tokenizer, "vocab") and isinstance(config.tokenizer.vocab, list)

        if has_tok_dir:
            model.change_vocabulary(
                new_tokenizer_dir=config.tokenizer.path,
                new_tokenizer_type=config.tokenizer.type,
            )
        elif has_char_vocab:
            model.change_vocabulary(new_vocabulary=list(config.tokenizer.vocab))

    else:
        # Continued training: restore from local checkpoint without changing vocab
        restore_path = getattr(config.model, "name")

        if not os.path.exists(restore_path):
            raise ValueError("config.model.name is required to be an archive file when first_training=False")

        model = nemo_asr.models.ASRModel.restore_from(restore_path=restore_path)

    return model


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified NeMo ASR training")
    p.add_argument("--config", required=True, help="Path to YAML config")

    # SpecAugment knobs (optional; apply only if present in model cfg)
    p.add_argument("--freq-masks", type=int, default=None)
    p.add_argument("--freq-width", type=int, default=None)
    p.add_argument("--time-masks", type=int, default=None)
    p.add_argument("--time-width", type=float, default=None)
    p.add_argument("--rect-freq", type=int, default=None)
    p.add_argument("--rect-time", type=int, default=None)
    p.add_argument("--rect-masks", type=int, default=None)

    # Hybrid-only auxiliary CTC loss weight
    p.add_argument("--aux-ctc-weight", type=float, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load YAML config
    config = load_config(args.config)

    # Load model according to first/continued training policy
    print(f"Preparing model '{config.model.name}' (first_training={getattr(config.model, 'first_training')})")
    model = load_model_from_config(config)

    # Ensure audio is mono in manifests
    check_and_convert_audio_channels(config.data_loaders.train.manifest_filepath)
    check_and_convert_audio_channels(config.data_loaders.valid.manifest_filepath)
    check_and_convert_audio_channels(config.data_loaders.test.manifest_filepath)

    # Freeze/unfreeze encoder
    if getattr(config.training, "freeze_encoder", False):
        model.encoder.freeze()
        model.encoder.apply(enable_bn_se)
        print("Encoder frozen (BN/SE in train mode)")
    else:
        model.encoder.unfreeze()
        print("Encoder unfrozen")

    # Optimizer
    model.setup_optimization(optim_config=config.optim)

    # Datasets
    model.setup_training_data(train_data_config=config.data_loaders.train)
    model.setup_validation_data(val_data_config=config.data_loaders.valid)
    model.setup_test_data(test_data_config=config.data_loaders.test)

    # Optional overrides
    maybe_apply_spec_augment(model, args)
    maybe_set_aux_ctc_weight(model, args.aux_ctc_weight)

    # Logger & callbacks
    wandb_logger = WandbLogger(project=config.wandb.project, name=config.wandb.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.training.checkpoint_dir,
        save_weights_only=True,
        save_last=True,
        monitor="val_wer",
        mode="min",
        save_top_k=config.training.save_top_k,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_wer",
        mode="min",
        patience=config.training.patience,
        verbose=True,
    )

    trainer = nl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=config.training.precision,
        max_epochs=config.training.epochs,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        logger=wandb_logger,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # Auto-resume
    resume = AutoResume(
        resume_if_exists=config.training.resume_if_exists,
        resume_from_directory=config.training.checkpoint_dir,
        resume_ignore_no_checkpoint=config.training.resume_ignore_no_checkpoint,
    )
    resume.setup(trainer)

    # Train
    try:
        trainer.fit(model)
    except Exception:
        print("Training interrupted, finishing logging...")
        wandb.finish()

    # Save
    model.save_to(config.training.save_model_path)

    # Test
    if hasattr(model.cfg, "test_ds") and getattr(model.cfg.test_ds, "manifest_filepath", None):
        if model.prepare_test(trainer):
            trainer.test(model)

    wandb.finish()
    print(f"Done training. NeMo model saved to: {config.training.save_model_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
    main()
