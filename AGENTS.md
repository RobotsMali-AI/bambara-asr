# Repository Guidelines

## Project Structure & Module Organization

This repository groups Bambara speech research by experiment. `afvoices/`, `kunkado/`, `an-be-kalan-bench/`, `quartznum/`, and `lau/` contain training scripts, YAML configurations, and experiment-specific resources. `slu-ic-slot-filling/` covers spoken-language understanding. The installable RLHF prototype lives in `rlnf/`, with `dataloaders/`, `ppo/`, and `reward/` subpackages. Shared preprocessing, tokenizer, logging, and job helpers belong in `utils/`. Model and dataset cards are under `docs/`; tokenizers and small manifests remain beside the experiment that consumes them.

Read the nearest module README before changing an experiment. Keep generated checkpoints, downloaded audio, W&B runs, and credentials out of version control.

## Build, Test, and Development Commands

Use Python 3.10 or newer and create an isolated environment. Dependencies are scoped by module rather than managed at the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r rlnf/requirements.txt
pip install -e rlnf
pip install -r afvoices/requirements.txt
python afvoices/scripts/train.py --config afvoices/config/soloba/soloba-ctc-v2.6.0.yaml
python afvoices/scripts/test.py --help
```

Install only the requirements needed for the experiment. Most training and evaluation commands require NVIDIA NeMo, PyTorch, datasets or manifests, and often CUDA-capable hardware. Consult each subdirectory README for required paths and CLI overrides.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, imports grouped by standard library, third-party, then local modules, and docstrings for non-obvious functions. Use `snake_case` for functions and variables, `PascalCase` for classes, and descriptive CLI flags. Name configs with the model and version pattern already used nearby, such as `soloba-ctc-v2.6.0.yaml`. Prefer reusable utilities in `utils/python/` over duplicating preprocessing logic.

## Testing Guidelines

There is no repository-wide automated test suite or enforced coverage threshold. Treat module evaluation scripts as integration checks. Before submitting, run `python -m compileall` on changed Python directories, invoke changed CLIs with `--help`, and execute the smallest relevant training or evaluation smoke test. Document hardware, dataset split, config file, and resulting metrics for model-affecting changes.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `Add configs ...`, `Update ...`, and `Fix bugs: ...`. Keep each commit focused and avoid committing model artifacts or local paths. Pull requests should explain the experiment or bug, list changed configs and validation commands, link related issues, and report WER or other relevant metrics. Include logs or screenshots only when they clarify behavior, and call out dependency or data-access changes explicitly.
