# Kunkado ASR Experiments

This directory contains training configs, scripts, and tokenizers for fine-tuning Bambara ASR on [`RobotsMali/kunkado`](https://huggingface.co/datasets/RobotsMali/kunkado). Kunkado represents present-day speech from Bamako and nearby areas: spontaneous radio/TV recordings, noise and music, overlapping speakers, and frequent French or Arabic code-switching. About 40 of its 161 hours are human-reviewed.

The first runs adapted the repository's `v0` checkpoints into `v1` releases. Later configs under `v3-config/` adapted AfVoices `v2` checkpoints into `v3` releases, replacing the old “coming soon” plan.

## Layout

- `config/`: versioned QuartzNet, Soloba CTC, and Soloni configs for `v1` experiments.
- `v3-config/`: later fine-tuning configs based on `v2` checkpoints.
- `scripts/train.py` and `scripts/test.py`: NeMo training and evaluation entry points.
- `scripts/`: normalization, tag cleanup, and number-processing helpers.
- `tokenizers/`: Soloba and Soloni SentencePiece resources.
- `requirements.txt`: pinned NeMo 2.5.0 environment.

## Run an Experiment

Execute from the repository root and replace local manifest/checkpoint paths in the selected YAML:

```bash
pip install -r kunkado/requirements.txt
python kunkado/scripts/train.py \
  --config kunkado/config/soloni/soloni-v1.5.0.yaml
python kunkado/scripts/test.py --help
```

Supervised releases use the `human-reviewed` split. The `semi-first` and `semi-second` subsets retain automatically generated labels and are better suited to explicitly labelled semi-supervised experiments. Normalize numbers and tags consistently with the target card before comparing metrics.

Released checkpoint lineage and WER/CER are documented in the [RobotsMali model cards](https://huggingface.co/RobotsMali/models). Kunkado's noisy, code-switched conditions complement the cleaner read-speech and AfVoices experiments, but results across those datasets are not directly interchangeable.
