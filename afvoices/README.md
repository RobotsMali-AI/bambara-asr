# African Next Voices ASR Experiments

This directory reproduces the Bambara ASR experiments run on the pre-completion subset of [`RobotsMali/afvoices`](https://huggingface.co/datasets/RobotsMali/afvoices). Those experiments produced the repository's `v2` QuartzNet, Soloba CTC, Soloba TDT, and Soloni hybrid TDT-CTC releases. They use NVIDIA NeMo 2.5.0.

The final Hugging Face dataset is larger than the subset used for the reported results. Use [`pre-manifests/`](pre-manifests/) to reconstruct the exact train/test partition used in the [African Next Voices dataset paper](https://arxiv.org/abs/2511.18557).

## Layout

- `config/`: versioned YAML configurations grouped by architecture.
- `scripts/train.py`: shared NeMo training entry point.
- `scripts/test.py`: checkpoint evaluation and best-model export.
- `scripts/hf_to_nemo_asr.py`: converts Hugging Face audio data to NeMo manifests.
- `scripts/`: additional normalization, inspection, and data-download helpers.
- `tokenizers/`: SentencePiece models and source corpora for Soloba and Soloni.
- `pre-manifests/`: the archived paper-era train/test manifests.

## Setup and Use

Run commands from the repository root so imports from `utils` resolve correctly:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r afvoices/requirements.txt
python afvoices/scripts/train.py \
  --config afvoices/config/soloni/soloni-v2.2.0.yaml
python afvoices/scripts/test.py --help
```

Config paths are experiment records, not portable defaults: update manifest, tokenizer, checkpoint, device, and W&B settings before training. Several releases were produced through sequential configs, so consult the corresponding [Hugging Face model card](https://huggingface.co/RobotsMali/models) for the final checkpoint's lineage and metrics.
