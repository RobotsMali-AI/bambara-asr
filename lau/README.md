# LAU: Listen, Attend, Understand

This directory implements **Listen, Attend, Understand (LAU)**, a semantic regularization method for low-resource end-to-end speech translation with high-variance labels. The experiments translate Bambara audio directly into French and accompany the paper [“Listen, Attend, Understand”](https://arxiv.org/abs/2601.01121).

## Approach

`HybridRNNTCTCLAUModel` extends NeMo's hybrid RNNT/CTC model with a projection head attached to the FastConformer encoder. During training, an auxiliary MSE or cosine loss aligns projected acoustic representations with frozen French sentence embeddings. The semantic branch regularizes training; inference still uses the TDT or CTC decoder.

The released [`lau-soloni-114m-mse-k1`](https://huggingface.co/RobotsMali/lau-soloni-114m-mse-k1) checkpoint uses MSE regularization with weight 1. The unregularized [`st-soloni-114m-tdt-ctc`](https://huggingface.co/RobotsMali/st-soloni-114m-tdt-ctc) model is the baseline.

## Layout

- `hybrid_rnnt_ctc_lau_models.py`: custom NeMo model class and semantic losses.
- `train_lau.py`: config-driven training entry point.
- `config/`: versioned MSE/cosine and loss-weight experiments.
- `eval/`: transcription, clustering, keyword, QA, summarization, and judging analyses.
- `tokenizer/`: French-output SentencePiece resources.
- `requirements.txt`: NeMo 2.5.0 and semantic-embedding dependencies.

## Setup and Training

```bash
git clone https://github.com/RobotsMali-AI/bambara-asr.git
cd bambara-asr
python -m venv .venv
source .venv/bin/activate
pip install -r lau/requirements.txt
python lau/train_lau.py --config lau/config/soloni-lau-v2.2.0.yaml
```

Run from the repository root and update manifest, checkpoint, device, and W&B paths in the config. Training uses Bambara audio/French translation pairs from [`RobotsMali/jeli-asr`](https://huggingface.co/datasets/RobotsMali/jeli-asr). Evaluation artifacts are published in [`RobotsMali/lau-eval`](https://huggingface.co/datasets/RobotsMali/lau-eval).

The evaluation utilities cover BLEU/WER/CER, encoder parameter drift, and semantic cluster purity/NMI. Consult each script before use; several analyses require external embedding or language models.

## Citation

```bibtex
@misc{diarra2026listenattendunderstandregularization,
  title={Listen, Attend, Understand: a Regularization Technique for Stable E2E Speech Translation Training on High Variance Labels},
  author={Yacouba Diarra and Michael Leventhal},
  year={2026},
  eprint={2601.01121},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.01121}
}
```
