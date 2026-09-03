# An Bɛ Kalan ASR Benchmark

Experiments for adapting Bambara ASR to children's speech and readings from RobotsMali's GAIFE educational books. The released [`soloni-be-kalan-v0`](https://huggingface.co/RobotsMali/soloni-be-kalan-v0) checkpoint is based on `soloni-114m-tdt-ctc-v2` and evaluated on [`RobotsMali/an-be-kalan-bench`](https://huggingface.co/datasets/RobotsMali/an-be-kalan-bench).

## Contents

- `config/quartznet/` and `config/soloni/`: experiment-specific NeMo YAML files.
- `scripts/train.py`: shared ASR fine-tuning entry point.
- `scripts/test.py`: evaluates `.nemo` archives and optional Lightning checkpoints.
- `scripts/hf_to_nemo_asr.py`: prepares NeMo JSONL manifests from Hugging Face data.
- `requirements.txt`: pinned NeMo 2.5.0 training environment.

## Reproduce an Experiment

Run from the repository root and adjust local manifest/checkpoint paths in the selected config:

```bash
pip install -r an-be-kalan-bench/requirements.txt
python an-be-kalan-bench/scripts/train.py \
  --config an-be-kalan-bench/config/soloni/soloni-be-kalan-exp4.yaml
python an-be-kalan-bench/scripts/test.py --help
```

The dataset contains a small `main` split and a much larger `duplicate` split with repeated book text read by many speakers. Preserve book-level separation when creating evaluation splits; otherwise repeated sentences can inflate results. See the dataset and model cards for cohort-level limitations and reported WER/CER.
