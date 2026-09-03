# Shared Utilities

Reusable helpers for the repository's ASR experiments:

- `python/preprocessing.py`: manifest/audio validation and channel conversion.
- `python/process_asr_text_tokenizer.py`: SentencePiece tokenizer preparation.
- `python/hf_to_nemo_asr.py`: Hugging Face dataset to NeMo JSONL conversion.
- `python/helpers.py` and `python/wandb.py`: configuration, model, and logging helpers.
- `python/hf_cards.py`: pull and push Hugging Face README cards tracked by `docs/huggingface.yaml`.
- `bash/install_dependencies.sh` and `bash/check_setup.sh`: environment setup checks.
- `bash/run_jobs.sh`: sequential experiment launcher.

Run Python entry points from the repository root so `utils.python` imports resolve. Inspect each command before use because training paths and credentials are environment-specific.

## Hugging Face Cards

Pull before editing when a card may have changed on the Hub, then push the reviewed local copy:

```bash
python utils/python/hf_cards.py pull model RobotsMali/MODEL_ID
python utils/python/hf_cards.py push model RobotsMali/MODEL_ID
python utils/python/hf_cards.py pull dataset RobotsMali/DATASET_ID
```

Only `README.md` is synchronized. Authentication and write access are required for pushes.
