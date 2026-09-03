# QuartzNum

QuartzNum is a narrow Bambara ASR experiment for spoken amounts, account numbers, and phone numbers. The released [`RobotsMali/quartznum-v0`](https://huggingface.co/RobotsMali/quartznum-v0) model fine-tunes the 18M-parameter QuartzNet 15x5 checkpoint `stt-bm-quartznet15x5-v2` with a character CTC vocabulary. It is used by [`mobilebamspeech`](https://github.com/RobotsMali-AI/mobilebamspeech) and the [`mobileBAMking`](https://github.com/RobotsMali-AI/mobileBAMking) proof of concept.

## Files

- `config/quartznum.yaml`: data loaders, optimizer, early stopping, and output paths.
- `train_quartznum.py`: NeMo 2.5.0 training entry point.
- `results.json`: archived test output for the released checkpoint.

The speech clips are derived from the banking voice collection processed by `slu-ic-slot-filling/prepare_slu_manifest.py`. That script extracts transfer amounts and account/phone numbers, then normalizes their Bambara labels.

## Train and Evaluate

Run from the repository root. Install a NeMo 2.5.0 ASR environment (for example, the pinned `afvoices/requirements.txt`), prepare `train-quartznum.jsonl` and `test-quartznum.jsonl`, then update their paths in the config:

```bash
python quartznum/train_quartznum.py --config quartznum/config/quartznum.yaml
python afvoices/scripts/test.py --help
```

The archived evaluation reports greedy CTC WER of **17.24%**. This number measures transcription error on the held-out narrow-domain split; it is not evidence of general Bambara ASR quality. Validate exact numeric parsing in the target application, especially for long digit sequences and noisy recordings.
