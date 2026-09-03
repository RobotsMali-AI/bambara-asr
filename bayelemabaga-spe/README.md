# Bayɛlɛmabaga SentencePiece Tokenizer

This directory contains a SentencePiece model and vocabulary trained on Bambara text from [`RobotsMali/transcription-scorer`](https://huggingface.co/datasets/RobotsMali/transcription-scorer) and a subset of [`RobotsMaliAI/bayelemabaga`](https://huggingface.co/datasets/RobotsMaliAI/bayelemabaga).

The files were generated with [`utils/python/process_asr_text_tokenizer.py`](../utils/python/process_asr_text_tokenizer.py):

- `tokenizer.model`: serialized SentencePiece tokenizer.
- `tokenizer.vocab`: SentencePiece vocabulary with scores.
- `vocab.txt`: plain token list for compatible consumers.

Keep all three files together when reproducing experiments. Token IDs are model-specific; do not substitute a newly trained vocabulary into an existing checkpoint without changing and retraining its decoder.
