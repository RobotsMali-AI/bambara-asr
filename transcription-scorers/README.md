# Transcription Scorers

This directory stores the legacy `test-reward-model.rw` checkpoint used in early experiments on reference-free scoring of Bambara ASR hypotheses. It accompanies the [`RobotsMali/transcription-scorer`](https://huggingface.co/datasets/RobotsMali/transcription-scorer) human-feedback dataset and the reward-model code under [`rlnf/reward/`](../rlnf/reward/).

The checkpoint predicts a scalar quality score from audio and a candidate transcript. It is a research artifact, not a calibrated replacement for WER or human review, and its narrow training distribution limits comparisons across domains and model families. For architecture and loading examples, see the [`reward-model` card](https://huggingface.co/RobotsMali/reward-model).
