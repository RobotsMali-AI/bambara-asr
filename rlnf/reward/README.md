# RLNF Reward Model

The reward module trains a reference-free scorer for Bambara ASR hypotheses. `RewardModel` combines a convolutional audio encoder with a SentencePiece/LSTM text encoder and maps their concatenated representation to one scalar quality score.

## Files

- `reward_model.py`: architecture plus save/load helpers.
- `train_reward_model.py`: YAML-configured training entry point.
- `train_utils.py`: training, validation, and checkpoint helpers.
- `config/default.yaml`: example paths and hyperparameters.

From the repository root:

```bash
python rlnf/reward/train_reward_model.py \
  --config rlnf/reward/config/default.yaml
```

Manifests require `audio_filepath`, `transcription`, and `score` (0–100). Update all paths and inspect the tokenizer pad ID before training. The published baseline is [`RobotsMali/reward-model`](https://huggingface.co/RobotsMali/reward-model), trained with [`RobotsMali/transcription-scorer`](https://huggingface.co/datasets/RobotsMali/transcription-scorer).

Human scores are subjective and only partially reviewed; the resulting model is not a calibrated universal ASR metric. Use it within the documented data distribution and compare its judgments against held-out human ratings.
