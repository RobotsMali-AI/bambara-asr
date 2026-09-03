# RLNF Toolkit

> [!CAUTION]
> RLNF is an early research prototype and has been largely inactive since late 2025. It lags behind the repository's current ASR, speech-translation, and narrow-application work. The code is retained for reproducibility and experimentation, but it is not production-ready and may require maintenance for newer dependencies.

RLNF (“Reinforcement Learning from Nouhoum Feedback”) explores reinforcement learning from human feedback for automatic speech recognition. It combines a NeMo ASR actor, an audio critic, a learned audio/transcript reward model, and Proximal Policy Optimization (PPO).

## Package Layout

- `trainer.py`: coordinates rollouts, PPO updates, validation, and checkpoints.
- `dataloaders/`: loads and preprocesses JSONL audio/reward manifests.
- `reward/`: reward-model architecture and config-driven training.
- `ppo/`: critic, rollout, loss, and optimizer components.
- `train_rlnf.py`: config-driven RLNF entry point.
- `rlnf-config.yaml`: example experiment configuration.
- `rlnf.png`: high-level architecture diagram.

## Installation

Python 3.10 or newer is required. The pinned research environment uses NeMo 2.5.0:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r rlnf/requirements.txt
pip install -e rlnf
```

Run entry points from the repository root. Generate a fresh configuration or start from the checked-in example:

```bash
python rlnf/train_rlnf.py --write-example /tmp/rlnf-example.yaml
python rlnf/train_rlnf.py --config rlnf/rlnf-config.yaml
python rlnf/reward/train_reward_model.py --config rlnf/reward/config/default.yaml
```

Update every path before training. RLNF expects NeMo-style JSONL audio manifests, a SentencePiece tokenizer, and a serialized reward model. The companion human-feedback data and baseline checkpoint are published as [`RobotsMali/transcription-scorer`](https://huggingface.co/datasets/RobotsMali/transcription-scorer) and [`RobotsMali/reward-model`](https://huggingface.co/RobotsMali/reward-model).

## Limitations

This implementation was built to test hypotheses, not to provide a mature RLHF library. It has no repository-wide test suite, limited scale/performance validation, and strong assumptions about CTC actors, tokenizer padding, preprocessing, and reward-score quality. Treat generated checkpoints as experimental and establish supervised baselines before interpreting PPO gains.
