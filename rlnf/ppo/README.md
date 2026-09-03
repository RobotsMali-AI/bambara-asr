# RLNF PPO Module

This package contains the prototype Proximal Policy Optimization path used by `RLNFTrainer`:

- `rollout.py`: transcribes an audio batch, scores hypotheses with the reward model, and computes CTC sequence log-probabilities.
- `critic_network.py`: predicts a scalar value from acoustic features.
- `loss.py`: implements the clipped PPO surrogate and value losses.
- `optimizer.py`: performs repeated actor/critic updates, entropy regularization, mixed-precision handling, finite-value checks, and gradient clipping.

The ASR model is the actor. Rewards come from the learned audio/transcript scorer, while the critic supplies the baseline used to compute advantages. Configuration such as `clip_eps`, `K_updates`, actor/critic learning rates, and precision is passed through `rlnf/rlnf-config.yaml`.

This implementation was developed around CTC-compatible NeMo actors and small experimental batches. It is not a general PPO implementation: confirm log-probability normalization, blank IDs, sequence lengths, and memory use when changing model families. See the [top-level RLNF README](../README.md) for project status and commands.
