# PPO Module

The `rlnf/ppo/` directory contains all components to perform Proximal Policy Optimization (PPO) on ASR models using human feedback as reward signals. This module is used internally by `RLNFTrainer` but can also be integrated directly into custom training loops.

---

## Contents

```
rlnf/ppo/
├── optimizer.py      # PPOOptimizer: handles actor & critic updates with a combined clipped Loss + MSE Loss and entropy bonus
├── loss.py           # PPOLoss: computes clipped surrogate objective
├── rollout.py        # collect_batch: functions to generate episodes from ASR model
└── critic_network.py # CriticModel: neural architecture for baseline value estimation
```

---

## Components

### `PPOOptimizer` (in `optimizer.py`)

Implements the core PPO update logic:

* **Initialization args**:

  * `actor`: the ASR model (policy network)
  * `critic`: instance of `CriticModel`
  * `actor_lr`, `critic_lr`: learning rates
  * `clip_epsilon`: clipping range for policy updates
  * `entropy_coef`: scale for entropy bonus
  * `K_updates`: Number of update per batch

* **Key method**:

  * `updates()`: takes a rollout batch, compute advantages and perform Actor-Critic optimization K times.

### `PPOLoss` (in `loss.py`)

Defines the surrogate clipped loss function:

* **Inputs**:

  * `log_probs_old`, `log_probs_new`: tensors of action log-probabilities before and after update
  * `advantages`: advantage estimates for each timestep

* **Outputs**:

  * Clipped policy loss

### `rollout` (in `rollout.py`)

Generates trajectories by running the ASR model on audio inputs:

* **Functions**:

  * `collect_batch(actor, reward_model, dataset, device)`: iterates through a `AudioDataset`, infers actor and decodes transcripts, scores them with `reward_model`, and collects `(log_prob, value, reward)` tuples.

### `CriticModel` (in `critic_network.py`)

Similar to the reward model's architecture, without the text encoder:

* Inputs: audio features

---


## Configuration Tips

* **`clip_epsilon`**: 0.1–0.3 is typical; lower values for more conservative updates.
* **Batch size**: trade-off between gradient stability and compute.
* **Entropy bonus**: helps maintain exploration, especially early in training.
* **Value network capacity**: match the complexity of your reward signal.

---

