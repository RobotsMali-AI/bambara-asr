# Reward Module

The `rlnf/reward/` folder contains all the code needed to train and evaluate neural reward models that predict human quality scores for ASR outputs. These models can later be used in an RLNF trainer or as proxy to human evaluation.

---

## Contents

```
rlnf/reward/
├── reward_model.py         # `RewardModel`: nn.Module defining architecture and forward pass
├── train_reward_model.py   # Script for training reward models from a reward Dataset manifest
├── train_reward_model.ipynb # Notebook example for reward-model training workflow
└── train_utils.py          # Utility functions for training and evaluating a reward model
```

---

## RewardModel

A PyTorch nn.Module that encapsulates:

* **Audio encoder**:  Simple Convolutional architecture that extracts higher level features from preprocessed audio
* **Text encoder**: Embedding lookup + RNN subnetwork (X LSTM layers) over token IDs
* **Concat**: Concatenate Audio Encoder's output and Text Encoder's output
* **Regression head**: Fully connected layers mapping concatenated embeddings to a scalar score

---

*See [train_reward_model.ipynb](train_reward_model.ipynb) for an example training notebook*