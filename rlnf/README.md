# RLNF Toolkit

The `rlnf/` directory houses a Python package for a basic implementation of **Reinforcement Learning from Human Feedback (RLHF)** applied to Automatic Speech Recognition (ASR). Using Proximal Policy Optimization (PPO) and custom reward models, this toolkit streamlines fine‑tuning pre‑trained ASR models based on human quality ratings.

---

## Installation

From the repository root:

```bash
pip install -r rlnf/requirements.txt
pip install -e rlnf
```

This will install dependencies such as PyTorch, NVIDIA NeMo, and SentencePiece and register the `rlnf` package in your environment in edit mode.

---

## Package Structure

```
rlnf/
├── trainer.py            # RLNFTrainer: orchestrates PPO training loop and interactions between actor, critic, and reward model
├── dataloaders/
│   ├── audio_dataset.py  # AudioDataset: loads raw audio files and applies AudioToMelSpectrogramPreprocessor
│   └── reward_dataset.py # RewardDataset: wraps human feedback (score) with transcript and preprocessed audio into PyTorch Dataset
├── reward/
│   ├── reward_model.py   # RewardModel: defines neural architectures for scoring ASR outputs (A simple Regression model with a Siamese-like encoder combining text and audio)
│   ├── train_reward_model.py # Script to train a reward model on annotated data
│   └── train_utils.py    # Utility functions for training and evaluating a reward-model
├── ppo/
│   ├── optimizer.py      # PPOOptimizer: implements actor update with clipping, entropy bonus, and logging
│   ├── loss.py           # PPOLoss: computes surrogate loss and value function loss
│   ├── rollout.py        # rollout functions: Collect an on policy bacth of data for PPO
│   └── critic_network.py # CriticModel: architecture for baseline value estimation
├── requirements.txt      # pin dependencies for the RLNF package
└── setup.py              # `setuptools` installer for the `rlnf` package
```

An overview image `rlnf.png` is also included for documentation and presentations.

---

## Quickstart

Below is a minimal example of fine‑tuning a NeMo ASR model with RLHF:

```python
from rlnf.reward.reward_model import RewardModel
from rlnf.ppo.critic_network import CriticModel
from rlnf.trainer import RLNFTrainer

import torch
import nemo.collections.asr as nemo_asr
from sentencepiece import SentencePieceProcessor

# 1. Load pretrained ASR and reward model
asr_model: nemo_asr.models.EncDecCTCModel = nemo_asr.models.EncDecCTCModel.from_pretrained("RobotsMali/stt-bm-quartznet15x5-V0")
asr_model.eval()

reward_model = RewardModel.from_pretrained("path/to/reward-model.rw") # Path to a pretrained reward model checkpoint

# Load your sentencepiece tokenizer (for instance the bambara tokenizer in this repo)
tokenizer = load_tokenizer("bambara-asr/bam-tokenizer-spe-bpe-v1024/tokenizer.model")

# Initialize the critic model
critic_model = CriticModel(n_mel=preprocessor_config['features'])

# 2. Prepare datasets and Audio Preprocessor config

## Audio preprocessor Config
preprocessor_config = {
    'normalize': 'per_feature',
    'window_size': 0.02,
    'sample_rate': 16000,
    'window_stride': 0.01,
    'window': 'hann',
    'features': 64,
    'n_fft': 512,
    'frame_splicing': 1,
    'dither': 1e-05,
    'stft_conv': False
}

# Training manifest file path
training_manifest = "path/to/train-manifest.jsonl"
validation_manifest = "path/to/test-manifest.jsonl" # Expected to contain transcripts, not only audios

# Choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Initialize RLHF trainer
trainer = RLNFTrainer(
    reward_model=reward_model,
    critic_model=critic_model,
    asr_model=asr_model,
    train_manifest=training_manifest,
    val_manifest=validation_manifest,
    audio_preprocessor_config=preprocessor_config,
    batch_size=2,
    epochs=3,
    num_workers=0,
    pin_memory=False,
    sp_tokenizer=tokenizer,
    device=device,
    wandb_logging=False,
)

# 4. Start training
trainer.train()
```

---

***This is a minimal implementation only aiming at testing a few hypothesis, it is therefore far from optimized***
---

