# bambara-asr

**Project Overview**

This repository contains a collection of tools, experiments, and utilities for building and fine-tuning automatic speech recognition (ASR) systems—particularly for low-resource languages like Bambara—and for experimenting with reinforcement learning from human feedback (RLHF) techniques in ASR training.

**License**

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Getting Started

1. **Install requirements** (for RLNF toolkit & core dependencies):

Before you try to experiment with Reinforcement Learning from Nouhoum Feedback

   ```bash
   git clone https://github.com/diarray-hub/bambara-asr.git --branch=rlnf-v2-gpu
  cd bambara-asr
   pip install .
   ```

OR 
```bash
pip install git+https://github.com/diarray-hub/bambara-asr.git@rlnf-v2-gpu
```

## How to use this package:
**want to train a reward model : [train_reward_model.py]()**

**want to test the reward model**

```python
import torch
from RLNF.Rewards.reward_config import RewardConfig
from RLNF.Rewards.reward_model import RewardModel
from RLNF.Rewards.reward_processor import RewardModelProcessor

audios = ["1.wav", "2.wav"]
texts = ["kelen", "fila."]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor : RewardModelProcessor = RewardModelProcessor.from_pretrained("RobotsMali/reward-model")
model : RewardModel = RewardModel.from_pretrained("RobotsMali/reward-model")

model.eval()
model.to(device)
    
out = processor(audios=audios, texts=texts)    
out = {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}
#out = dict(list(out.items())[:-2])

with torch.no_grad() :
  preds = model(**out).logits
    
    
for i, (t, val) in enumerate(zip(texts, preds)):
  print(f"Audio : {audios[i]:<10} | Text: {t:<10} | Score: {val.item() * 100:.4f}")

```
**want to train a RLNF model : [train_rlnf_model.py]()**

coming soon......


**want to test the RLNF model**

coming soon......



