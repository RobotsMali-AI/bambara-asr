# bambara-asr

**Project Overview**

This repository contains a collection of tools, experiments, and utilities for building and fine-tuning automatic speech recognition (ASR) systems—particularly for low-resource languages like Bambara—and for experimenting with reinforcement learning from human feedback (RLHF) techniques in ASR training.

**License**

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Repository Structure

```
├── bam-tokenizer-spe-bpe-v1024/      # Pretrained SentencePiece tokenizer model & vocab; trained on bambara text
├── docs/                            # Technical reports, model cards, and documentation drafts
├── early-experiments/               # Initial ASR fine‑tuning scripts and configs
├── rlnf/                            # Core RLHF toolkit and training library
├── utils/                           # Helper scripts and utilities
├── test_rlnf.ipynb                  # Notebook demo for rlnf training workflow
├── LICENSE                          # Project license (MIT)
└── README.md                        
```
---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RobotsMali-AI/bambara-asr.git
   cd bambara-asr
   ```

2. **Install requirements** (for RLNF toolkit & core dependencies):

Before you try to experiment with Reinforcement Learning from Nouhoum Feedback

   ```bash
   pip install -r rlnf/requirements.txt
   pip install -e rlnf
   ```
   
   Note that the second line will install the package in edit mode, remove -e option to create a permanent install

3. **Explore sub‑folders**:

   * **bam-tokenizer-spe-bpe-v1024/**: Contains `tokenizer.model`, `tokenizer.vocab`, and `vocab.txt` for BPE sentencepiece tokenization.
   * **docs/**: Technical reports, model cards, and documentation drafts for datasets and ASR models that we released on HuggingFace.
   * **early-experiments/**: Prototype scripts and configuration files for initial ASR model fine‑tuning experiments using NeMo.
   * **rlnf/**: Core Python package implementing reinforcement learning from human feedback (RLHF) for ASR.
   * **utils/**: Collection of utility scripts and helper functions.

4. **Run an RLNF demo training session quickly**:
   Open `test_rlnf.ipynb`, follow the step‑by‑step example of a PPO‑based ASR fine‑tuning workflow.

   Note: You'll need a few audio samples organized by a manifest.jsonl file and a significant amount of RAM to test the pipeline on a CPU device at the moment. The code is not optimized.

---

## Contributions

Pretty much everything in this repo is still very experimental so contributions are really welcome! Please open an issue or submit a pull request for any bug fix or feature request. For questions or support, file an issue in this repo.

---

*README files for individual modules are provided in their respective folders. Start with **`rlnf/README.md`** for detailed instructions on the RLHF toolkit.*
