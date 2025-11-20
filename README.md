# bambara-asr

**Project Overview**

This repository contains a collection of tools, experiments, and utilities for building and fine-tuning automatic speech recognition (ASR) systems—particularly for low-resource languages like Bambara—and for experimenting with reinforcement learning from human feedback (RLHF) techniques for ASR training.

**License**

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

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

Each subfolder in this repo holds code and config for different experiments that we have done. Each one also possesses a dedicated README for further details and instructions. 

4. **Run an RLNF demo training session quickly**:
   Open `test_rlnf.ipynb`, follow the step‑by‑step example of a PPO‑based ASR fine‑tuning workflow.

   Note: You'll need a few audio samples organized by a manifest.jsonl file and a significant amount of RAM to test the pipeline on a CPU device at the moment. The code is not optimized.

---

## Contributions

Pretty much everything in this repo is still very experimental so contributions are really welcome! Please open an issue or submit a pull request for any bug fix or feature request. For questions or support, file an issue in this repo.

---

*README files for individual modules are provided in their respective folders. Start with **`rlnf/README.md`** for detailed instructions on the RLHF toolkit.*
