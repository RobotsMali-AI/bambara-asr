# LAU: Listen, Attend, Understand (Semantic Regularization for ST)

This directory contains the code and configurations for the **LAU (Listen, Attend, Understand)** framework, a semantic regularization technique designed to stabilize End-to-End Speech Translation (E2E-ST) in low-resource settings with high-variance labels.

## 🚀 Overview

The LAU framework introduces a directional auxiliary loss that grounds the acoustic encoder’s latent space into a high-resource semantic space (e.g., French BERT/RoBERTa). This repository provides:

* A custom NeMo model class: `HybridRNNTCTCLAUModel`.
* Semantic projection heads for the acoustic encoder.
* Support for two regularization losses (MSE and Cosine Similarity).
* Training configurations for low-resource Bambara-French translation with Nvidia NeMo.

## 🏗️ Architecture: `HybridRNNTCTCLAUModel`

We have extended the standard NVIDIA NeMo `EncDecHybridRNNTCTCBPEModel`. The custom class adds a semantic "anchor" branch to the encoder.

### Key Modifications:

* **Semantic Projection Head:** A linear/MLP layer that projects the encoder’s high-level features into the same dimensionality as the frozen text embeddings.
* **Auxiliary Loss Function:** Implements the regularization weight .
* **Multi-task Objective:** Model trained with LAU do not merely learn ST, they also incorporate semantics 


## 📦 Installation

This project requires **NVIDIA NeMo**. We recommend using the provided `requirements.txt` or a NeMo Docker container.

```bash
# Clone the parent repository
git clone https://github.com/diarray-hub/bambara-asr.git
cd bambara-asr/lau

# Install dependencies
pip install nemo_toolkit['all']

```

## ⚙️ Configuration

The `.yaml` files in the `/configs` directory are optimized for the **Jeli-ASR** dataset.

| Name              | Type                              | Params | Trainable |
|-------------------|-----------------------------------|--------|-----------|
| preprocessor      | AudioToMelSpectrogramPreprocessor | 0      | NO |
| encoder           | ConformerEncoder                  | 108 M  | YES |
| decoder           | RNNTDecoder                       | 3.6 M  | YES |
| joint             | RNNTJoint                         | 1.1 M  | YES |
| loss              | RNNTLoss                          | 0      | NO |
| spec_augmentation | SpectrogramAugmentation           | 0      | NO |
| wer               | WER                               | 0      | NO |
| ctc_decoder       | ConvASRDecoder                    | 263 K  | YES |
| ctc_loss          | CTCLoss                           | 0      | NO |
| ctc_wer           | WER                               | 0      | NO |
| embedding_model   | SentenceTransformer               | 110 M  | NO |
| semantic_head     | Sequential                        | 656 K  | YES |
| semantic_loss_fn  | MSELoss                           | 0      | NO |

## 📊 Evaluation & Metrics

Beyond standard WER and BLEU, this folder includes scripts to calculate the metrics introduced in our paper:

1. **Total Parameter Drift:** Measures the structural reorganization of the encoder weights.
2. **Cluster Purity & NMI:** Evaluates the semantic coherence of the latent space using the `lau-eval` dataset labels.

## 📜 Citation

If you use this code or our model checkpoints, please cite this paper (coming soon on arxiv):

```bibtex
@misc{diarra2026listenattendunderstandregularization,
      title={Listen, Attend, Understand: a Regularization Technique for Stable E2E Speech Translation Training on High Variance labels}, 
      author={Yacouba Diarra and Michael Leventhal},
      year={2026},
      eprint={2601.01121},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.01121}, 
}

```
