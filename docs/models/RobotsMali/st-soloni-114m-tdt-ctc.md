---
language:
- bm
- fr
library_name: nemo
datasets:
- RobotsMali/jeli-asr
thumbnail: null
tags:
- speech-translation
- audio
- FastConformer
- Conformer
- pytorch
- Bambara
- French
- NeMo
license: cc-by-4.0
base_model: RobotsMali/soloni-114m-tdt-ctc-v1
model-index:
- name: st-soloni-114m-tdt-ctc
  results:
  - task:
      name: Speech Translation
      type: speech-translation
    dataset:
      name: Jeli-ASR
      type: RobotsMali/jeli-asr
      split: test
      args:
        language: bm-fr
    metrics:
    - name: Test BLEU
      type: bleu
      value: 24.18
    - name: Test WER
      type: wer
      value: 70.43
    - name: Test CER
      type: cer
      value: 55.98
metrics:
- bleu
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# AST-Soloni 114M (End-to-End Speech Translation)

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--ST-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-114M-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm2fr-orange#model-badge)](#datasets)

`st-soloni-114m-tdt-ctc` is an end-to-end **Speech Translation (ST)** model designed to translate **Bambara audio directly into French text**. It is based on the FastConformer architecture and pretrained for ASR on jeli-asr and Kunkado ([soloni-v1](https://huggingface.co/RobotsMali/soloni-114m-tdt-ctc-v1)) before being fine-tuned for translation.

## **🚨 Important Note**
This model is a baseline for research on low-resource speech translation. It was trained on "amateur" translations which exhibit high variance.

## NVIDIA NeMo: Training

To use this model, ensure you have the NVIDIA NeMo toolkit installed:

```bash
pip install nemo-toolkit['asr']
```

## NeMo 2.5.0 Compatibility

This checkpoint was created with NeMo 2.5.0. Loading it with newer NeMo versions (observed with 2.7.x) can fail because the strict decoding schema expects `key_phrase_items_list`; see [NVIDIA-NeMo/Speech#15658](https://github.com/NVIDIA-NeMo/Speech/issues/15658). This workaround was tested with Python 3.12:

```python
from pathlib import Path

from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCBPEModel
from omegaconf import OmegaConf

model_name = "RobotsMali/st-soloni-114m-tdt-ctc"
cfg = ASRModel.from_pretrained(model_name, return_config=True)
OmegaConf.set_struct(cfg, False)

for decoder in ("greedy", "beam"):
    boosting_tree = OmegaConf.select(cfg, f"decoding.{decoder}.boosting_tree")
    if boosting_tree is not None:
        boosting_tree.key_phrase_items_list = None

config_path = Path("patched_config.yaml").resolve()
OmegaConf.save(cfg, config_path)
st_model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name=model_name,
    override_config_path=str(config_path),
    strict=False,
)
config_path.unlink()
```

## How to Use This Model
Load Model with NeMo

```python
import nemo.collections.asr as nemo_asr
# This model uses the Hybrid RNNT-CTC encoder-decoder structure adapted for ST
st_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="RobotsMali/st-soloni-114m-tdt-ctc")
```

Translate audio

```python
# Translates Bambara audio directly to French text
st_model.transcribe(['bambara_sample.wav'])
```

## Model Architecture

This model utilizes the FastConformer encoder, which features 8x depthwise-separable convolutional downsampling for efficiency. While originally an ASR architecture, this model is trained as an E2E-ST system where the decoder predicts French text tokens directly from Bambara speech features.

## Training

The model was trained following a two-stage process:

1. Pre-training: Initialized from RobotsMali/soloni-114m-tdt-ctc-v1

2. Finetuning: Trained on the Jeli-ASR dataset (30 hours) with the Audio-French pairs

3. Hyperparameters: Optimized using AdamW with a Noam scheduler, a peak learning rate of 0.001, and a 1,000-step warmup.

The finetuning codes and configurations can be found at [RobotsMali-AI/bambara-asr](https://github.com/RobotsMali-AI/bambara-asr/).

## Dataset

This model was trained and evaluated on Jeli-ASR, a corpus of ~30 hours of Bambara speech with French translations provided by native speakers. The translations are semi-professional with only 10h completed by trained linguists.

## Evaluation

Thus model was evaluated on the test set of Jeli-ASR. We report the Word Error Rate (WER), the Character Error Rate (CER) and the Bilingual Evaluation Understudy (BLEU).

| Benchmark | Decoding | WER (%) &darr; | CER (%) &darr; | BLEU &uarr; |
|---------------|----------|-----------------|-----------------|------------|
| Jeli-ASR Test | CTC | 73.90 | **55.98** | 17.28 |
| Jeli-ASR Test | TDT | **70.43** | 58.17 | **24.18** |

## License

This model is released under the CC-BY-4.0 license.

---

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/RobotsMali-AI/bambara-asr/issues) on GitHub for help or contributions.
