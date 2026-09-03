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
- Semantic-Regularization
- LAU
license: cc-by-4.0
base_model: RobotsMali/soloni-114m-tdt-ctc-v0
model-index:
- name: lau-soloni-114m-mse-k1
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
      value: 14.29
    - name: Test WER
      type: wer
      value: 76.08
    - name: Test CER
      type: cer
      value: 58.64
metrics:
- bleu
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# LAU-Soloni 114M (MSE Semantic Anchor, λ=1)

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-HybridRNNTCTCLAU-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-114M-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm2fr-orange#model-badge)](#datasets)

`lau-soloni-114m-mse-k1` is an end-to-end **Speech Translation (ST)** model that incorporates **Listen, Attend, Understand (LAU)** semantic regularization. It translates **Bambara audio directly into French text**. Unlike standard ST models, it uses a semantic anchor during training to stabilize the acoustic encoder against high-variance "amateur" labels.

## **🚨 Important Note**
This model is a research artifact focused on semantic stability in low-resource and high variance settings. As noted in the associated research, it was trained on "amateur" translations which exhibit high variance. Users should expect:
- **High performance on semantic intent** but potential orthographic mistakes in the French output.
- **Better performance using the CTC decoding branch** for this specific checkpoint.

## NVIDIA NeMo: Custom Model Class

To use this model, you must use the custom `HybridRNNTCTCLAUModel` class, which overrides the standard NeMo `EncDecHybridRNNTCTCBPEModel` to support the semantic loss and head integration.

The full implementation of this class, along with training and evaluation scripts, is available in our [**GitHub repository**](https://github.com/RobotsMali-AI/bambara-asr/tree/main/lau)


```bash
pip install nemo-toolkit['asr']
# Ensure you have the custom LAU model class from our repository in your python path.

```

## NeMo 2.5.0 Compatibility

This checkpoint was created with NeMo 2.5.0. Loading it with newer NeMo versions (observed with 2.7.x) can fail because the strict decoding schema expects `key_phrase_items_list`; see [NVIDIA-NeMo/Speech#15658](https://github.com/NVIDIA-NeMo/Speech/issues/15658). This workaround was tested with Python 3.12:

```python
from pathlib import Path

from nemo.collections.asr.models import ASRModel
from lau.hybrid_rnnt_ctc_lau_models import HybridRNNTCTCLAUModel
from omegaconf import OmegaConf

model_name = "RobotsMali/lau-soloni-114m-mse-k1"
cfg = ASRModel.from_pretrained(model_name, return_config=True)
OmegaConf.set_struct(cfg, False)

for decoder in ("greedy", "beam"):
    boosting_tree = OmegaConf.select(cfg, f"decoding.{decoder}.boosting_tree")
    if boosting_tree is not None:
        boosting_tree.key_phrase_items_list = None

config_path = Path("patched_config.yaml").resolve()
OmegaConf.save(cfg, config_path)
st_model = HybridRNNTCTCLAUModel.from_pretrained(
    model_name=model_name,
    override_config_path=str(config_path),
    strict=False,
)
config_path.unlink()
```

## How to Use This Model

### Load Model

```python
from lau.hybrid_rnnt_ctc_lau_models import HybridRNNTCTCLAUModel
# Loading the custom LAU-regularized model
st_model = HybridRNNTCTCLAUModel.from_pretrained(model_name="RobotsMali/lau-soloni-114m-mse-k1")

```



### Translate Audio (CTC Recommended)

```python
# Switch to CTC or TDT decoding
ctc_decoding_cfg = st_model.cfg.aux_ctc.decoding
st_model.change_decoding_strategy(decoder_type='ctc', decoding_cfg=ctc_decoding_cfg)

# Translate
st_model.transcribe(['bambara_sample.wav'])

```

## Model Architecture

This model features a **FastConformer** encoder. A projection head is attached to the encoder's output, this head is used only during training to regularize using a **Mean Squared Error (MSE)** loss against a frozen high-resource semantic text embedding. This "anchors" the acoustic features to a known linguistic space.

## Training

The training followed the LAU framework:

1. **Pre-training:** Initialized from [`soloni-114m-tdt-ctc-v0`](https://huggingface.co/RobotsMali/soloni-114m-tdt-ctc-v0).
2. **Semantic Regularization:** Fine-tuned on **Jeli-ASR** (30h) using a dual-objective: standard translation loss and a semantic auxiliary loss with MSE.
3. **Hyperparameters:** AdamW optimizer, Noam scheduler, 1,000-step warmup, and a peak LR of 0.001.

## Dataset

The model was trained on **Jeli-ASR**, a corpus of ~30 hours of Bambara speech. The translations are "semi-professional," with a significant portion provided by native speakers without formal linguistic training, creating the high-variance environment that LAU is designed to handle.

## Evaluation

Performance is measured on the Jeli-ASR test set using Word Error Rate (WER), Character Error Rate (CER), and BLEU.

| Benchmark | Decoding | WER (%) &darr; | CER (%) &darr; | BLEU &uarr; |
| --- | --- | --- | --- | --- |
| Jeli-ASR Test | CTC | **76.08** | **58.64** | **14.29** |
| Jeli-ASR Test | TDT | 85.27 | 69.60 | 7.45 |


## License

This model is released under the **CC-BY-4.0** license.
