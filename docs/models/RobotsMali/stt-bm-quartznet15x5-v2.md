---
language:
- bm
library_name: nemo
datasets:
- RobotsMali/afvoices

thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- CTC
- QuartzNet
- pytorch
- Bambara
- NeMo
license: cc-by-4.0
base_model: RobotsMali/stt-bm-quartznet15x5-v0
model-index:
- name: stt-bm-quartznet15x5-v2
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: African Next Voices
      type: RobotsMali/afvoices
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 42.57205678504852
    - name: Test CER
      type: cer
      value: 18.708413949318106
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Nyana Eval
      type: RobotsMali/nyana-eval
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 48.97
    - name: Test CER
      type: cer
      value: 24.22

metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# QuartzNet 15x5 CTC Series

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-QuartzNet-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-18M-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm-orange#model-badge)](#datasets)

`stt-bm-quartznet15x5-v2` is a fine-tuned version of [`RobotsMali/stt-bm-quartznet15x5-v0`](https://huggingface.co/RobotsMali/stt-bm-quartznet15x5-v0). This model cannot write **Punctuations and Capitalizations**, it utilizes a character encoding scheme, and transcribes text in the standard character set that is provided in its training set.

The model was fine-tuned using **NVIDIA NeMo** and is trained with **CTC (Connectionist Temporal Classification) Loss**.

## **🚨 Important Note**
This model, along with its associated resources, is part of an **ongoing research effort**, improvements and refinements are expected in future versions. Users should be aware that:

- **The model may not generalize very well across all speaking conditions and dialects.**
- **Community feedback is welcome, and contributions are encouraged to refine the model further.**

## NVIDIA NeMo: Training

To fine-tune or use the model, install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend installing it after setting up the latest PyTorch version.

```bash
pip install nemo-toolkit['asr']
```

## How to Use This Model

### Load Model with NeMo
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="RobotsMali/stt-bm-quartznet15x5-v2")
```

### NeMo 2.5.0 Compatibility

This checkpoint was created with NeMo 2.5.0. Loading it with newer NeMo versions (observed with 2.7.x) can fail because the strict decoding schema expects `key_phrase_items_list`; see [NVIDIA-NeMo/Speech#15658](https://github.com/NVIDIA-NeMo/Speech/issues/15658). This workaround was tested with Python 3.12:

```python
from pathlib import Path

from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf

model_name = "RobotsMali/stt-bm-quartznet15x5-v2"
cfg = ASRModel.from_pretrained(model_name, return_config=True)
OmegaConf.set_struct(cfg, False)

for decoder in ("greedy", "beam"):
    boosting_tree = OmegaConf.select(cfg, f"decoding.{decoder}.boosting_tree")
    if boosting_tree is not None:
        boosting_tree.key_phrase_items_list = None

config_path = Path("patched_config.yaml").resolve()
OmegaConf.save(cfg, config_path)
asr_model = ASRModel.from_pretrained(
    model_name=model_name,
    override_config_path=str(config_path),
    strict=False,
)
config_path.unlink()
```

### Transcribe Audio
```python
# Assuming you have a test audio file named sample_audio.wav
asr_model.transcribe(['sample_audio.wav'])
```

### Input

This model accepts **16 kHz mono-channel audio (wav files)** as input. But it is equipped with its own preprocessor doing the resampling so you may input audios at higher sampling rates.

### Output

This model provides transcribed speech as an hypothesis object with a text attribute containing the transcription string for a given speech sample.

## Model Architecture

QuartzNet is a convolutional architecture, which consists of **1D time-channel separable convolutions** optimized for speech recognition. More information on QuartzNet can be found here: [QuartzNet Model](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#quartznet).

## Training

The NeMo toolkit was used to fine-tune this model for **62,976 steps** over the `RobotsMali/stt-bm-quartznet15x5-v0` model. The finetuning codes and configurations can be found at [RobotsMali-AI/bambara-asr](https://github.com/RobotsMali-AI/bambara-asr/).

## Dataset
This model was fine-tuned on a 100 hours pre-completion subset of the [African Next Voices](https://huggingface.co/datasets/RobotsMali/afvoices) dataset. You can reconstitute that subset with these [manifest files](https://github.com/RobotsMali-AI/bambara-asr/tree/main/afvoices/pre-manifests).

## Performance

The performance of Automatic Speech Recognition models is measured using Word Error Rate (WER%) and Character Error Rate (CER), two edit distance metrics .

| Benchmark | Decoding | WER (%) &darr; | CER (%) &darr; |
|---------------|----------|-----------------|-----------------|
| African Next Voices (afvoices) | CTC | 42.57  | 18.70 |
| Nyana Eval    | CTC | 48.97 | 24.22 |

These are **greedy WER numbers without external LM**.

## License
This model is released under the **CC-BY-4.0** license. By using this model, you agree to the terms of the license.

---

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/RobotsMali-AI/bambara-asr/issues) on GitHub for help or contributions.
