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
- Transducer
- FastConformer
- Conformer
- pytorch
- Bambara
- NeMo
license: cc-by-4.0
base_model: RobotsMali/soloba-ctc-0.6b-v0
model-index:
- name: soloba-ctc-0.6b-v2
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
      value: 30.85416567085703
    - name: Test CER
      type: cer
      value: 14.448940587985465
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
      value: 40.0
    - name: Test CER
      type: cer
      value: 22.339

metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# Soloba-CTC-600M Series

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--CTC-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-0.6B-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm-orange#model-badge)](#datasets)

`soloba-ctc-0.6b-v2` is a fine tuned version of [`RobotsMali/soloba-ctc-0.6b-v0`](https://huggingface.co/RobotsMali/soloba-ctc-0.6b-v0) on the African Next Voices dataset (ANV). This model does not consistently produce Capitalizations and Punctuations and it cannot produce acoustic event tags like those found in the ANV dataset in its transcriptions. It was fine-tuned using **NVIDIA NeMo**.

## **🚨 Important Note**
This model, along with its associated resources, is part of an **ongoing research effort**, improvements and refinements are expected in future versions. Users should be aware that:

- **The model may not generalize very well across all speaking conditions and dialects.**
- **Community feedback is welcome, and contributions are encouraged to refine the model further.**

## NVIDIA NeMo: Training

To fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after you've installed latest PyTorch version.

```bash
pip install nemo-toolkit['asr']
```

## How to Use This Model

Note that this model has been released for research purposes primarily.

### Load Model with NeMo
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="RobotsMali/soloba-ctc-0.6b-v2")
```

### NeMo 2.5.0 Compatibility

This checkpoint was created with NeMo 2.5.0. Loading it with newer NeMo versions (observed with 2.7.x) can fail because the strict decoding schema expects `key_phrase_items_list`; see [NVIDIA-NeMo/Speech#15658](https://github.com/NVIDIA-NeMo/Speech/issues/15658). This workaround was tested with Python 3.12:

```python
from pathlib import Path

from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf

model_name = "RobotsMali/soloba-ctc-0.6b-v2"
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
asr_model.eval()
# Assuming you have a test audio file named sample_audio.wav
asr_model.transcribe(['sample_audio.wav'])
```

### Input

This model accepts any **mono-channel audio (wav files)** as input and resamples them to *16 kHz sample rate* before performing the forward pass

### Output

This model provides transcribed speech as an hypothesis object with a text attribute containing the transcription string for a given speech sample. (nemo>=2.3)

## Model Architecture

This model uses a FastConformer encoder and a Convolutional decoder with CTC Loss. FastConformer is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

## Training

The NeMo toolkit was used for finetuning this model for **165,247 steps** over `RobotsMali/soloba-ctc-0.6b-v0` model. The finetuning codes and configurations can be found at [RobotsMali-AI/bambara-asr](https://github.com/RobotsMali-AI/bambara-asr/).

The tokenizer for this model was trained on the text transcripts of the train set of RobotsMali/afvoices using this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

## Dataset
This model was fine-tuned on a 100 hours pre-completion subset of the [African Next Voices](https://huggingface.co/datasets/RobotsMali/afvoices) dataset. You can reconstitute that subset with these [manifest files](https://github.com/RobotsMali-AI/bambara-asr/afvoices/pre-manifests).

## Performance

We report the Word Error Rate (WER) and Character Error Rate (CER) for this model:

| Benchmark | Decoding | WER (%) &darr; | CER (%) &darr; |
|---------------|----------|-----------------|-----------------|
| African Next Voices (afvoices) | CTC | 30.85 | 14.44 |
| Nyana Eval    | CTC | 40.01 | 22.34 |

## License
This model is released under the **CC-BY-4.0** license. By using this model, you agree to the terms of the license.

---

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/RobotsMali-AI/bambara-asr/issues) on GitHub for help or contributions.
