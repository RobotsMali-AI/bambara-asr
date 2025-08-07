---
language:
- bm
library_name: nemo
datasets:
- RobotsMali/bam-asr-all

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
base_model: stt_fr_quartznet15x5
model-index:
- name: stt-bm-quartznet15x5
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: bam-asr-all
      type: RobotsMali/bam-asr-all
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 46.5

metrics:
- wer
pipeline_tag: automatic-speech-recognition
---

# QuartzNet 15x5 CTC Bambara

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-QuartzNet-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-19M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm-lightgrey#model-badge)](#datasets)

`stt-bm-quartznet15x5` is a fine-tuned version of NVIDIA’s [`stt_fr_quartznet15x5`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_quartznet15x5) optimized for **Bambara ASR**. This model cannot write **Punctuations and Capitalizations**, it utilizes a character encoding scheme, and transcribes text in the standard character set that is provided in the training set of bam-asr-all dataset.

The model was fine-tuned using **NVIDIA NeMo** and is trained with **CTC (Connectionist Temporal Classification) Loss**.

## NVIDIA NeMo: Training

To fine-tune or use the model, install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend installing it after setting up the latest PyTorch version.

```bash
pip install nemo_toolkit['asr']
```

## How to Use This Model

### Load Model with NeMo
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="RobotsMali/stt-bm-quartznet15x5")
```

### Transcribe Audio
```python
# Assuming you have a test audio file named sample_audio.wav
asr_model.transcribe(['sample_audio.wav'])
```

### Input

This model accepts **16 kHz mono-channel audio (wav files)** as input.

### Output

This model provides transcribed speech as a string for a given speech sample.

## Model Architecture

QuartzNet is a convolutional architecture, which consists of **1D time-channel separable convolutions** optimized for speech recognition. More information on QuartzNet can be found here: [QuartzNet Model](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#quartznet).

## Training

The NeMo toolkit was used to fine-tune this model for **25939 steps** over the `stt_fr_quartznet15x5` model. This model is trained with this [base config](https://github.com/diarray-hub/bambara-asr/blob/main/configs/quartznet-20m-config-v2.yaml). The full training configurations, scripts, and experimental logs are available here:

🔗 [Bambara-ASR Experiments](https://github.com/diarray-hub/bambara-asr)

## Dataset
This model was fine-tuned on the [bam-asr-all](https://huggingface.co/datasets/RobotsMali/bam-asr-all) dataset, which consists of **37 hours of transcribed Bambara speech data**. The dataset is primarily derived from **Jeli-ASR dataset** (~87%).

## Performance

The performance of Automatic Speech Recognition models is measured using **Word Error Rate (WER%)**.

|**Version**|**Tokenizer**|**Vocabulary Size**|**bam-asr-all (test set)**|
|---------|-----------------------|-----------------|---------|
| V2  | Character-wise | 45            | 46.5         |

These are **greedy WER numbers without external LM**.

## License
This model is released under the **CC-BY-4.0** license. By using this model, you agree to the terms of the license.

---

More details are available in the **Experimental Technical Report**:
📄 [Draft Technical Report - Weights & Biases](https://wandb.ai/yacoudiarra-wl/bam-asr-nemo-training/reports/Draft-Technical-Report-V1--VmlldzoxMTIyOTMzOA).

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/diarray-hub/bambara-asr/issues) on GitHub if you have any contributions.

---
