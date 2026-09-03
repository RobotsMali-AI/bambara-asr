---
language:
- bm
library_name: nemo
datasets:
- RobotsMali/bam-asr-early

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
- name: stt-bm-quartznet15x5-v0
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Bam ASR Early
      type: RobotsMali/bam-asr-early
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 46.66408818410365
    - name: Test CER
      type: cer
      value: 21.65830309580792
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
      value: 65.421
    - name: Test CER
      type: cer
      value: 30.662

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

`stt-bm-quartznet15x5-v0` is a fine-tuned version of NVIDIA’s [`stt_fr_quartznet15x5`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_quartznet15x5) trained for Automatic Speech Recognition of Bambara speech. This model cannot write **Punctuations and Capitalizations**, it utilizes a character encoding scheme, and transcribes text in the standard character set that is provided in the training set of bam-asr-early dataset.

The model was fine-tuned using **NVIDIA NeMo** and is trained with **CTC (Connectionist Temporal Classification) Loss**.

## **🚨 Important Note**  
This model, along with its associated resources, is part of an **ongoing research effort**, improvements and refinements are expected in future versions. Users should be aware that:  

- **The model may not generalize very well accross all speaking conditions and dialects.**  
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
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="RobotsMali/stt-bm-quartznet15x5-v0")
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

The NeMo toolkit was used to fine-tune this model for **25939 steps** over the `stt_fr_quartznet15x5` model. The finetuning codes and configurations can be found at [RobotsMali-AI/bambara-asr](https://github.com/RobotsMali-AI/bambara-asr/).

## Dataset
This model was fine-tuned on the [bam-asr-early](https://huggingface.co/datasets/RobotsMali/bam-asr-early) dataset, which consists of **37 hours of transcribed Bambara speech data**. The dataset is primarily derived from **Jeli-ASR dataset** (~87%).

## Performance

The performance of Automatic Speech Recognition models is measured using Word Error Rate (WER%) and Character Error Rate (CER), two edit distance metrics .

| Benchmark | Decoding | WER (%) &darr; | CER (%) &darr; |
|---------------|----------|-----------------|-----------------|
| Bam ASR Early | CTC | 46.66  | 21.65 |
| Nyana Eval    | CTC | 65.42 | 30.66 |

These are **greedy WER numbers without external LM**.

## License
This model is released under the **CC-BY-4.0** license. By using this model, you agree to the terms of the license.

---

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/RobotsMali-AI/bambara-asr/issues) on GitHub for help or contributions.
