---
language:
- bm
library_name: nemo
datasets:
- RobotsMali/kunkado

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
base_model: RobotsMali/stt-bm-quartznet15x5-V0
model-index:
- name: stt-bm-quartznet15x5-v1
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: kunkado (human-reviewed)
      type: RobotsMali/kunkado
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 55.5

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

`stt-bm-quartznet15x5-v1` is a fine-tuned version of [`RobotsMali/stt-bm-quartznet15x5-V0`](https://huggingface.co/RobotsMali/stt-bm-quartznet15x5-V0) on [RobotsMali/kunkado](https://huggingface.co/datasets/RobotsMali/kunkado). This model cannot write **Punctuations and Capitalizations**, it utilizes a character encoding scheme, and transcribes text in the standard character set that is provided in its training dataset.

This is the smallest of a series of model that we are developing to be able to transcribe modern Bamako Bambara. The model doesn't tag code swicthed expressions in its transcription since for training this model we decided to treat them as a modern variant of the Bambara Language removing all tags and markages. The model was fine-tuned using **NVIDIA NeMo** and is trained with **CTC (Connectionist Temporal Classification) Loss**.

## **🚨 Important Note**  
This model, along with its associated resources, is part of an **ongoing research effort**, improvements and refinements are expected in future versions. A human evaluation report of the model is coming soon. Users should be aware that:  

- **The model may not generalize very well accross all speaking conditions and dialects.**  
- **Community feedback is welcome, and contributions are encouraged to refine the model further.** 

## NVIDIA NeMo: Training

To fine-tune or use the model, install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend installing it after setting up the latest PyTorch version.

```bash
pip install nemo_toolkit['asr']
```

## How to Use This Model

### Load Model with NeMo
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="RobotsMali/stt-bm-quartznet15x5-v1")
```

### Transcribe Audio
```python
asr_model.eval()
# Assuming you have a test audio file named sample_audio.wav
output = asr_model.transcribe(['sample_audio.wav'])
print(output.text)
```

### Input

This model accepts any **mono-channel audio (wav files)** as input and resamples them to *16 kHz sample rate* before performing the forward pass

### Output

This model provides transcribed speech as a string for a given speech sample and return an Hypothesis object (under nemo>=2.3)

## Model Architecture

QuartzNet is a convolutional architecture, which consists of **1D time-channel separable convolutions** optimized for speech recognition. More information on QuartzNet can be found here: [QuartzNet Model](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#quartznet).

## Training

The NeMo toolkit (version 2.3.0) was used to fine-tune this model for **64300 steps** over the `RobotsMali/stt-bm-quartznet15x5-V0` model. This model is trained with this [base config](https://github.com/RobotsMali-AI/bambara-asr/blob/main/kunkado-training/config/quartznet/quartznet-v1.3.0.yaml). The full training configurations, scripts, and experimental logs are available here:

🔗 [Bambara-ASR Experiments](https://github.com/RobotsMali-AI/bambara-asr)

## Dataset
This model was fine-tuned on the [kunkado](https://huggingface.co/datasets/RobotsMali/kunkado) dataset, the human-reviewed subset, which consists of **~40 hours of transcribed Bambara speech data**. The text was normalized with the [bambara-normalizer](https://pypi.org/project/bambara-normalizer/) prior to training, normalizing numbers, removing punctuations, removings tags and converting to lower case.

## Performance

The performance of Automatic Speech Recognition models is measured using **Word Error Rate (WER%)**.

|**Version**|**Tokenizer**|**Vocabulary Size**|**bam-asr-all**|**Kunkado**|
|---------|-----------------------|-----------------|---------|-----------|
| v0  | Character-wise | 45            | 46.5         |         -         |
| v1  | Character-wise | 46            | -         |         55.5         |

These are **greedy WER numbers without external LM** and no beam search decoding.

## License
This model is released under the **CC-BY-4.0** license. By using this model, you agree to the terms of the license.

---

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/RobotsMali-AI/bambara-asr/issues) on GitHub if you have any contributions.

---
