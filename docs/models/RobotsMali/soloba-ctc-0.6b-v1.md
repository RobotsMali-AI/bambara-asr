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
- Transducer
- FastConformer
- Conformer
- pytorch
- Bambara
- NeMo
license: cc-by-4.0
base_model: RobotsMali/soloba-ctc-0.6b-v0
model-index:
- name: soloba-ctc-0.6b-v1
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Kunkado
      type: RobotsMali/kunkado
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 42.80104712041885
    - name: Test CER
      type: cer
      value: 24.927870915653497
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
      value: 40.19
    - name: Test CER
      type: cer
      value: 20.94

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

`soloba-ctc-0.6b-v1` is a fine tuned version of [`RobotsMali/soloba-ctc-0.6b-v0`](https://huggingface.co/RobotsMali/soloba-ctc-0.6b-v0) on RobotsMali/kunkado. This model does not consistently produce Capitalizations and Punctuations and it cannot produce acoustic event tags like those found in Kunkado its transcriptions. It was fine-tuned using **NVIDIA NeMo**.

## **🚨 Important Note**
This model, along with its associated resources, is part of an **ongoing research effort**, improvements and refinements are expected in future versions. A human evaluation report of the model is coming soon. Users should be aware that:

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
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="RobotsMali/soloba-ctc-0.6b-v1")
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

The NeMo toolkit was used for finetuning this model for **162,445 steps** over `RobotsMali/soloba-ctc-0.6b-v0` model. The finetuning codes and configurations can be found at [RobotsMali-AI/bambara-asr](https://github.com/RobotsMali-AI/bambara-asr/).

The tokenizer for this model was trained on the text transcripts of the train set of RobotsMali/kunkado using this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

## Dataset
This model was fine-tuned on the [kunkado](https://huggingface.co/datasets/RobotsMali/kunkado) dataset, the human-reviewed subset, which consists of **~40 hours of transcribed Bambara speech data**. The text was normalized with the [bambara-normalizer](https://pypi.org/project/bambara-normalizer/) prior to training, normalizing numbers, removing punctuations and removing tags.


## Performance

We report the Word Error Rate (WER) and Character Error Rate (CER) for this model:

| Benchmark | Decoding | WER (%) &darr; | CER (%) &darr; |
|---------------|----------|-----------------|-----------------|
| Kunkado | CTC | 42.80 | 24.92 |
| Nyana Eval    | CTC | 40.19 | 20.94 |

## License
This model is released under the **CC-BY-4.0** license. By using this model, you agree to the terms of the license.

---

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/RobotsMali-AI/bambara-asr/issues) on GitHub for help or contributions.
