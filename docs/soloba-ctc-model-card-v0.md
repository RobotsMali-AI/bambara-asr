---
language:
- bm
library_name: nemo
datasets:
- RobotsMali/kunkado
- RobotsMali/bam-asr-early

thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- Transducer
- TDT
- FastConformer
- Conformer
- pytorch
- Bambara
- NeMo
license: cc-by-4.0
base_model: nvidia/parakeet-ctc-0.6b
model-index:
- name: soloba-ctc-0.6b-v0
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: bam-asr-early
      type: RobotsMali/bam-asr-early
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 35.15760898590088

metrics:
- wer
pipeline_tag: automatic-speech-recognition
---

# Soloni TDT-CTC 114M Bambara

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--CTC-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-0.6B-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm-orange#model-badge)](#datasets)

`soloba-ctc-0.6b-v0` is a fine tuned version of [`nvidia/parakeet-ctc-0.6b`](https://huggingface.co/nvidia/parakeet-ctc-0.6b) on RobotsMali/kunkado and RobotsMali/bam-asr-early. This model cannot does produce Capitalizations but not Punctuations. The model was fine-tuned using **NVIDIA NeMo**.

The model doesn't tag code swicthed expressions in its transcription since for training this model we decided to treat them as a modern variant of the Bambara Language removing all tags and markages.

## **🚨 Important Note**  
This model, along with its associated resources, is part of an **ongoing research effort**, improvements and refinements are expected in future versions. A human evaluation report of the model is coming soon. Users should be aware that:  

- **The model may not generalize very well accross all speaking conditions and dialects.**  
- **Community feedback is welcome, and contributions are encouraged to refine the model further.** 

## NVIDIA NeMo: Training

To fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after you've installed latest PyTorch version.

```bash
pip install nemo_toolkit['asr']
``` 

## How to Use This Model

Note that this model has been released for research purposes primarily.

### Load Model with NeMo
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="RobotsMali/soloba-ctc-0.6b-v0")
```

### Transcribe Audio
```python
model.eval()
# Assuming you have a test audio file named sample_audio.wav
asr_model.transcribe(['sample_audio.wav'])
```

### Input

This model accepts any **mono-channel audio (wav files)** as input and resamples them to *16 kHz sample rate* before performing the forward pass

### Output

This model provides transcribed speech as a string for a given speech sample and return an Hypothesis object (under nemo>=2.3)

## Model Architecture

This model uses a FastConformer Ecoder and a CTC decoder. FastConformer is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer). And a Convolutional Neural Net with CTC loss, the ***Connectionist Temporal Classification*** decoder

## Training

The NeMo toolkit (version 2.3.0) was used for finetuning this model for **183,086 steps** over `nvidia/parakeet-ctc-0.6b` model. This version is trained with this [base config](https://github.com/diarray-hub/bambara-asr/blob/main/kunkado-training/config/soloba/soloba-ctc-v0.0.0.yaml). The full training configurations, scripts, and experimental logs are available here:

🔗 [Bambara-ASR Experiments](https://github.com/diarray-hub/bambara-asr)

The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

## Dataset
This model was fine-tuned on the [kunkado](https://huggingface.co/datasets/RobotsMali/kunkado) dataset, the semi-labelled subset, which consists of **~120 hours of automatically annotated Bambara speech data**, and the [bam-asr-early](https://huggingface.co/datasets/RobotsMali/bam-asr-early) dataset.

## Performance

We report the Word Error Rate on the test set of bam-asr-early.

|**Decoder (Version)**|**Tokenizer**|**Vocabulary Size**|**bam-asr-all**|
|---------|-----------------------|-----------------|---------|
| v0 | BPE | 512            | 35.16       |


## License
This model is released under the **CC-BY-4.0** license. By using this model, you agree to the terms of the license.

---

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/diarray-hub/bambara-asr/issues) on github if you have any contributions 

---

