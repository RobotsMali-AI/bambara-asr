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
- TDT
- FastConformer
- Conformer
- pytorch
- Bambara
- NeMo
license: cc-by-4.0
base_model: RobotsMali/soloni-114m-tdt-ctc-V0
model-index:
- name: soloni-114m-tdt-ctc-v1
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
    - name: Test WER (TDT)
      type: wer
      value: 42.862898111343384
    - name: Test WER (CTC)
      type: wer
      value: 39.15117383003235

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

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--TDT-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-114M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm-lightgrey#model-badge)](#datasets)

`soloni-114m-tdt-ctc-v1` is a fine tuned version of nvidia's [`RobotsMali/soloni-114m-tdt-ctc-V0`](https://huggingface.co/RobotsMali/soloni-114m-tdt-ctc-V0) on [RobotsMali/kunkado](https://huggingface.co/datasets/RobotsMali/kunkado). This model cannot write Punctuations and Capitalizations since these were absent from its training. The model was fine-tuned using **NVIDIA NeMo** and supports **both TDT (Token-and-Duration Transducer) and CTC (Connectionist Temporal Classification) decoding**.

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
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="RobotsMali/soloni-114m-tdt-ctc-v1")
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

This model uses a Hybrid FastConformer-TDT-CTC architecture. FastConformer is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

It also features two independent decoders:

- A reccurent neural net Transducer that jointly preditcs tokens and their durations, the so called [Token-and-Duration][https://arxiv.org/abs/2304.06795] Transcuder by Nvidia
- A classical Convolutional Neural Net with CTC loss, the ***Connectionist Temporal Classification*** decoder

## Training

The NeMo toolkit (version 2.3.0) was used for finetuning this model for **100,551 steps** over `RobotsMali/soloni-114m-tdt-ctc-V0` model. This version is trained with this [base config](https://github.com/diarray-hub/bambara-asr/blob/main/kunkado-training/config/soloni/soloni-v1.4.0.yaml). The full training configurations, scripts, and experimental logs are available here:

🔗 [Bambara-ASR Experiments](https://github.com/diarray-hub/bambara-asr)

The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

## Dataset
This model was fine-tuned on the [kunkado](https://huggingface.co/datasets/RobotsMali/kunkado) dataset, the human-reviewed subset, which consists of **~40 hours of transcribed Bambara speech data**. The text was normalized with the [bambara-normalizer](https://pypi.org/project/bambara-normalizer/) prior to training, normalizing numbers, removing punctuations, removings tags and converting to lower case.

## Performance

The performance of Automatic Speech Recognition models is measured using Word Error Rate. Since this model has two decoders operating independently, each decoder is evaluated independently too.

The following table summarizes the performance of the available models in this collection with the Transducer decoder. Performances of the ASR models are reported in terms of **Word Error Rate (WER%)**. 

|**Decoder (Version)**|**Tokenizer**|**Vocabulary Size**|**bam-asr-all**|**kunkado**|
|---------|-----------------------|-----------------|---------|---------|
| CTC (v0) | BPE | 1024            | 40.6         |          -          |
| TDT (v0) | BPE | 1024            | 66.7         |          -          |
| CTC (v1) | BPE | 512            | -         |          39.15          |
| TDT (v1) | BPE | 512            | -       |          42.86            |

These are greedy WER numbers without external LM. By default the main decoder branch is the TDT branch, if you would like to switch to the CTC decoder simply run this block of code before calling the .transcribe method

```python
# Retrieve the CTC decoding config
ctc_decoding_cfg = model.cfg.aux_ctc.decoding
# Then change the decoding strategy
asr_model.change_decoding_strategy(decoder_type='ctc', decoding_cfg=ctc_decoding_cfg)
# Transcribe with the CTC decoder
asr_model.transcribe(['sample_audio.wav'])
```

## License
This model is released under the **CC-BY-4.0** license. By using this model, you agree to the terms of the license.

---

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/diarray-hub/bambara-asr/issues) on github if you have any contributions 

---
