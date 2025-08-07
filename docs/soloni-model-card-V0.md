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
- Transducer
- TDT
- FastConformer
- Conformer
- pytorch
- Bambara
- NeMo
license: cc-by-4.0
base_model: nvidia/parakeet-tdt_ctc-110m
model-index:
- name: soloni-114m-tdt-ctc
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
    - name: Test WER (TDT)
      type: wer
      value: 66.7
    - name: Test WER (CTC)
      type: wer
      value: 40.6

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

`soloni-114m-tdt-ctc` is a fine tuned version of nvidia's [`parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) that transcribes bambara language speech. Unlike its base model, this model cannot write Punctuations and Capitalizations since these were absent from its training. 
The model was fine-tuned using **NVIDIA NeMo** and supports **both TDT (Token-and-Duration Transducer) and CTC (Connectionist Temporal Classification) decoding**.

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
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="RobotsMali/soloni-114m-tdt-ctc")
```

### Transcribe Audio
```python
# Assuming you have a test audio file named sample_audio.wav
asr_model.transcribe(['sample_audio.wav'])
```

### Input

This model accepts **16000 Hz mono-channel** audio (wav files) as input.

### Output

This model provides transcribed speech as a string for a given audio sample.

## Model Architecture

This model uses a Hybrid FastConformer-TDT-CTC architecture. FastConformer is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

## Training

The NeMo toolkit was used for finetuning this model for **16,296 steps** over `parakeet-tdt_ctc-110m` model. This model is trained with this [base config](https://github.com/diarray-hub/bambara-asr/blob/main/configs/parakeet-110m-config-v6.yaml). The full training configurations, scripts, and experimental logs are available here:
🔗 [Bambara-ASR Experiments](https://github.com/diarray-hub/bambara-asr)

The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

## Dataset
This model was fine-tuned on the [bam-asr-all](https://huggingface.co/datasets/RobotsMali/bam-asr-all) dataset, which consists of 37 hours of transcribed Bambara speech data. The dataset is primarily derived from **Jeli-ASR dataset** (~87%).

## Performance

The performance of Automatic Speech Recognition models is measured using Word Error Rate. Since this model has two decoders operating independently, each decoder is evaluated independently too.

The following table summarizes the performance of the available models in this collection with the Transducer decoder. Performances of the ASR models are reported in terms of **Word Error Rate (WER%)**. 

|**Decoder (Version)**|**Tokenizer**|**Vocabulary Size**|**bam-asr-all (test set)**|
|---------|-----------------------|-----------------|---------|
| CTC (V6) | BPE | 1024            | 40.6         |
|---------|-----------------------|-----------------|---------|
| TDT (V6) | BPE | 1024            | 66.7         |

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

More details are available in the **Experimental Technical Report**:
📄 [Draft Technical Report - Weights & Biases](https://wandb.ai/yacoudiarra-wl/bam-asr-nemo-training/reports/Draft-Technical-Report-V1--VmlldzoxMTIyOTMzOA).

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/diarray-hub/bambara-asr/issues) on github if you have any contributions 

---
