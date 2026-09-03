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
- Transducer
- TDT
- FastConformer
- Conformer
- pytorch
- Bambara
- NeMo
- RNNT
license: cc-by-4.0
base_model: nvidia/parakeet-tdt_ctc-110m
model-index:
- name: soloni-114m-tdt-ctc-v0
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
      value: 36.588667569135566
    - name: Test CER
      type: cer
      value: 21.41897629892689
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
      value: 40.75
    - name: Test CER
      type: cer
      value: 24.711
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# Soloni TDT-CTC 114M Series

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--TDT-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-114M-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm-orange#model-badge)](#datasets)

`soloni-114m-tdt-ctc-v0` is a fine tuned version of nvidia's [`parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) that transcribes bambara language speech. Unlike its base model, this model cannot write Punctuations and Capitalizations since these were absent from its training.
The model was fine-tuned using **NVIDIA NeMo** and supports **both TDT (Token-and-Duration Transducer) and CTC (Connectionist Temporal Classification) decoding**.

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
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="RobotsMali/soloni-114m-tdt-ctc-v0")
```

### Transcribe Audio
```python
# Assuming you have a test audio file named sample_audio.wav
asr_model.transcribe(['sample_audio.wav'])
```

Note that the decoding strategy for the TDT decoder use CUDA Graphs by default but not all GPUs and versions of cuda support that parameter. If you run into a `RuntimeError: CUDA error: invalid argument` you should set that argument to false in the decoding strategy before calling asr_model.transcribe()

```python
decoding_cfg = asr_model.cfg.decoding
# Disable CUDA Graphs
decoding_cfg.greedy.use_cuda_graph_decoder = False
# Then change the decoding strategy
asr_model.change_decoding_strategy(decoding_cfg=decoding_cfg)
```
### Input

This model accepts **16 kHz mono-channel** audio (wav files) as input. But it is equipped with its own preprocessor doing the resampling so you may input audios at higher sampling rates.

### Output

This model provides transcribed speech as an hypothesis object with a text attribute containing the transcription string for a given speech sample.

## Model Architecture

This model uses a Hybrid FastConformer-TDT-CTC architecture. FastConformer is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. It possesses two independent but jointly trained decoders, one auto-regressive TDT decoder and a convolutional decoder with CTC loss. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

## Training

The NeMo toolkit was used for finetuning this model for **16,296 steps** over `parakeet-tdt_ctc-110m` model.The finetuning codes and configurations can be found at [RobotsMali-AI/bambara-asr](https://github.com/RobotsMali-AI/bambara-asr/).

The tokenizer for this model was trained on the text transcripts of the train set of RobotsMali/bam-asr-early using this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

## Dataset
This model was fine-tuned on the [bam-asr-early](https://huggingface.co/datasets/RobotsMali/bam-asr-early) dataset, which consists of 37 hours of transcribed Bambara speech data. The dataset is primarily derived from **Jeli-ASR dataset** (~87%).

## Performance

The performance of Automatic Speech Recognition models is commonly measured using Word Error Rate (WER) and Character Error Rate (CER). Since this model has two decoders operating independently at inference time, each decoder is evaluated independently too.

The following table shows these two metrics for each decoder:

| Benchmark | Decoding | WER (%) &darr; | CER (%) &darr; |
|---------------|----------|-----------------|-----------------|
| Bam ASR Early | CTC | 40.56 | 22.01 |
| Nyana Eval    | CTC | 40.75 | 24.70 |
| Bam ASR Early | TDT | 36.58  | 21.41 |
| Nyana Eval    | TDT | 47.10 | 31.27 |

These are greedy WER numbers without external LM. By default the main decoder branch is the TDT branch, if you would like to switch to the CTC decoder simply run this block of code before calling the .transcribe method

```python
# Retrieve the CTC decoding config
ctc_decoding_cfg = asr_model.cfg.aux_ctc.decoding
# Then change the decoding strategy
asr_model.change_decoding_strategy(decoder_type='ctc', decoding_cfg=ctc_decoding_cfg)
# Transcribe with the CTC decoder
asr_model.transcribe(['sample_audio.wav'])
```

## License
This model is released under the **CC-BY-4.0** license. By using this model, you agree to the terms of the license.

---

Feel free to open a discussion on Hugging Face or [file an issue](https://github.com/RobotsMali-AI/bambara-asr/issues) on GitHub for help or contributions.
