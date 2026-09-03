---
language:
- bm
library_name: nemo
datasets:
- RobotsMali/an-be-kalan-bench
thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- CTC
- QuartzNet
- legacy-model
- deprecated
- pytorch
- Bambara
- NeMo
license: cc-by-4.0
base_model: RobotsMali/stt-bm-quartznet15x5-v2
model-index:
- name: anbekalanNet
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: An be kalan Children's Reading Benchmark
      type: RobotsMali/an-be-kalan-bench
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 40.0
    - name: Test CER
      type: cer
      value: 15.0
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# anbekalanNet (QuartzNet 15x5 char CTC Series) — [LEGACY]

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-QuartzNet-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-18M-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm-orange#model-badge)](#datasets)

`anbekalanNet` is the final domain-specific release of the convolutional QuartzNet framework adapted for Bambara children's reading materials. It is a fine-tuned version of [`RobotsMali/stt-bm-quartznet15x5-v2`](https://huggingface.co/RobotsMali/stt-bm-quartznet15x5-v2). Like its predecessors, the model was fine-tuned using **NVIDIA NeMo** and trained with **CTC (Connectionist Temporal Classification) Loss**.

## **🚨 Obsolescence Notice**

This architecture is officially retired. Field testing and benchmark evaluations demonstrate that this convolutional foundation exhibits unstable alignment paths under tight, low-resource constraints compared to hybrid attention-transducer systems.


## NVIDIA NeMo: Installation

To load or run evaluations on this legacy checkpoint, install the standard [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) package:

```bash
pip install nemo-toolkit['asr']

```

## How to Use This Model

### Load Model with NeMo

```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="RobotsMali/anbekalanNet")

```

### Transcribe Audio

```python
# Downsamples or processes input natively via its internal preprocessor
asr_model.transcribe(['sample_audio.wav'])

```

### Input / Output

* **Input:** Accepts **16 kHz mono-channel audio (wav files)**.
* **Output:** Generates a transcribed speech hypothesis object with a lowercase `.text` string attribute containing character-encoded text. It does not output punctuations or capitalizations.

## Model Architecture

QuartzNet is a convolutional ASR model consisting of **1D time-channel separable convolutions** designed to minimize parameter count while maintaining acoustic representations. This specific variant utilizes a **15x5 block structure** with roughly 18 million parameters.

## Training & Fine-Tuning Configurations

Four experimental setups were designed to test vocabulary limits and regularization effects. This final artifact (`anbekalanNet`) used the following strict parameters:

* 
**Optimization Window:** Regulated with an **Early Stopping mechanism** set to a **15-epoch patience window** monitored against validation metrics.

* 
**Convergence Behavior:** Due to high training-batch lexical convergence (<4% WER), validation metrics flatlined early. Operational shutdown was forced at **epoch 30** to protect the encoder from total generalization collapse.

## Dataset

The model was fine-tuned on the combined **Main + Duplicate** expanded subsets (**45.6 hours** total) of the [RobotsMali/an-be-kalan-bench](https://huggingface.co/datasets/RobotsMali/an-be-kalan-bench) educational children's book corpus.

* 
**Main Split (1.6h):** Pristine recordings of unique readings across 22 GAIFE books by 8 distinct speakers.


* 
**Duplicate Split (44h):** High-density, redundant multi-speaker tracks reading identical textual literature to introduce physical vocal variance (pitch, child vocal acoustics, and regional accents).



## Performance

The performance metrics below illustrate how expanding data volume rescued the QuartzNet framework from catastrophic lexical overfitting.

### Overall Evaluation Metrics

| Experimental Pass | Dataset Baseline Configuration | SpecAugment | Training Volume | Test WER (%) ↓ | Test CER (%) ↓ |
| --- | --- | --- | --- | --- | --- |
| **anbekalanNet-exp3 (this release)** | <br>**Main + Duplicate** | <br>**None** | <br>**45.6 Hours** | <br>**40.0%** | <br>**15.0%** |
| *anbekalanNet-exp1* | <br>*Main Only* | <br>*None* | <br>*1.6 Hours* | <br>*93.0%*  | <br>*80.0%* |
| *anbekalanNet-exp2* | <br>*Main Only* | <br>*Active* | <br>*1.6 Hours* | <br>*64.0%* | <br>*23.0%* |
| *anbekalanNet-exp4* | <br>*Main + Duplicate* | <br>*Active* | <br>*45.6 Hours* | <br>*42.0%* | <br>*16.0%* |

*All results indicate greedy decoding performance without external Language Models (LMs).*

## License

This legacy checkpoint is archived and released under the **CC-BY-4.0** license.

---

**Repository & Issues:** Technical tracking for this legacy series can be referenced at [RobotsMali-AI/bambara-asr](https://github.com/RobotsMali-AI/bambara-asr/). No further architectural expansions or fine-tuning updates are planned for this model card sequence.
