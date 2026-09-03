---
dataset_info:
- config_name: human-corrected
  features:
  - name: text
    dtype: string
  - name: duration
    dtype: float64
  - name: audio
    dtype: audio
  - name: label-v1
    dtype: string
  - name: label-v2
    dtype: string
  splits:
  - name: train
    num_bytes: 62771143761
    num_examples: 253290
  - name: test
    num_bytes: 1515394591
    num_examples: 6718
  download_size: 59319505964
  dataset_size: 64286538352
- config_name: model-annotated
  features:
  - name: duration
    dtype: float64
  - name: audio
    dtype: audio
  - name: label-v1
    dtype: string
  - name: label-v2
    dtype: string
  splits:
  - name: train
    num_bytes: 55616591334
    num_examples: 355571
  download_size: 66321575877
  dataset_size: 55616591334
- config_name: short
  features:
  - name: audio
    dtype: audio
  - name: duration
    dtype: float64
  - name: label-v1
    dtype: string
  - name: label-v2
    dtype: string
  splits:
  - name: train
    num_bytes: 16345361845
    num_examples: 259183
  download_size: 16319374978
  dataset_size: 16345361845
configs:
- config_name: human-corrected
  data_files:
  - split: train
    path: human-corrected/train-*
  - split: test
    path: human-corrected/test-*
- config_name: model-annotated
  data_files:
  - split: train
    path: model-annotated/train-*
- config_name: short
  data_files:
  - split: train
    path: short/train-*
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
language:
- bm
tags:
- bambara
- African-Next-Voices
- ANV
- RobotsMali
- afvoices
- asr
pretty_name: African Next Voices - Bambara
---

# 📘 **African Next Voices – Bambara (AfVoices)**

The **AfVoices** dataset is the largest open corpus of spontaneous Bambara speech at its release in late 2025. It contains **423 hours** of segmented audio and **612 hours** of original raw recordings collected across southern Mali. Speech was recorded in natural, conversational settings and annotated using a semi-automated transcription pipeline combining ASR pre-labels and human corrections. We release all the data processing code on [GitHub](https://github.com/RobotsMali-AI/afvoices).

---

## 🔎 **Quick Facts**

| Category                                 | Value                                                                                             |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Total raw hours**                      | 612 h (1,777 raw recordings; publicly available on GCS)                                           |
| **Total segmented hours**                | 423 h (874,762 segments)                                                                          |
| **Speakers**                             | 512                                                                                               |
| **Regions**                              | Bamako, Ségou, Sikasso, Bagineda, Bougouni                                                        |
| **Avg. segment duration**                | ~2 seconds                                                                                        |
| **Subsets**                              | 159 h human-corrected, 212 h model-annotated, 52 h short (<1s)                                    |
| **Age distribution**                     | Broad, across young to elderly speakers (90% between 18 and 45)                                   |
| **Topics**                               | Health, agriculture, Miscellaneous (art, education, history etc.)                                 |
| **SNR distribution (raw recordings)**    | 71.75% High or Very High SNR                                                                      |
| **Train / Test split**                   | 155 h / 4 h                                                                                       |

---

## Load the Dataset

Choose a configuration explicitly because the subsets have different supervision levels:

```python
from datasets import load_dataset

reviewed = load_dataset(
    "RobotsMali/afvoices",
    "human-corrected",
    split="train",
)
automatic = load_dataset(
    "RobotsMali/afvoices",
    "model-annotated",
    split="train",
)
```

Use `text` as the validated reference only in `human-corrected`. The `label-v1` and `label-v2` fields are model-generated hypotheses and must not be presented as human ground truth. The ASR results reported in the accompanying paper and `v2` model cards used an earlier pre-completion subset; its exact manifests are archived in the [`bambara-asr` repository](https://github.com/RobotsMali-AI/bambara-asr/tree/main/afvoices/pre-manifests).

## **Motivation**

The **African Next Voices (ANV)** project is a multi-country effort aiming to gather over **9,000 hours of speech** across 18 African languages. Its goal is to build high-quality datasets that empower local communities, support inclusive AI research, and provide strong foundations for ASR in underrepresented languages.

As part of this initiative, **RobotsMali** led the Bambara data collection for Mali. This dataset reflects RobotsMali’s broader mission to advance AI and NLP research malian languages, with a long-term focus on improving education, access, and technology across Mali and the wider Manding linguistic region.

---

## 🎙️ **Characteristics of the Dataset**

### **Data Collection**

* Speech was collected through trained **facilitators** who guided participants, ensured audio quality, and encouraged natural, topic-focused conversations.
* All recordings are **spontaneous speech**, not read text.
* A custom **Flutter mobile app** ([open-source](https://github.com/RobotsMali-AI/Africa-Voice-App)) was used to simplify the process and reduce training time.
* Geographic focus: **Southern Mali**, to limit extreme accent variation and build a clean baseline corpus.

### **Segmentation and Preprocessing**

* Raw audio was segmented using **Silero VAD**, retaining ~70% of the original duration.
* Segments range from **240 ms to 30 s**.
* Voice activity detection helped remove long silences and improve data usability.

### **Transcriptions**

* Pre-transcribed using the ASR model **soloni-114m-tdt-ctc-v0**.
* Human annotators corrected the transcripts.
* A second model (**soloni-114m-tdt-ctc-v2**) was trained using the corrected transcripts and used to regenerate improved labels.
* Two automatic transcription variants exist for each sample: **v1** (from soloni-v0) and **v2** (from soloni-v2).

### **Acoustic Event Tags**

The following tags appear in transcriptions:

| Tag       | Meaning                                                       |
| --------- | ------------------------------------------------------------- |
| `[um]`    | Vocalized pauses, filler sounds                               |
| `[cs]`    | Code-switched or foreign word                                 |
| `[noise]` | Background noise (applause, coughing, children, etc.)         |
| `[?]`     | Inaudible or overlapped speech                                |
| `[pause]` | Long silence (>5 seconds or >3 seconds at segment boundaries); due to VAD segmentation this tag is rarely used |

---

## 📂 **Subsets**

### **1. Human-corrected (159 h, 260k samples)**

* Fully reviewed and corrected by annotators.
* Only subset with a definitive `text` field containing the validated transcription.

### **2. Model-annotated (212 h, 355k samples)**

* Includes automatic labels: `v1` (soloni-v0) and `v2` (soloni-v2).
* No human review.

### **3. Short subset (52 h, 259k samples)**

* Segments <1 second (formulaic expressions, discourse markers).
* Excluded from human annotation for optimization purposes.
* Automatically labeled (v1 & v2).

---

## ⚠️ **Limitations**

* **Clean dataset vs real-world noise:**
  Over 70% of recordings can be categorized as relatively clean speech. Models trained solely on this dataset may underperform in noisy street or radio environments typical in Mali. See this [report](https://zenodo.org/records/17672774) if you are interested in learning more about the strengths and weaknesses of RobotsMali's ASR models.

* **Reduced code-switching:**
  French terms were often replaced by `[cs]` or normalized into Bambara phonology. This improves model stability but reduces realism for natural bilingual speech.

* **Geographic homogeneity:**
  Focused on the southern region to control accent variability. Broader dialectal coverage might require additional data.

* **Simplified linguistic conditions:**
  Overlaps, multi-speaker settings, and conversational chaos are minimized—again improving training stability at the cost of deployment realism.

---

## 📑 **Citation**

```bibtex
@misc{diarra2025dealinghardfactslowresource,
      title={Dealing with the Hard Facts of Low-Resource African NLP},
      author={Yacouba Diarra and Nouhoum Souleymane Coulibaly and Panga Azazia Kamaté and Madani Amadou Tall and Emmanuel Élisé Koné and Aymane Dembélé and Michael Leventhal},
      year={2025},
      eprint={2511.18557},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.18557},
}
```

---

You may want to download the original 612 hours dataset with its associated metadata for research purposes or to create a derivative. You will find the codes and manifest files to download those files from Google Cloud Storage in this repository: [RobotsMali-AI/afvoices](https://github.com/RobotsMali-AI/afvoices). Do not hesitate to open an issue for Help or suggestions 🤗
