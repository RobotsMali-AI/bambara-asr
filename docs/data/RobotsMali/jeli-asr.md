---
language: 
- bm  # ISO 639-1 code for Bambara
- fr  # ISO 639-1 code for French
pretty_name: "Jeli-ASR Audio Dataset"
version: "1.0.1"  # Explicit versioning
tags:
- audio
- transcription
- multilingual
- Bambara
- French
license: "cc-by-4.0"
task_categories:
- automatic-speech-recognition
- text-to-speech
- translation
task_ids:
- audio-language-identification  # Identifying languages in audio
- keyword-spotting              # Detecting keywords in audio
annotations_creators:
- semi-expert
language_creators:
- crowdsourced  # If the data was annotated or recorded by a team
source_datasets:
- jeli-asr  
size_categories:
- 10GB<
- 10K<n<100K
dataset_info:
  audio_format: "arrow"
  features:
  - name: audio
    dtype: audio
  - name: duration
    dtype: float
  - name: bam
    dtype: string
  - name: french
    dtype: string
  total_audio_files: 33643
  total_duration_hours: ~32

configs:
- config_name: jeli-asr-rmai
  data_files:
  - split: train
    path: "jeli-asr-rmai/train/data-*.arrow"
  - split: test
    path: "jeli-asr-rmai/test/data-*.arrow"
- config_name: bam-asr-oza
  data_files:
  - split: train
    path: "bam-asr-oza/train/data-*.arrow"
  - split: test
    path: "bam-asr-oza/test/data-*.arrow"
- config_name: jeli-asr
  default: true
  data_files:
  - split: train
    path:
      - "bam-asr-oza/train/data-*.arrow"
      - "jeli-asr-rmai/train/data-*.arrow" 
  - split: test
    path: 
      - "bam-asr-oza/test/data-*.arrow"
      - "jeli-asr-rmai/test/data-*.arrow" 
description: |
  The **Jeli-ASR Audio Dataset** is a multilingual dataset converted into the optimized Arrow format, 
  ensuring fast access and compatibility with modern data workflows. It contains audio samples in Bambara 
  with semi-expert transcriptions and French translations. Each subset of the dataset is organized by 
  configuration (`jeli-asr-rmai`, `bam-asr-oza`, and `jeli-asr`) and further split into training and testing sets. 
  The dataset is designed for tasks like automatic speech recognition (ASR), text-to-speech synthesis (TTS), 
  and translation. Data was recorded in Mali with griots, then transcribed and translated into French.
---

# Jeli-ASR Dataset

This repository contains the **Jeli-ASR** dataset, which is primarily a reviewed version of Aboubacar Ouattara's **Bambara-ASR** dataset (drawn from jeli-asr and available at [oza75/bambara-asr](https://huggingface.co/datasets/oza75/bambara-asr)) combined with the best data retained from the former version: `jeli-data-manifest`. This dataset features improved data quality for automatic speech recognition (ASR) and translation tasks, with variable length Bambara audio samples, Bambara transcriptions and French translations.

## Important Notes

1. Please note that this dataset is currently in development and is therefore not fixed. The structure, content, and availability of the dataset may change as improvements and updates are made.

---

## **Key Changes in Version 1.0.1 (December 17th)**

Jeli-ASR 1.0.1 introduces several updates and enhancements, focused entirely on the transcription side of the dataset. There have been no changes to the audio files since version 1.0.0. Below are the key updates:

1. **Symbol Removal:**  
   All non-vocabulary symbols deemed unnecessary for Automatic Speech Recognition (ASR) were removed, including:  
   `[` `]` `(` `)` `«` `»` `°` `"` `<` `>`  

2. **Punctuation Removal:**  
   Common punctuation marks were removed to streamline the dataset for ASR use cases. These include:  
   `:` `,` `;` `.` `?` `!`  
   The exception is the hyphen (`-`), which remains as it is used in both Bambara and French compound words. While this punctuation removal enhances ASR performance, the previous version with full punctuation may still be better suited for other applications. You can still reconstruct the previous version with the archives.

3. **Bambara Normalization:**  
   The transcription were normalized using the [Bambara Normalizer](https://pypi.org/project/bambara-normalizer/), a python package designed to normalize Bambara text for different NLP applications.

4. **Optimized Data Format:**  
   This version introduces `.arrow` files for efficient data storage and retrieval and compatibility with HuggingFace tools.

Let us know if you have feedback or additional use suggestions for the dataset by opening a discussion or a pull request. You can find a record or updates of the dataset in [VERSIONING.md](VERSIONING.md)

---

## **Dataset Details**

- **Total Duration**: 32.48 hours
- **Number of Samples**: 33,643
  - **Training Set**: 32,180 samples (\~95%)
  - **Testing Set**: 1,463 samples (\~5%)

### **Subsets**:

- **Oza's Bambara-ASR**: \~29 hours (clean subset).
- **Jeli-ASR-RMAI**: \~3.5 hours (filtered subset).

Note that since the two subsets were drawn from the original Jeli-ASR dataset, they are just different variation of the same data.

---

## **Usage**

The data in the main branch are in .arrow format for compatibility with HF's Datasets Library. So you don't need any ajustement to load the dataset directly with datasets:

```python
from datasets import load_dataset

# Load the dataset into Hugging Face Dataset object
dataset = load_dataset("RobotsMali/jeli-asr")
```

However, an "archives" branch has been added for improved versioning of the dataset and to facilitate usage for those working outside the typical Hugging Face workflow. Precisely the archives are created from the directory of version 1.0.0 tailored for usage with NVIDIA's NEMO. If you prefer to reconstrcut the dataset from archives you can follow the instructions below.

### Downloading the Dataset:

You could download the dataset by git cloning this branch:

```bash
# Clone dataset repository maintaining directory structure for quick setup with Nemo
git clone --depth 1 -b archives https://huggingface.co/datasets/RobotsMali/jeli-asr
```

Or you could download the individual archives that you are interested in, thus avoiding the git overload

```bash
# Download the audios with wget
wget https://huggingface.co/datasets/RobotsMali/jeli-asr/resolve/archives/audio-archives/jeli-asr-1.0.0-audios.tar.gz
# Download the manifests in the same way
wget https://huggingface.co/datasets/RobotsMali/jeli-asr/resolve/archives/manifests-archives/jeli-asr-1.0.1-manifests.tar.gz
```

Finally, untar those files to reconstruct the default Directory structure of jeli-asr 1.0.0:

```bash
# untar the audios
tar -xvzf jeli-asr-1.0.0-audios.tar.gz
# untar the manifests
tar -xvzf jeli-asr-1.0.1-manifests.tar.gz
```

This approach allow you to combine the data from different versions and restructure your working directory as you with, with more ease and without necessarily having to write code.

## **Known Issues**

While significantly improved, this dataset may still contain some misaligned samples. It has conserved most of the issues of the original dataset such as: 

- Inconsistent transcriptions
- Non-standardized naming conventions.
- Language and spelling issues
- Inaccurate translations

---

## **Citation**

If you use this dataset in your research or project, please credit the creators of the original datasets.

- **Jeli-ASR dataset**: [Original Jeli-ASR Dataset](https://github.com/robotsmali-ai/jeli-asr). 
- **Oza's Bambara-ASR dataset**: [oza75/bambara-asr](https://huggingface.co/datasets/oza75/bambara-asr)
