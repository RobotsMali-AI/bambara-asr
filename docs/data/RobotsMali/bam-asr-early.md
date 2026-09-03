---
language: 
- bm  # ISO 639-1 code for Bambara
- fr  # ISO 639-1 code for French
pretty_name: "Bambara-ASR-All Audio Dataset"
version: "1.0.1" # Explicit versioning
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
- oza-mali-pense
- rt-data-collection
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
  total_audio_files: 38769
  total_duration_hours: ~37

configs:
- config_name: bam-asr-all
  default: true
  data_files:
  - split: train
    path: 
      - "oza-mali-pense/train/*.arrow"
      - "rt-data-collection/train/*.arrow"
      - "bam-asr-oza/train/*.arrow"
      - "jeli-asr-rmai/train/*.arrow"
  - split: test
    path: 
      - "bam-asr-oza/test/*.arrow"
      - "jeli-asr-rmai/test/*.arrow" 
- config_name: jeli-asr
  data_files:
  - split: train
    path: 
      - "bam-asr-oza/train/*.arrow"
      - "jeli-asr-rmai/train/*.arrow"
  - split: test
    path: 
      - "bam-asr-oza/test/*.arrow"
      - "jeli-asr-rmai/test/*.arrow" 
- config_name: oza-mali-pense
  data_files:
  - split: train
    path: "oza-mali-pense/train/*.arrow"
- config_name: rt-data-collection
  data_files:
  - split: train
    path: "rt-data-collection/train/*.arrow"

description: |
  The **Bambara-ASR-Early Audio Dataset** is a multilingual dataset containing audio samples in Bambara, accompanied by semi-expert transcriptions and French translations. 
  The dataset includes various subsets: `jeli-asr`, `oza-mali-pense`, and `rt-data-collection`. Each audio file is aligned with Bambara transcriptions or French translations, making it suitable for tasks such as automatic speech recognition (ASR) and translation. 
  Data sources include all publicly available collections of audio with Bambara transcriptions as of December 2024, organized for accessibility and usability.

---

# All Bambara ASR Dataset

This is the dataset that fueled our early ASR experiments that gave as results the V0 models. It is primarily composed of the **Jeli-ASR** dataset (available at [RobotsMali/jeli-asr](https://huggingface.co/datasets/RobotsMali/jeli-asr)), along with the **Mali-Pense** data curated and published by Aboubacar Ouattara (available at [oza75/bambara-tts](https://huggingface.co/datasets/oza75/bambara-tts)). Additionally, it includes 1 hour of audio recently collected by the RobotsMali AI4D Lab, featuring children's voices reading some of RobotsMali GAIFE books. This dataset is designed for automatic speech recognition (ASR) task primarily.

## Important Notes

1. Please note that this dataset is currently in development and is therefore not fixed. The structure, content, and availability of the dataset may change as improvements and updates are made.

---

## **Key Changes in Version 1.0.1 (December 17th)**

This version extends the same updates as Jeli-ASR 1.0.1 at the transcription level. The transcription were normalized using the [Bambara Normalizer](https://pypi.org/project/bambara-normalizer/), a python package designed to normalize Bambara text for different NLP applications.

Please, let us know if you have feedback or additional use suggestions for the dataset by opening a discussion or a pull request. You can find a record or updates of the dataset in [VERSIONING.md](VERSIONING.md)

---

## **Dataset Details**

- **Total Duration**: 37.41 hours
- **Number of Samples**: 38,769 
  - **Training Set**: 37,306 samples
  - **Testing Set**: 1,463 samples

### **Subsets**:

- **Oza's Bambara-ASR**: \~29 hours (clean subset).
- **Jeli-ASR-RMAI**: \~3.5 hours (filtered subset).
- **oza-tts-mali-pense**: \~4 hours
- **reading-tutor-data-collection**: \~1 hour

---

## **Usage**

The data in the main branch are in .arrow format for compatibility with HF's Datasets Library. So you don't need any ajustement to load the dataset directly with datasets:

```python
from datasets import load_dataset

# Load the dataset into Hugging Face Dataset object
dataset = load_dataset("RobotsMali/bam-asr-all")
```

However, an "archives" branch has been added for improved versioning of the dataset and to facilitate usage for those working outside the typical Hugging Face workflow. Precisely the archives are created from the directory of version 1.0.0 tailored for usage with NVIDIA's NEMO. If you prefer to reconstrcut the dataset from archives you can follow the instructions below.

### Downloading the Dataset:

You could download the dataset by git cloning this branch:

```bash
# Clone dataset repository maintaining directory structure for quick setup with Nemo
git clone --depth 1 -b archives https://huggingface.co/datasets/RobotsMali/bam-asr-all
```

Or you could download the individual archives that you are interested in, thus avoiding the git overload

```bash
# Download the audios with wget
wget https://huggingface.co/datasets/RobotsMali/bam-asr-all/resolve/archives/audio-archives/bam-asr-all-1.0.0-audios.tar.gz
# Download the manifests in the same way
wget https://huggingface.co/datasets/RobotsMali/bam-asr-all/resolve/archives/manifests-archives/bam-asr-all-1.0.1-manifests.tar.gz
```

Finally, untar those files to reconstruct the default Directory structure of jeli-asr 1.0.0:

```bash
# untar the audios
tar -xvzf bam-asr-all-1.0.0-audios.tar.gz
# untar the manifests
tar -xvzf bam-asr-all-1.0.1-manifests.tar.gz
```

This approach allow you to combine the data from different versions and restructure your working directory as you with, with more ease and without necessarily having to write code.

## **Known Issues**

This dataset also has most of the issues of Jeli-ASR, including a few misaligned samples. Additionally a few samples from the mali pense subset and all the data from the rt-data-collection subset don't currently have french translations

---

## **Citation**

If you use this dataset in your research or project, please credit the creators of these datasets.

- **Jeli-ASR dataset**: [Jeli-ASR Dataset](https://huggingface.co/datasets/RobotsMali/jeli-asr). 
- **Oza's Bambara-ASR dataset**: [oza75/bambara-asr](https://huggingface.co/datasets/oza75/bambara-asr)
- **Oza's Bambara-TTS dataset**: [oza75/bambara-tts](https://huggingface.co/datasets/oza75/bambara-tts)
