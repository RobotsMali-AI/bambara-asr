---
license: cc-by-4.0
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: duration
    dtype: float64
  - name: transcription
    dtype: string
  splits:
  - name: test
    num_bytes: 130530705
    num_examples: 500
  download_size: 128930905
  dataset_size: 130530705
language:
- bm
task_categories:
- automatic-speech-recognition
tags:
- bambara
- asr
- low-resource
- eval
---
# Bam_ASR_Eval_500 Dataset

## Dataset Description

**Bam_ASR_Eval_500** is a curated evaluation dataset for Automatic Speech Recognition (ASR) models in Bambara (Bamanakan), a major language spoken in Mali and West Africa. This dataset comprises **500 audio recordings** totaling approximately **36.69 minutes** of annotated speech, designed specifically for benchmarking ASR systems. It focuses on real-world challenges in low-resource languages like Bambara, including spontaneous speech, dialectal variations, code-switching with French and Arabic, proper names, and environmental noise.

The dataset is particularly useful for:
- Evaluating ASR model performance on diverse Bambara audio (e.g., interviews, readings, street conversations).
- Fine-tuning Bambara ASR models, with a focus READ data, acoustic robustness and code-swicthing; although the dataset is primarily intended for evaluating models due to its small size.
- Research in multilingual ASR for African languages, addressing issues like out-of-vocabulary (OOD) words (e.g., names like "Kouyaté" or "Diabaté") and code-mixing.

**Key Statistics**:
- **Total Samples**: 500
- **Total Duration**: ~36.69 minutes (average ~4.4 seconds per sample)
- **Audio Format**: Mono-channel WAV files at 16-44.1 kHz sampling rate
- **Languages**: Primary: Bambara (Bamanankan); Secondary: Occasional French code-switching
- **License**: CC-BY-4.0 License (open for research, commercial use with attribution)


This dataset was compiled by Robots Mali AI4D Lab as part of ongoing efforts to advance AI for underrepresented malian languages. It serves as a benchmark corpus for models like those published by [RobotsMali Hugging Face organization](https://huggingface.co/RobotsMali).

### Features/Columns

| Column          | Type     | Description                                                                 | Example Value |
|-----------------|----------|-----------------------------------------------------------------------------|---------------|
| `audio`         | Audio    | Raw audio waveform (array + sampling rate: 16 kHz or 44.1 kHz) or file path. | `{"path": "audio_1.wav", "array": [...], "sampling_rate": 16000}` |
| `duration`      | Float64  | Length of the audio clip in seconds                                         | 0.5         |
| `transcription` | String   | Bambara text                                                                | "adama dusukasilen ye a sigi" |

### Splits
- **test Split**: Full 500 samples.

To load in Python (via Hugging Face Datasets):
```python
from datasets import load_dataset
dataset = load_dataset("RobotsMali/Bam_ASR_Eval_500", split="test")
print(dataset["test"][0])  # Example: {'audio': ..., 'duration': 3.45, 'transcription': 'nɔgɔ ye a ka tɔɔrɔ ye'}
```

  Semi-supervised data from interviews and spontaneous discourse. Source: [RobotsMali/kunkado](https://huggingface.co/datasets/RobotsMali/kunkado). Focus: Natural conversations with dialectal variations.

- **Ref. 2: jeli-ASR street interviews subset** – 30 audios (~1.85 minutes).  
  Street interviews Subset from the jeli-asr project. Source: [jeli-asr](https://github.com/robotsmali-ai/jeli-asr/)

- **Ref. 3: Readings of Excerpts from An Bɛ Kalan app (RobotsMali)** – 220 audios (~20.06 minutes).  
  User-generated readings and interactions from the mobile app for Bambara learning. source: [RobotsMali-AI/an-be-kalan](https://github.com/Robotsmali-ai/an-be-kalan)

## Metadata
- **Creator**: Robots Mali AI4D Lab
- **Version**: 1.0 (November 2025).
- **Creation Date**: Compiled in late 2025.
- **Update Frequency**: Not specified / As-needed (e.g., expansions for new subsets).
- **Download Size**: ~150 MB (compressed audios + metadata).
- **Ethical Notes**: Data sourced ethically; anonymized where possible. No sensitive personal info. Intended for non-commercial research; cite Robots Mali for use.

## Citation
```bibtex
@dataset{RobotsMali_Bam_ASR_Eval_500,
  author       = {RobotsMali AI4D Lab},
  title        = {Bam_ASR_Eval_500: 500 Stratified Bambara Speech Samples for ASR Evaluation},
  year         = {2025},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/RobotsMali/Bam_ASR_Eval_500},
}