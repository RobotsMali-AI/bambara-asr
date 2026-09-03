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
  - name: reference
    dtype: string
  - name: RobotsMali/stt-bm-quartznet15x5-v0
    dtype: string
  - name: RobotsMali/stt-bm-quartznet15x5-v1
    dtype: string
  - name: RobotsMali/soloba-ctc-0.6b-v0
    dtype: string
  - name: RobotsMali/soloba-ctc-0.6b-v1
    dtype: string
  - name: RobotsMali/soloni-114m-tdt-ctc-v0
    dtype: string
  - name: RobotsMali/soloni-114m-tdt-ctc-v1
    dtype: string
  - name: RobotsMali/stt-bm-quartznet15x5-v2
    dtype: string
  - name: soloni-114m-tdt-ctc-v2
    dtype: string
  splits:
  - name: test
    num_bytes: 20690456
    num_examples: 45
  download_size: 19975985
  dataset_size: 20690456
task_categories:
- automatic-speech-recognition
language:
- bm
tags:
- speech
- asr
- bambara
- low-resource
---
# Nyana-Eval Dataset

## Dataset Description

**Nyana-Eval** is a compact, stratified evaluation subset for benchmarking Automatic Speech Recognition (ASR) models in Bambara. It consists of **45 audio recordings** totaling approximately **3.03 minutes**, carefully selected to represent real-world linguistic and acoustic challenges in low-resource Bambara speech. This dataset is derived from the larger [RobotsMali/Bam_ASR_Eval_500](https://huggingface.co/datasets/RobotsMali/Bam_ASR_Eval_500) corpus and is optimized for quick, reproducible human evaluation.

Nyana-Eval is ideal for:
- Rapid evaluation of Bambara ASR models (e.g., WER/CER computation on diverse conditions).
- Human-assisted qualitative analysis (e.g., semantic fidelity, code-switching handling).
- Testing models on low-resource settings gaps: dialectal variations, noise, proper names, and code-mixing with French.

**Key Statistics**:
- **Total Samples**: 45 (balanced: 15 per source subset).
- **Total Duration**: ~3.03 minutes (average ~4.0 seconds per sample).
- **Audio Format**: Mono-channel WAV files at 16 or 44.1k kHz sampling rate.
- **Languages**: Primary: Bambara (Bamana); Secondary: French code-switching (~15% of samples).
- **License**: CC-BY-4.0 License (open for research, commercial use with attribution).

Compiled by Robots Mali AI4D Lab, this dataset powers the human-comparative analysis in the [Bambara ASR Models Evaluation Report].

## Dataset Structure

Nyana-Eval is a single-split dataset (default: `test`), with each entry including raw audio, duration, transcriptions (reference) and models transcriptions.

### Features/Columns

| Column          | Type     | Description                                                                 | Example Value |
|-----------------|----------|-----------------------------------------------------------------------------|---------------|
| `audio`        | Audio   | Raw audio waveform (array + sampling rate: 16 or 44.1k kHz) or file path.      | `{"path": "1.1.wav", "array": [...], "sampling_rate": 16000}` |
| `duration`     | Float64 | Length of the audio clip in seconds (range: 0.62s – 15s).                     | 3.45         |
| `references`   | String  | Bambara text                                                                  | "nɔgɔ ye a ka tɔɔrɔ ye" |
| '8 * models transcriptions' | String  | ASR provised transcriptions                                      |  |

### Splits
- **Default Split**: Full 45 samples (`test` for evaluation).
- **Subsets by Source**: Balanced 15 samples each from the three parent subsets (see Sources below).

To load in Python (via Hugging Face Datasets):
```python
from datasets import load_dataset
dataset = load_dataset("RobotsMali/nyana-eval", split="test")
print(len(dataset))  # Output: 45
print(dataset[0])    # Example: {'audio': ..., 'duration': 3.45, 'transcription': 'adama dusukasilen ye a sigi'}
```

## Sources and Compilation

Nyana-Eval is a **balanced subsample (15 per subset)** from the full 500-sample [RobotsMali/Bam_ASR_Eval_500](https://huggingface.co/datasets/RobotsMali/Bam_ASR_Eval_500) corpus (~36.69 minutes total). Selection criteria ensured diversity: voice variety (age/gender/accents), acoustic challenges (noise/volume/overlaps), and linguistic phenomena (code-switching, proper names etc.)

**Parent Subsets Breakdown** (15 samples each in Nyana-Eval):
- **Ref. 1: RobotsMali/kunkado (Hugging Face)** – 15 audios (~1.96 minutes scaled).  
  Semi-supervised interviews and spontaneous discourse. Source: [RobotsMali/kunkado](https://huggingface.co/datasets/RobotsMali/kunkado). Focus: Dialectal variations and natural flow.

- **Ref. 2: jeli-ASR street interviews subset** – 30 audios (~1.85 minutes).  
  Street interviews Subset from the jeli-asr project. Source: [jeli-asr](https://github.com/robotsmali-ai/jeli-asr/)

- **Ref. 3: Readings of Excerpts from An Bɛ Kalan app (RobotsMali)** – 220 audios (~20.06 minutes).  
  User-generated readings and interactions from the mobile app for Bambara learning, captures learner speech with occasional errors or pauses. source: [RobotsMali-AI/an-be-kalan](https://github.com/Robotsmali-ai/an-be-kalan)

## Metadata

### General Metadata
- **Creator**: Robots Mali AI4D Lab
- **Version**: 1.0 (November 2025).
- **Creation Date**: Derived November 2025 from Bam_ASR_Eval_500.
- **Update Frequency**: Static (expansions via parent dataset).
- **Download Size**: ~25 MB (audios + metadata).
- **Ethical Notes**: Ethically sourced/anonymized; focuses on public-domain cultural speech. For research; cite Robots Mali.

### Challenges Represented:
  - Code-switching: samples (e.g., "Segou ville").
  - Proper names (e.g., "Sunjata," "Traoré").
  - Noise/Overlaps: (e.g., low-volume interviews, multi-speaker).

## Related Resources
- **Parent Dataset**: [RobotsMali/Bam_ASR_Eval_500](https://huggingface.co/datasets/RobotsMali/Bam_ASR_Eval_500) (full 500 samples).
- **Models**: Test with [RobotsMali ASR models](https://huggingface.co/RobotsMali/models)
- **App**: Collect similar data via [An Bɛ Kalan](https://play.google.com/store/apps/details?id=org.robotsmali.literacy_app).

This README is self-contained; explore the attached report PDF for detailed human annotations and model rankings on these exact 45 samples!

## Citation

```bibtex
@dataset{robotsmali_nyana_eval_2025,
  author       = {RobotsMali AI4D Lab},
  title        = {Nyana-Eval: 45-sample Human-Evaluated Bambara ASR Test Set},
  year         = {2025},
  url          = {https://huggingface.co/datasets/RobotsMali/nyana-eval},
  note         = {Stratified subset of Bam_ASR_Eva_500 used for human + WER evaluation}
}