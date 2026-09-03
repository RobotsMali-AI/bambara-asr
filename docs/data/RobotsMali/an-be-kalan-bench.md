---
language:
- bm
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
tags:
- audio
- speech
- low-resource-languages
- bambara
- education
dataset_info:
- config_name: duplicate
  features:
  - name: BookTitle
    dtype: string
  - name: sentenceID
    dtype: int64
  - name: text
    dtype: string
  - name: speakerAge
    dtype: int64
  - name: speakerGender
    dtype: string
  - name: speakerID
    dtype: string
  - name: duration
    dtype: float64
  - name: audio
    dtype: audio
  splits:
  - name: train
    num_bytes: 3749295635.885
    num_examples: 33481
  download_size: 4921102766
  dataset_size: 3749295635.885
- config_name: main
  default: true
  features:
  - name: BookTitle
    dtype: string
  - name: sentenceID
    dtype: int64
  - name: text
    dtype: string
  - name: speakerAge
    dtype: int64
  - name: speakerGender
    dtype: string
  - name: speakerID
    dtype: string
  - name: duration
    dtype: float64
  - name: audio
    dtype: audio
  splits:
  - name: train
    num_bytes: 184917975.603
    num_examples: 1203
  - name: test
    num_examples: 724
  download_size: 180050364
  dataset_size: 184917975.603
configs:
- config_name: duplicate
  data_files:
  - split: train
    path: duplicate/train-*
- config_name: main
  default: true
  data_files:
  - split: train
    path: main/train-*
  - split: test
    path: main/test-*
---
# Bambara Educational Speech Dataset

This dataset is a collection of READ Bambara text based on educational children's books from RobotsMali's GAIFE project. It is designed to support the training and benchmarking of Automatic Speech Recognition (ASR) models, with a particular focus on child speech, regional acoustics, and repetitive text structures (inherent to the domain).

The dataset is structured into two separate subsets to support specialized training and evaluation paradigms:
1. **`main`**: Contains non-overlapping text and speakers divided into standard training and test splits.
2. **`duplicate`**: A highly dense, multi-speaker redundant training set featuring multiple recordings of the same source literature by a diverse pool of speakers.

---

## Dataset Architecture & Splits

The dataset enforces a strict split philosophy to ensure structural and evaluation integrity. A book present in the test split (the benchmark) **never** appears in another split. Though there might be rare speaker overlap due to inconsistent speaker identity handling araising from the data collection pipeline.

### Summary Statistics Table

| Metric | `main` (Train) | `main` (Test) | `duplicate` (Train) | Combined |
| :--- | :---: | :---: | :---: | :---: |
| **Total Unique Speakers** | 8 | 11 | 113 | 113 |
| **Mean Audio Duration** | 4.86s | 4.42s | 7.73s | 4.74s |
| **Audio Duration Range** | 0.72s – 25.60s | 0.32s – 16.48s | 0.08s – 37.52s | 0.08s – 37.52s |
| **Mean Sentence Length** | 8.06 words | 7.18 words | 7.85 words | 7.85 words |
| **Sentence Length Range** | 1 – 26 words | 1 – 21 words | 1 – 26 words | 1 – 26 words |
| **Missing Speaker Metadata**| 0 utterances | 38 utterances | 0 utterances | 0 utterances |

### Demographic Distributions

* **Age Profile**: The dataset exclusively features children and young adolescents. Across the entire combined footprint, speaker age ranges from a **minimum of 8 to a maximum of 19 years old**.
* **Gender Splits (Combined)**: **Female**: 17,471 utterances | **Male**: 17,213 utterances.
* **Gender Splits (`main` Test)**: **Female**: 522 | **Male**: 164 | **Unknown**: 38.
* **Gender Splits (`main` Train)**: **Female**: 783 | **Male**: 420.

---

## Detailed Book Inventory & Duplicate Metrics

The duplicates subset is derived from 20 core books.

### `duplicate` (Train) Subset Book Inventory (44.02 h)

| Book Title | Total Reads (Proxied) | Unique Speaker IDs | Female Reads | Male Reads | Age Distribution Split (<10 / 10-15 / 15-20) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Aminata Fari Fisayara** | 36 | 34 | 17 | 19 | 3 / 19 / 14 |
| **Bakɔrɔnin Saba** | 37 | 36 | 18 | 19 | 3 / 20 / 14 |
| **Bɛnkɛ Tɔm Ka So** | 26 | 24 | 14 | 12 | 1 / 13 / 12 |
| **Dawuda ni a Mɔkɛ** | 30 | 30 | 15 | 15 | 1 / 15 / 14 |
| **Dɔgɔtɔrɔ ni Farafinfurabɔla a** | 23 | 22 | 14 | 9 | 2 / 10 / 11 |
| **Filomani** | 32 | 31 | 18 | 14 | 2 / 16 / 14 |
| **Gawusu ni Masakɛ Sidiki** | 29 | 29 | 14 | 15 | 3 / 16 / 10 |
| **Gerenadi-Feerew** | 29 | 27 | 14 | 15 | 1 / 17 / 11 |
| **Gesedala Musa** | 27 | 26 | 15 | 12 | 2 / 15 / 10 |
| **Gundola Kuma** | 31 | 27 | 18 | 13 | 3 / 18 / 10 |
| **Kalo la Taama** | 26 | 25 | 13 | 13 | 1 / 15 / 10 |
| **Kan Orobotik** | 27 | 27 | 15 | 12 | 2 / 12 / 13 |
| **Korokara Yɛrɛdɔnbali** | 24 | 24 | 12 | 12 | 0 / 14 / 10 |
| **Kurun** | 25 | 25 | 11 | 14 | 0 / 14 / 11 |
| **Lamini Ka Don Kɛrɛnkɛrɛnnen** | 24 | 24 | 10 | 14 | 1 / 13 / 10 |
| **Mama ka Sama** | 24 | 24 | 10 | 14 | 0 / 13 / 11 |
| **Ne ni Mama ka Gafe Kalan** | 27 | 26 | 12 | 15 | 3 / 16 / 8 |
| **Subahana Daga** | 29 | 28 | 12 | 17 | 2 / 17 / 10 |
| **Sɔminiminɛnw** | 29 | 29 | 13 | 16 | 3 / 13 / 13 |
| **Yɛlɛ Ka Di Npogotiginin Mi Ye** | 21 | 20 | 9 | 12 | 0 / 13 / 8 |

### `main` (Train) Subset Book Inventory (1.62 h)
Made of unique readings of 22 books, the 20 books in `duplicate` + 2: 
* *Saratu*
* *Tulonkɛw*

### Test Subset Book Inventory (0.89 h)
The following distinct literary works compose the `main` test subset footprint (no repetition):
* *Anw Bɛ Baara Kɛ!*
* *Bako Cɛnin Ŋaniya Ɲuman*
* *Bama Miirina*
* *Cɛni Tulogɛlɛn*
* *Donfɛnw*
* *Fali Nalonma Ni Ba Kegunma ani Ɲininkaliw ni u j*
* *Jate*
* *Kogo*
* *Kulɔriw*
* *Ne ni Papa ka Gafe Kalan*
* *Ni a tun bɛ se ...*

---

## Critical Considerations: Features vs. Weaknesses

When developing models on this dataset, users should balance its unique profile against known recording constraints:

### 1. The High-Volume Duplication Matrix
* **As a Feature:** Due to the severe scarcity of open-source text and educational literature in Bambara, collecting deep audio variations on a finite set of text was a deliberate design choice. This split provides a robust playground for specific speech experiments, such as **acoustic multi-speaker verification, text-constrained acoustic profiling, and downstream speech representation probing**.
* **As a Weakness:** The textual diversity in the duplicate subset is inherently bottlenecked by the underlying literature. Models trained aggressively on the duplicate split without constraint can rapidly overfit to the vocabulary, tone structures, and phonetic bounds of these 20 specific books.

### 2. Known Metadata Inconsistencies
* **The Identifier Inconsistency:** The log metrics show 113 unique speaker IDs in the combined dataset slice. However, on-the-ground project coordinators reported a real-world count of **approximately 60 unique speakers**.
* **Implication:** This discrepancy highlights an operational metadata inflation error where individual speakers were assigned differing tracking IDs across different recording sessions, days, or environments. Users should exercise caution when benchmarking strict zero-shot speaker verification algorithms on this dataset without manual speaker clustering.

---

## Dataset Format

Data manifests are deployed using the standard JSON Lines (`.jsonl`) configuration natively compatible with deep learning toolkits like NVIDIA NeMo and ESPnet:

```json
{
  "BookTitle": "Aminata Fari Fisayara",
  "sentenceID": 1,
  "text": "Mɔgɔ caman tun bɛ yen.",
  "speakerAge": 15,
  "speakerGender": "male",
  "speakerID": "spk_1778182779107_12bb3ea9c8",
  "duration": 2.56,
  "audio_filepath": "data/audios/dup_Aminata_Fari_Fisayara_1_1200.wav"
}

```

### Citation

If you utilize this dataset or its subsets in research, please cite the repository card details accordingly.

```
BIBTEX ENTRY COMING SOON
```