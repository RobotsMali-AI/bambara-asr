---
dataset_info:
- config_name: human-reviewed
  features:
  - name: audio
    dtype: audio
  - name: duration
    dtype: float64
  - name: semi-label
    dtype: string
  - name: corrected-label
    dtype: string
  splits:
  - name: train
    num_bytes: 9838671594.158
    num_examples: 33282
  - name: test
    num_bytes: 1754436439.925
    num_examples: 5775
  download_size: 11503545995
  dataset_size: 11593108034.083
- config_name: semi-first
  features:
  - name: audio
    dtype:
      audio:
        decode: false
  - name: duration
    dtype: float64
  - name: semi-label
    dtype: string
  splits:
  - name: train
    num_bytes: 12219340374.572
    num_examples: 41366
  download_size: 12146855870
  dataset_size: 12219340374.572
- config_name: semi-second
  features:
  - name: audio
    dtype:
      audio:
        decode: false
  - name: duration
    dtype: float64
  - name: semi-label
    dtype: string
  splits:
  - name: train
    num_bytes: 11588268645.052
    num_examples: 38502
  download_size: 23763534431
  dataset_size: 11588268645.052
configs:
- config_name: human-reviewed
  data_files:
  - split: train
    path: human-reviewed/train-*
  - split: test
    path: human-reviewed/test-*
- config_name: semi-first
  data_files:
  - split: train
    path: semi-first/train-*
- config_name: semi-second
  data_files:
  - split: train
    path: semi-second/train-*
license: cc-by-sa-4.0
task_categories:
- automatic-speech-recognition
- translation
language:
- bm
- fr
tags:
- semi-labelled
- asr
- speech-recognition
- code-switching
- bambara
pretty_name: Kunnafonidilaw ka cadeau (Kunkado)
size_categories:
- 100K<n<1M
---

# Kunnafonidilaw ka **cadeau**  🇲🇱

*A messy‑real Bambara ASR corpus for developing modern speech models & code‑switch studies*

## Quick Facts

|                     | value                                                               |
| ------------------- | ------------------------------------------------------------------- |
| **Total duration**  | **161.15 h**                                                        |
| **Reviewed subset** | 39.3 h (≈ 25 %)                                                     |
| **Total segments**  | 118 925                                                             |
| **Languages**       | Bambara (majority) • French (code‑switch) • misc. Arabic (translit) |
| **LICENSE**         | CC‑BY‑SA 4.0                                                        |

`kunkado` aims to mirror how Malians speak bambara **today**: fast, informal, and full of French code‑switching.  We hope it fuels *robust* ASR systems and research on contact phenomena in Mande languages.

---

## Motivation

Low‑resource corpora are often small & squeaky‑clean, giving models a shock once deployed. For this dataset we wanted as features all the things that we conventionally reject in manageable ASR datasets. It is especially designed for training end-to-end Deep Learning systems as the learning task is quite complex (if no normalization applied). In this dataset you will encounter:

* code‑switches (tagged with `__double underscores__`)
* music, jingles & phone buzzes
* accept rough silence‑proxy segmentation (cut words are marked with `…`)

Only \~25 % could be human‑reviewed in the project timeframe, but **community PRs are welcome**.

---

## Characteristics of the dataset

* ***Present Bamako Bambara***: Reflective of how Malian people naturally speak Bamanankan.
  → This includes urban speech patterns, contractions, and informal expressions commonly heard in Bamako.

* **Broad range of topics**:
  → Content spans casual conversations, news, politics, religion, comedy, marketplace discussions, and social commentary.

* **Numbers transcribed as digits**:
  → This choice was intented primarily for faster human transcription and unifying the semi labels which sometimes uses different style

* **A lot of code-switching (represented in the transcriptions with underscores)**:
  → French & Arabic insertions are delimited with `__` markers, making it easier to identify multilingual segments.

* **Most segments feature more than one voice and interactions/interference between speakers**:
  → This reflects the natural occurrence of overlapping speech in real-world dialogues and group settings.


### Duration buckets (seconds)

| bucket (s) | human‑reviewed | semi-first | semi‑second | total       |
| ---------- | -------------- | --------   | ----------- | ----------- |
| 0.6 – 15   | 39 057         | 41366      | 38 502      | **111 746** |
| 15 – 30    | 0              | 0          | 5 402       | 5 402       |
| 30 – 45    | 0              | 0          | 1 777       | 1 777       |

### Subsets

The dataset has been uploaded in three subsets, the human reviewed subset is separated from the other  for organization purposes and the semi labelled entries have been slitted into two subsets due to resources constraints during upload

#### **human‑reviewed**  *(default)*

* **Total** 39.27 h – 39 057 short utterances
* **Splits**
  • **Train** 33 282 utt. – 33.47 h
  • **Test**   5 775 utt. – 5.80 h (≈ 15 %)

#### **semi-first**

* **Total** 41.47 h – 41 366 short utterances

#### **semi-second**

* **Total** 80.42 h – 38 502 variable length utterances (0.6 to 45s)

### Tags

| Tag                  | Meaning                   |
| -------------------- | ------------------------- |
| `<BRUITS>`           | generic noise             |
| `<INCOMPRÉHENSIBLE>` | fully inaudible speech    |
| `<CHEVAUCHEMENT>`    | speaker overlap           |
| `<RIRES>`            | laughter                  |
| `<MUSIQUE>`          | music / jingle (no lyrics)|
| `<TOUX>`             | cough                     |
| `<INVOCATION>`       | prayers, quranic excerpts |
| `<ECHO>`             | echo artefact             |
| `<APPLAUDISSEMENTS>` | applause                  |
| `<CRIS>`             | screams                   |
| `<PLEURES>`          | crying                    |

### Recommended Normalisation

Numbers appear mostly **as digits** and may violate a single style.  We strongly advise applying [`number normalization`](https://pypi.org/project/bambara-normalizer/) them before training.

---

## Source & Provenance

| Donor                                  | Hours | Media type      |
| -------------------------------------- | ----- | --------------- |
| Radio Benkouma “La voix du Baramousso” | 32.7  | Community Radio |
| Mousso TV                              | 23.2  |       TV        |
| [ORTM](https://ortm.ml/) (National TV) | 7     |    TV/Radio     |
| Radio Sahel FM                         | 98.4  | Regional Radio  |

Audio was graciously provided by the broadcasters listed above; the complete corpus is released under **CC‑BY‑SA 4.0**.  Automatic transcripts were generated with **[soloni‑114M](https://huggingface.co/RobotsMali/soloni-114m-tdt-ctc-V0)** and then manually corrected for \~40 h by our team of annotators.

*NB: Semi Labels might be updated in future versions*

### Annotators

Karounga Kanté • Boureima Traoré • Diakaridia Bengaly • Tidiane Koné • Lanseni
Mallé • Séni Togninè • Assanatou Soumaoro • *Alassane Koné*  •
*Benogo Fofana* • *Aboubacar Traoré*

Huge thanks to our donors & reviewers – ***aw ni ce***!

---

## Known Issues & Caveats

* **Segmentation**: silence‑proxy; some utterances cut mid‑word.
* **Spelling issues**: Misspelling of foreign phrases and arabic transliterations
* **Code‑switch inconsistencies**: Arabic phrases sometimes tagged, sometimes not.
* **Number style**: digits vs. letters  not strictly respected.
* **Rare pure French segments** remain in the dataset.

---

## Usage Example

```python
from datasets import load_dataset

ds = load_dataset("RobotsMali/kunkado", split="train")
print(ds[0]["corrected-label"])
```

---

## License

**Creative Commons Attribution–ShareAlike 4.0 International** (CC‑BY‑SA 4.0). You may share and adapt, even commercially, as long as you credit the
contributors **and** keep derivatives under the same licence.*  No warranty.

---

## Citation

```bibtex
@misc{diarra2025kunnafonidilawkacadeauasr,
      title={Kunnafonidilaw ka Cadeau: an ASR dataset of present-day Bambara}, 
      author={Yacouba Diarra and Panga Azazia Kamate and Nouhoum Souleymane Coulibaly and Michael Leventhal},
      year={2025},
      eprint={2512.19400},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.19400}, 
}
```
---

*Maintained by RobotsMali AI4D Lab — PRs & issue reports welcome!*