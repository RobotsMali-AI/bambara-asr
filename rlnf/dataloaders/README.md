# Data Loaders

The `rlnf/dataloaders/` package provides PyTorch-`Dataset` classes to serve both the RLHF pipeline and reward-model training workflows. It handles audio preprocessing, transcript tokenization, and normalization of human feedback scores.

The data loaders rely on:

* `nemo_toolkit` (for `AudioToMelSpectrogramPreprocessor` class)
* `sentencepiece` (for text tokenization)
* `torch`

These are declared in `rlnf/requirements.txt` and installed with the main package.

---

## Modules

### `audio_dataset.py`

**Class: `AudioDataset`**

Takes a manifest file path, loads raw audio files and applies the preprocessor.

#### Initialization

```python
from rlnf.dataloaders.audio_dataset import AudioDataset

# Audio preprocessor Config
preprocessor_config = {
    'normalize': 'per_feature',
    'window_size': 0.02,
    'sample_rate': 16000,
    'window_stride': 0.01,
    'window': 'hann',
    'features': 64,
    'n_fft': 512,
    'frame_splicing': 1,
    'dither': 1e-05,
    'stft_conv': 
}

dataset = AudioDataset(
    manifest_path="path/audio_manifest.jsonl",  # JSON list of {"audio_filepath"}
    preprocessor_config=preprocessor_config, 
)
```

* Reads `manifest_path` (one JSON per line).
* Instantiates an `AudioToMelSpectrogramPreprocessor` under the hood and returns preprocessed audio as items.

#### `__getitem__`

Returns:

```python
audio_feats: Tensor [N_MEL, T]
```

Both ready for batching in a PyTorch `DataLoader`.

### `reward_dataset.py`

**Class: `RewardDataset`**

Bundles audio, transcript text, and human-provided quality scores for training the reward model or RLHF rollouts.

#### Initialization

```python
from rlnf.dataloaders.reward_dataset import RewardDataset

dataset = RewardDataset(
    manifest_path="path/reward_manifest.jsonl"",  # JSONL: audio_path, transcript, score
    tokenizer_model_path="path/to/tokenizer.model",
    preprocessor_config=preprocessor_config,
)
```

* Reads a JSONL with attributes: `audio_path`, `transcript`, `score`.
* Extracts audio features just like `AudioDataset` and tokenizes text transcripts using the provided pretrained sentencepiece tokenizer model.
* Normalizes `score` to a 0–1 range if specified.

#### `__getitem__`

Returns a tuple:

```python
(audio_feats, transcript_ids, score)
```

* `audio_feats`: FloatTensor \[MEL, T]
* `transcript_ids`: LongTensor \[L]
* `score`: FloatTensor scalar

---
