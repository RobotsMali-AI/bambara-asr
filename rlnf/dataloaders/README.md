# RLNF Data Loaders

The package provides two PyTorch datasets backed by newline-delimited JSON manifests:

- `AudioDataset` loads `audio_filepath`, resamples audio to the configured rate, and returns mel features for RLNF rollouts.
- `RewardDataset` requires `audio_filepath`, `transcription`, and a human `score` from 0 to 100. It returns mel features, SentencePiece token IDs, and a score normalized to 0–1.

Variable-length features and transcripts are padded by the module-specific collate functions. Dependencies are pinned in `rlnf/requirements.txt`.

```python
from rlnf.dataloaders.reward_dataset import get_dataloaders

preprocessor = {
    "sample_rate": 16000,
    "features": 64,
    "window_size": 0.02,
    "window_stride": 0.01,
    "window": "hann",
    "n_fft": 512,
    "normalize": "per_feature",
    "dither": 1e-5,
    "frame_splicing": 1,
    "stft_conv": False,
}

train_loader, test_loader = get_dataloaders(
    "train.jsonl",
    "test.jsonl",
    "tokenizer.model",
    preprocessor_config=preprocessor,
    batch_size=4,
    num_workers=0,
)
```

The code assumes mono-compatible audio input and uses a fixed tokenizer pad ID. Validate shapes, tokenizer IDs, and score normalization before using a new dataset.
