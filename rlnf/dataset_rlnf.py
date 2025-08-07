from typing import List, Optional, Callable

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio

from tokenizer_rlnf import RLNFSentencePieceTokenizer
from feature_extractor_rlnf import RLNFFeatureExtractor
from processor_rlnf import RLNFProcessor
from config_rlnf import RLNFConfig

TOKENIZER_PAD_ID = 1

class RLNFDataset(Dataset):
    """
    PyTorch Dataset for audio-text-score examples using pre-loaded arrays.

    Expects dataset with columns:
      - 'array': 1D numpy array of audio samples
      - 'sample_rate': int sample rate
      - 'transcription': str
      - 'score': numeric (0-100)

    Returns:
      - 'audio_feats': Tensor [n_mels, T]
      - 'input_ids': Tensor [L]
      - 'score': float (0.0-1.0)
    """

    def __init__(
        self,
        hf_dataset,
        config: str,
        tokenizer_model_path: str,
        audio_transform: Optional[Callable] = None,
    ):
        self.samples = hf_dataset
        cfg = RLNFConfig.from_pretrained(config)
        tokenizer = RLNFSentencePieceTokenizer(tokenizer_model_path)
        feature_extractor = RLNFFeatureExtractor(cfg)
        self.processor = RLNFProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        self.audio_transform = audio_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        rec = self.samples[idx]
        audio = rec["audio"]
        wav = torch.tensor(audio['array'], dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        sr = audio['sampling_rate']
        if self.audio_transform:
            wav = self.audio_transform(wav)
        # Resample if needed
        target_sr = self.processor.feature_extractor.preprocessor._sample_rate
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            sr = target_sr
        audio_input = wav.squeeze(0)
        # Process audio and text, return tensors
        outputs = self.processor(audio=audio_input, text=rec['transcription'], return_tensors='pt')
        feats = outputs["input_features"].squeeze(0)
        ids = outputs["input_ids"].squeeze(0)
        
        if ids.numel() == 0 :
            ids = torch.tensor([TOKENIZER_PAD_ID])
        
        score = float(rec.get('score', 0.0)) / 100.0
        return {'audio_feats': feats, 'input_ids': ids, 'score': score}

    @staticmethod
    def collate(batch: List[dict]) -> dict:
        feats = [b['audio_feats'] for b in batch]
        ids = [b['input_ids'] for b in batch]
        scores = [b['score'] for b in batch]
        n_mels = feats[0].size(0)
        lengths = [f.size(1) for f in feats]
        max_len = max(lengths)
        audio_batch = torch.zeros(len(batch), n_mels, max_len)
        for i, f in enumerate(feats):
            audio_batch[i, :, :f.size(1)] = f
        audio_lengths = torch.tensor(lengths, dtype=torch.long)
        text_lengths = torch.tensor([t.size(0) for t in ids], dtype=torch.long)
        text_batch = pad_sequence(ids, batch_first=True, padding_value=TOKENIZER_PAD_ID)
        score_batch = torch.tensor(scores, dtype=torch.float)
        return {
            'audio_feats': audio_batch,
            'audio_lengths': audio_lengths,
            'input_ids': text_batch,
            'text_lengths': text_lengths,
            'score': score_batch,
        }
