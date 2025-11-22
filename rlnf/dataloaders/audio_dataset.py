"""
Copyright 2025 RobotsMali AI4D Lab.

Licensed under the MIT License; you may not use this file except in compliance with the License.  
You may obtain a copy of the License at:

https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.
"""
import json
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

class AudioDataset(Dataset):
    """
    PyTorch Dataset for loading audio, text, and human-score samples
    for a reward model.

    Each item is a tuple: (audio_feat, text_ids, score)
      - audio_feat: torch.Tensor, shape [N_mel, T]
      - text_ids: List[int], token IDs of transcription
      - score: float, normalized between 0 and 1
    """

    def __init__(
        self,
        manifest_path: str,
        preprocessor_config: dict,
        audio_transform: Optional[Callable] = None,
    ):
        """
        Args:
            manifest_path (str): Path to the .jsonl manifest file.
            tokenizer_model_path (str): Path to SentencePiece .model file.
            preprocessor (torch.nn.Module): Pretrained QuartzNet audio preprocessor.
            sample_rate (int, optional): If provided, resample audio to this rate.
            audio_transform (Callable, optional): Additional audio transforms.
        """
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preprocessor_config)
        self.sample_rate = self.preprocessor._sample_rate
        self.audio_transform = audio_transform

        # Read manifest into memory
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                # Ensure required keys
                assert "audio_filepath" in entry, f"Manifest line missing keys: {entry}"
                self.samples.append(entry)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        record = self.samples[idx]

        # ----- Audio preprocessing -----
        waveform, sr = torchaudio.load(record["audio_filepath"])
        # Optional resampling
        if self.sample_rate is not None and sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate

        length_tensor = torch.tensor([waveform.size(1)], dtype=torch.long)

        # Preprocess the raw audio sequences -> [1, N_mel, T]
        feats, _ = self.preprocessor(input_signal=waveform, length=length_tensor)

        # Remove batch dim -> [N_mel, T]
        audio_feat = feats.squeeze(0)

        # Additional audio transforms
        if self.audio_transform is not None:
            audio_feat = self.audio_transform(audio_feat)

        return audio_feat, length_tensor

def collate_fn(batch):
    """
    Collate function to batch variable-length audio and text data.

    Args:
        batch: List of tuples (audio_feat, text_ids, score).

    Returns:
        dict with:
            audio_batch: Tensor [B, N_mel, max_T]
            audio_lengths: Tensor [B]
            text_batch: Tensor [B, max_seq_len]
            text_lengths: Tensor [B]
            score_batch: Tensor [B]
    """
    # Unzip batch
    audio_feats, length_tensors = zip(*batch)

    # Audio padding
    mel_bins = audio_feats[0].size(0)
    lengths = [length_tensor.item() for length_tensor in length_tensors]
    max_length = max(lengths)
    audio_batch = torch.zeros(len(batch), mel_bins, max_length)
    for i, feat in enumerate(audio_feats):
        audio_batch[i, :, :feat.size(1)] = feat
    audio_lengths = torch.tensor(lengths, dtype=torch.long)

    return {
        "audio_batch": audio_batch,
        "audio_lengths": audio_lengths
    }


def get_audio_loader(
    train_manifest: str,
    preprocessor_config: dict,
    batch_size: int,
    audio_transform: Optional[Callable] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Utility to create DataLoaders for training and testing.

    Args:
        train_manifest (str): Path to training manifest .jsonl.
        test_manifest (str): Path to testing manifest .jsonl.
        tokenizer_model_path (str): Path to SentencePiece .model file.
        preprocessor_conifg (Dict): Pretrained QuartzNet preprocessor.
        batch_size (int): Batch size.
        sample_rate (int, optional): Desired audio sample rate.
        audio_transform (Callable, optional): Additional audio transforms.
        num_workers (int): Number of worker processes.

    Returns:
        train_loader, test_loader: DataLoader objects.
    """
    train_ds = AudioDataset(
        train_manifest,
        preprocessor_config,
        audio_transform,
    )

    audio_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return audio_loader
