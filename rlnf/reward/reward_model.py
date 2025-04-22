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
import os
import tempfile
import zipfile
import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from reward_dataset import TOKENIZER_PAD_ID

def masked_mean_pooling(outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute mean pooling over time dimension, masking padded positions.

    Args:
        outputs: Tensor of shape [B, T, D]
        lengths: LongTensor of shape [B] with actual lengths

    Returns:
        pooled: Tensor of shape [B, D]
    """
    _, max_len, dim = outputs.size()
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(2).expand(-1, -1, dim)
    outputs = outputs * mask.float()
    summed = outputs.sum(dim=1)
    lengths = lengths.unsqueeze(1).float().clamp(min=1)
    return summed / lengths


class RewardModel(nn.Module):
    """
    Reward model that predicts a human assigned score in [0,1] from audio and text.
    Basically a regression model with two encoders (audio and text) and a regression head.
    
    Architecture:
      - Audio encoder: Conv1d stack + AdaptiveAvgPool1d
      - Text encoder: Embedding + BiLSTM + masked mean pooling
      - Regression head: MLP ending in Sigmoid
    """
    def __init__(
        self,
        n_mel: int,
        vocab_size: int,
        embed_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        audio_conv_channels: int = 128,
        audio_conv_layers: int = 3,
        head_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        # save config for loading
        self.config = {
            'n_mel': n_mel,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'lstm_hidden': lstm_hidden,
            'lstm_layers': lstm_layers,
            'audio_conv_channels': audio_conv_channels,
            'audio_conv_layers': audio_conv_layers,
            'head_hidden': head_hidden,
            'dropout': dropout,
        }

        # Audio encoder
        convs = []
        in_ch = n_mel
        for _ in range(audio_conv_layers):
            convs += [
                nn.Conv1d(in_ch, audio_conv_channels, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(audio_conv_channels),
                nn.ReLU(inplace=True)
            ]
            in_ch = audio_conv_channels
        self.audio_encoder = nn.Sequential(*convs, nn.AdaptiveAvgPool1d(1))

        # Text encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=TOKENIZER_PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Regression head
        combined_dim = audio_conv_channels + (2 * lstm_hidden)
        self.head = nn.Sequential(
            nn.Linear(combined_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor, # Keep this parameter for consistency
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio: [B, N_mel, T]
            audio_lengths: [B] (unused in conv encoder)
            text: [B, max_seq_len]
            text_lengths: [B]

        Returns:
            preds: [B] scores in [0,1]
        """
        # Audio path
        # input to conv: [B, C, T]
        audio_enc = self.audio_encoder(audio).squeeze(-1)  # [B, audio_conv_channels]

        # Text path
        emb = self.embedding(text)  # [B, L, D]
        packed = pack_padded_sequence(emb, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, L, 2*H]
        text_enc = masked_mean_pooling(out, text_lengths)  # [B, 2*H]

        # Combine
        combined = torch.cat([audio_enc, text_enc], dim=1)  # [B, combined_dim]
        preds = self.head(combined).squeeze(-1)  # [B]
        return preds

    def save(self, save_path: str):
        """
        Save model config and state_dict into a single archive file.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            # write state and config
            state_path = os.path.join(tmpdir, 'pytorch_model.pt')
            torch.save(self.state_dict(), state_path)
            config_path = os.path.join(tmpdir, 'config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            # create zip
            with zipfile.ZipFile(save_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                z.write(state_path, arcname='pytorch_model.bin')
                z.write(config_path, arcname='config.json')

    @classmethod
    def from_pretrained(cls, archive_path: str, device: torch.device = None) -> 'RewardModel':
        """
        Load a RewardModel from a .ckpt archive produced by .save() method.

        Args:
            archive_path: Path to .ckpt containing 'config.json' and 'pytorch_model.bin'
            device: torch.device (defaults to cpu or cuda)

        Returns:
            RewardModel instance in eval mode
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(tmpdir)
            # load config
            cfg_path = os.path.join(tmpdir, 'config.json')
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            # instantiate and load
            model = cls(**cfg)
            state_path = os.path.join(tmpdir, 'pytorch_model.bin')
            state = torch.load(state_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            # print configuration
            print("Loaded RewardModel with config:", json.dumps(cfg, indent=2))
            return model
