"""
Copyright 2025 RobotsMali AI4D Lab.

Licensed under the MIT License; you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the License is provided "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""

import os
import json
import tempfile
import zipfile

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from config_rlnf import RLNFConfig
from dataset_rlnf import TOKENIZER_PAD_ID


def masked_mean_pooling(outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute mean pooling over time dimension, masking padded positions.
    """
    _, max_len, dim = outputs.size()
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(2).expand(-1, -1, dim)
    outputs = outputs * mask.float()
    summed = outputs.sum(dim=1)
    lengths = lengths.unsqueeze(1).float().clamp(min=1)
    return summed / lengths


class RewardModelRLNF(PreTrainedModel):
    """
    HuggingFace-compatible Reward Model predicting a score in [0,1].
    Combines audio conv encoder and text BiLSTM encoder with regression head.
    """
    config_class = RLNFConfig
    base_model_prefix = "reward_model"
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config: RLNFConfig):
        super().__init__(config)
        # Audio encoder
        convs = []
        in_ch = config.model_config["n_mel"]
        for _ in range(config.model_config["audio_conv_layers"]):
            convs += [
                nn.Conv1d(in_ch, config.model_config["audio_conv_channels"], kernel_size=5, padding=2),
                nn.BatchNorm1d(config.model_config["audio_conv_channels"]),
                nn.ReLU(inplace=True)
            ]
            in_ch = config.model_config["audio_conv_channels"]
        self.audio_encoder = nn.Sequential(*convs, nn.AdaptiveAvgPool1d(1))

        # Text encoder
        self.embedding = nn.Embedding(
            config.model_config["vocab_size"], config.model_config["embed_dim"], padding_idx=TOKENIZER_PAD_ID
        )
        self.lstm = nn.LSTM(
            config.model_config["embed_dim"],
            config.model_config["lstm_hidden"],
            num_layers=config.model_config["lstm_layers"],
            batch_first=True,
            bidirectional=True,
        )

        # Regression head
        combined_dim = config.model_config["audio_conv_channels"] + 2 * config.model_config["lstm_hidden"]
        self.head = nn.Sequential(
            nn.Linear(combined_dim, config.model_config["head_hidden"]),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model_config["dropout"]),
            nn.Linear(config.model_config["head_hidden"], config.model_config["head_hidden"]),
            nn.ReLU(inplace=True),
            nn.Linear(config.model_config["head_hidden"], 1),
            nn.Sigmoid(),
        )

        # Initialize weights and register config
        self.post_init()

    def forward(
        self,
        audio_feats: torch.FloatTensor,
        audio_lengths: torch.LongTensor,
        input_ids: torch.LongTensor,
        text_lengths: torch.LongTensor,
        labels: torch.FloatTensor = None
    ) -> ModelOutput:
        # Audio path
        audio_enc = self.audio_encoder(audio_feats).squeeze(-1)  # [B, C]

        # Text path
        emb = self.embedding(input_ids)
        
        packed = pack_padded_sequence(
            emb, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        text_enc = masked_mean_pooling(out, text_lengths)

        # Combine
        combined = torch.cat([audio_enc, text_enc], dim=1)
        preds = self.head(combined).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(preds, labels)

        return ModelOutput(loss=loss, logits=preds)

    def save_pretrained(self, save_directory: str):
        """
        Save model and config to a HuggingFace-formatted directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Save config
        self.config.save_pretrained(save_directory)
        # Save weights
        super().save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        Load model from HuggingFace hub or local directory.
        """
        config = RLNFConfig.from_pretrained(pretrained_model_name_or_path)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            *args,
            **kwargs
        )
        return model

