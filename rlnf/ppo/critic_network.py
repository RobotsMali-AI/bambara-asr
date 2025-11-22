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

class CriticModel(nn.Module):
    """
    Critic model that predicts the expected value of the best on-policy action from audio/states.
    Basically a regression model.
    
    Architecture:
      - Audio encoder: Conv1d stack + AdaptiveAvgPool1d
      - Regression head: MLP ending in Sigmoid
    """
    def __init__(
        self,
        n_mel: int,
        audio_conv_channels: int = 128,
        audio_conv_layers: int = 3,
        head_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        # save config for loading
        self.config = {
            'n_mel': n_mel,
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

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(audio_conv_channels, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio: [B, N_mel, T]

        Returns:
            preds: [B] scores in [0,1]
        """
        # Audio path
        # input to conv: [B, C, T]
        audio_enc = self.audio_encoder(audio).squeeze(-1)  # [B, audio_conv_channels]

        # Feed the regression head with the encoded audio
        preds = self.head(audio_enc).squeeze(-1)  # [B]
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
    def from_pretrained(cls, archive_path: str, device: torch.device = None) -> 'CriticModel':
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
            print("Loaded CriticModel with config:", json.dumps(cfg, indent=2))
            return model
