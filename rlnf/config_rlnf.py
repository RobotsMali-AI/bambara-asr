from transformers import PretrainedConfig


class RLNFConfig(PretrainedConfig):
    """
    Configuration for the RewardModel, including both audio preprocessor
    and model architecture parameters.
    """
    model_type = "RewardModelRLNF"

    def __init__(
        self,
        # Preprocessor parameters
        normalize: str = "per_feature",
        window_size: float = 0.02,
        sample_rate: int = 16000,
        window_stride: float = 0.01,
        window: str = "hann",
        features: int = 64,
        n_fft: int = 512,
        frame_splicing: int = 1,
        dither: float = 1e-05,
        stft_conv: bool = False,
        
        # Model architecture parameters
        n_mel: int = 64,
        vocab_size: int = 1024,
        embed_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        audio_conv_channels: int = 128,
        audio_conv_layers: int = 3,
        head_hidden: int = 256,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Audio preprocessor config
        self.preprocessor_config = {
            'normalize': normalize,
            'window_size': window_size,
            'sample_rate': sample_rate,
            'window_stride': window_stride,
            'window': window,
            'features': features,
            'n_fft': n_fft,
            'frame_splicing': frame_splicing,
            'dither': dither,
            'stft_conv': stft_conv,
        }

        # Model-specific config
        self.model_config = {
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

__all__ = ["RLNFConfig"]