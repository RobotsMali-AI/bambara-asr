import numpy as np
import torchaudio
import torch
from transformers import (
    ProcessorMixin,
    T5Tokenizer,
)

from .reward_feature_extraction import RewardFeatureExtractor
from typing import Union, List

class RewardModelProcessor(ProcessorMixin):
    
    """
    Processor for RewardModel that handles audio and text preprocessing.
        - compute log-Mel spectrograms manually
      
    Attributes:
        tokenizer: T5Tokenizer
        audio_fn: function to convert audio to features.
        processor: Hugging Face processor.
    """
    
    
    feature_extractor_class = "RewardFeatureExtractor"
    tokenizer_class = "T5Tokenizer"

    def __init__(self, feature_extractor : RewardFeatureExtractor, tokenizer : T5Tokenizer):
        
        super().__init__(feature_extractor, tokenizer)
         
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        
        self.sample_rate = self.feature_extractor.model.preprocessor._sample_rate
    
        self.audio_fn = self._nemo_audio_pipeline


    def _load_waveform(self, wav_input: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Load a waveform from a file path, numpy array, or torch tensor.
        Always returns a mono float32 torch tensor.

        Args:
            wav_input (str | np.ndarray | torch.Tensor): Input audio data or file path.

        Returns:
            torch.Tensor: Waveform tensor (mono, float32).
        """
        if isinstance(wav_input, str):
            wav, sr = torchaudio.load(wav_input)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        elif isinstance(wav_input, np.ndarray):
            wav = torch.from_numpy(wav_input)
        elif isinstance(wav_input, torch.Tensor):
            wav = wav_input
        else:
            raise TypeError(f"Unsupported type: {type(wav_input)}")

        # Convert stereo to mono if needed
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        return wav.float()
    
    def _nemo_audio_pipeline(self, audios):
        """
        Return tensors [B, T] and their length.
        """
        if not isinstance(audios, (list, tuple)):
            audios = [audios]

        waveforms = []
        lengths = []
        for wav_input in audios:
            wav = self._load_waveform(wav_input)
            waveforms.append(wav)
            lengths.append(wav.shape[-1])

        # pad  waveforms to max
        padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    
        lengths = torch.tensor(lengths, dtype=torch.long)
        return padded, lengths


    def __call__(self, audios: list, texts: list):
        
        """
        Process a batch of audio files + text strings
        Returns a dict containing:
          - audio features
          - audio lengths
          - tokenized text (input_ids, attention_mask)
          - nemo_audio and nemo_audio_length : for nemo model
        """
        

        audio_feats, audio_len = self.audio_fn(audios)
        
        audio_batch, audio_batch_len = self.feature_extractor(audio_feats, audio_len)
        
        
        text_batch = self.tokenizer(texts, padding=True, return_tensors="pt", return_attention_mask=True)
        
        #device = "cuda" if torch.cuda.is_available() else "cpu"
    
        return {
            "audio": audio_batch , #.to(self.feature_extractor.model.device),
            "audio_len": audio_batch_len , #.to(self.feature_extractor.model.device),
            "text": text_batch["input_ids"],
            "text_attention_mask": text_batch["attention_mask"],
            "_audio" : audio_feats
        }
        
