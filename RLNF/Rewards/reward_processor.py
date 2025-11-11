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
        
        self.sample_rate = self.feature_extractor.sampling_rate
    
        self.audio_fn = self._default_audio_pipeline


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

    def _default_audio_pipeline(
        self,
        audios: Union[
            str,
            np.ndarray,
            torch.Tensor,
            List[str],
            List[np.ndarray],
            List[torch.Tensor],
        ]
    ):
        """
        Converts one or multiple waveforms (path, numpy array, or tensor)
        into log-Mel spectrograms and pads them to the same length.

        Returns:
            (padded_mels, lengths)
            - padded_mels: torch.Tensor of shape [B, n_mel, T]
            - lengths: torch.Tensor of shape [B] with original time lengths

        Examples >>
        >>> proc = RewardModelProcessor()
        >>> mel, len_ = proc._default_audio_pipeline("audio.wav")
        >>> mel.shape
        torch.Size([1, 80, T])

        >>> mel, len_ = proc._default_audio_pipeline(["a.wav", "b.wav"])
        >>> mel.shape
        torch.Size([2, 80, Tmax])

        >>> x = np.random.randn(16000)
        >>> mel, len_ = proc._default_audio_pipeline(x)
        >>> mel.shape
        torch.Size([1, 80, T])
        """

        # Ensure input is a list for unified processing
        if not isinstance(audios, (list, tuple)):
            audios = [audios]
            
        feats = []
        for wav_input in audios:
            wav = self._load_waveform(wav_input)
            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().numpy()
            else:
                wav = np.asarray(wav, dtype=np.float32).squeeze()
            feats.append(wav)

            

        return feats


    def __call__(self, audios: list, texts: list, truncation = False, 
                 return_attention_mask = True, return_tensors = "pt", do_normalize = True, 
                 padding = True, sampling_rate: int = 16_000):
        """
        Process a batch of audio files + text strings
        Returns a dict containing:
          - audio features
          - audio lengths
          - tokenized text (input_ids, attention_mask)
          - nemo_audio and nemo_audio_length : for nemo model
        """
        
        #if self.use_pre == "default":
        audio_feats = self.audio_fn(audios)
        
        audio_batch = self.feature_extractor(audio_feats, truncation=truncation, 
                                       return_attention_mask=return_attention_mask, 
                                       return_tensors=return_tensors,padding=padding,
                                       do_normalize=do_normalize, sampling_rate=self.sample_rate)
        
        
        nemo_audio, nemo_audio_len = self._nemo_audio_pipeline(audios)
        
        text_batch = self.tokenizer(texts, padding=True, return_tensors="pt", return_attention_mask=True)
        return {
            "audio": audio_batch["input_features"],
            "audio_attention_mask": audio_batch["attention_mask"],
            "text": text_batch["input_ids"],
            "text_attention_mask": text_batch["attention_mask"],
            "nemo_audio": nemo_audio,
            "nemo_audio_len" : nemo_audio_len
        }
        
