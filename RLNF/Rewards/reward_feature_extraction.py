
from transformers import WhisperFeatureExtractor, AutoFeatureExtractor

class RewardFeatureExtractor(WhisperFeatureExtractor) :
        
     def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        chunk_length: int = 30,
        padding_value: float = 0.0,
        dither: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ):  
         super().__init__(feature_size=feature_size, 
                         sampling_rate = sampling_rate,
                         chunk_length = chunk_length, n_fft = n_fft, 
                         padding_value = padding_value, 
                         dither=dither, 
                         hop_length=hop_length,
                        return_attention_mask = return_attention_mask, 
                        **kwargs)
        
        
AutoFeatureExtractor.register("RewardFeatureExtractor", RewardFeatureExtractor)

        
        