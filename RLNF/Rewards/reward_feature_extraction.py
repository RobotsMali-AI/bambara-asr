
from typing import Tuple
from torch import Tensor
import torch
from transformers import AutoFeatureExtractor
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE

class RewardFeatureExtractor :
        
     def __init__(self, asr_model : EncDecCTCModelBPE | EncDecCTCModel): 
          
          self.model = asr_model
          
          
     def __call__(self, audios: Tensor, audios_lens : Tensor) -> Tuple[Tensor, Tensor]:
          
          """
          audios: (B, T)
          audios_lens: (B,)
          return: (features, features_lens)
          """
          
          #device = self.model.device #"cuda" if torch.cuda.is_available() else "cpu"
          device = audios.device
          self.model.preprocessor.featurizer.fb = (
               self.model.preprocessor.featurizer.fb.to(device)
          )
                    
          return self.model.preprocessor(input_signal=audios, length=audios_lens)
          

      
AutoFeatureExtractor.register("RewardFeatureExtractor", RewardFeatureExtractor)

        
        