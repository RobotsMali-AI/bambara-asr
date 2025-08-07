import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers.processing_utils import ProcessorMixin

from feature_extractor_rlnf import RLNFFeatureExtractor
from tokenizer_rlnf import RLNFSentencePieceTokenizer
from transformers import AutoFeatureExtractor, AutoTokenizer


AutoFeatureExtractor.register("RLNFFeatureExtractor", RLNFFeatureExtractor)
AutoTokenizer.register("RLNFSentencePieceTokenizer", RLNFSentencePieceTokenizer)


class RLNFProcessor(ProcessorMixin):
    """
    Combines RLNF tokenizer and Nemo feature extractor.
    """
    
    feature_extractor_class = "RLNFFeatureExtractor"
    tokenizer_class = "RLNFSentencePieceTokenizer"


    def __init__(self,
                 tokenizer: RLNFSentencePieceTokenizer,
                 feature_extractor: RLNFFeatureExtractor,
                 **kwargs):
        super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __call__(self,
                 audio: Union[str,
                              List[str],
                              np.ndarray,
                              List[np.ndarray],
                              torch.Tensor,
                              List[torch.Tensor]
                              ],
                 text: Optional[Union[str, list]] = None,
                 return_tensors: Optional[str] = None) -> Dict[str, Any]:
        
        # Audio -> log-mel
        feats = self.feature_extractor(audio)
        output: Dict[str, Any] = feats
        if text is not None:
            enc = self.tokenizer(text, padding=True, return_tensors=return_tensors)
            output.update(enc)
        return output