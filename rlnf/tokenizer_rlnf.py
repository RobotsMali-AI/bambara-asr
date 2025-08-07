
from typing import List, Optional, Dict, Tuple
import os

from transformers import PreTrainedTokenizer
from sentencepiece import SentencePieceProcessor
import shutil


class RLNFSentencePieceTokenizer(PreTrainedTokenizer):
    """
    HuggingFace-compatible tokenizer wrapping a SentencePiece model.
    """

    def __init__(
        self,
        model_file: str,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        **kwargs
    ):
        # Load SentencePiece model before superclass init
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"SentencePiece model not found: {model_file}")
        self.sp_model = SentencePieceProcessor()
        self.sp_model.Load(model_file)
        self.model_file = model_file 

        # Initialize base tokenizer (calls get_vocab internally)
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs
        )

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp_model.IdToPiece(index)

    @property
    def vocab_size(self) -> int:
        return self.sp_model.GetPieceSize()

    def get_vocab(self) -> Dict[str, int]:
        return {self.sp_model.IdToPiece(i): i for i in range(self.vocab_size)}

    def encode(self, text: str, **kwargs) -> List[int]:
        return self.sp_model.EncodeAsIds(text)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.sp_model.DecodeIds(token_ids)

   
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        prefix = filename_prefix or "sentencepiece"
        save_path = os.path.join(save_directory, f"{prefix}.model")
        shutil.copyfile(self.model_file, save_path)
        return (save_path,)

