import torchaudio
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List
from config_rlnf import RLNFConfig

class RLNFFeatureExtractor(SequenceFeatureExtractor):
    """
    Wrapper HF autour de Nemo AudioToMelSpectrogramPreprocessor.
    Expose un extracteur compatible avec les pipelines Transformers :
    - __call__ renvoie dict avec clé 'input_features'
    """

    def __init__(self, config: RLNFConfig, **kwargs):
        # Initialise la classe de base avec les paramètres nécessaires
        super().__init__(
            feature_size=config.model_config["n_mel"],
            sampling_rate=config.preprocessor_config["sample_rate"],
            padding_value=0.0,
            **kwargs
        )
        self.config = config
        # Création du préprocesseur Nemo
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**config.preprocessor_config)

    def to_dict(self) -> dict:
        """
        Sérialisation JSON du feature extractor (sans objets complexes).
        """
        return {
            "feature_extractor_type": self.__class__.__name__,
            **self.config.preprocessor_config
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Reconstruit un extracteur à partir d'un dict JSON.
        """
        data = data.copy()
        data.pop("feature_extractor_type", None)
        return cls(**data)

    def __call__(
        self,
        raw_speech: Union[
            str,
            List[str],
            np.ndarray,
            List[np.ndarray],
            torch.Tensor,
            List[torch.Tensor]
        ],
        return_tensors: str = None,
        **kwargs
    ) -> dict:
        """
        Convertit raw_speech en log-Mel spectrogram et emballe en dict HF.

        Args:
            raw_speech: chemin(s), array(s) ou tensor(s) audio.
            return_tensors: 'pt', 'np' ou 'tf' pour le type de tenseur.

        Returns:
            dict avec clé 'input_features' contenant Tensor [B, n_mels, T], converti si demandé.
        """
        paths: List[str] = []
        arrays: List[np.ndarray] = []
        tensors: List[torch.Tensor] = []

        # 1. Identifie et collecte les signaux
        if isinstance(raw_speech, str):
            paths = [raw_speech]
        elif isinstance(raw_speech, list) and all(isinstance(x, str) for x in raw_speech):
            paths = raw_speech  # type: ignore
        elif isinstance(raw_speech, np.ndarray):
            arrays = [raw_speech]
        elif isinstance(raw_speech, list) and all(isinstance(x, np.ndarray) for x in raw_speech):
            arrays = raw_speech  # type: ignore
        elif isinstance(raw_speech, torch.Tensor):
            tensors = [raw_speech]
        elif isinstance(raw_speech, list) and all(isinstance(x, torch.Tensor) for x in raw_speech):
            tensors = raw_speech  # type: ignore
        else:
            raise ValueError(f"Unsupported raw_speech type: {type(raw_speech)}")

        for path in paths:
            wav, sr = torchaudio.load(path)
            if sr != self.sampling_rate:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(wav)
            # Convertir en mono si nécessaire
            if wav.dim() == 2:
                wav = wav.mean(dim=0)
            tensors.append(wav)

        # 3. Convertit numpy en tensors mono
        for arr in arrays:
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            tensors.append(torch.tensor(arr, dtype=torch.float))

        # 4. Compose le batch
        if not tensors:
            raise ValueError("Aucun signal audio à traiter.")
        # Si un seul vecteur 1D
        if len(tensors) == 1 and tensors[0].dim() == 1:
            batch = tensors[0].unsqueeze(0)
        else:
            lengths = torch.tensor([t.numel() for t in tensors], dtype=torch.long)
            max_len = int(lengths.max().item())
            padded = [F.pad(t, (0, max_len - t.numel())) for t in tensors]
            batch = torch.stack(padded, dim=0)

        # 5. Préparation des longueurs pour Nemo
        lengths = torch.tensor([batch.size(-1)] * batch.size(0), dtype=torch.long)
        # 6. Préprocesseur Nemo -> Mel spectrogram
        mel, _ = self.preprocessor(input_signal=batch, length=lengths)
        

        # 7. Emballage HF
        data = {"input_features": mel}
        return data

__all__ = ["RLNFFeatureExtractor"]
