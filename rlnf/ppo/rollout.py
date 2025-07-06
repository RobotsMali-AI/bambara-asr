import torch
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
from sentencepiece import SentencePieceProcessor
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from rlnf.reward.reward_model import RewardModel
from rlnf.ppo.critic_network import CriticModel

def decode_batch(
    log_probs: torch.Tensor,
    enc_len: torch.Tensor,
    asr_model: EncDecCTCModel | EncDecCTCModelBPE,
    return_hypotheses: bool = False,
) -> List[str]:
    """  
    Decode a batch of ASR log-probabilities into text strings.  
  
    Officially supports EnDecCTC, and EncDecCTCBPE models.  
  
    Args:  
        log_probs: Tensor of shape [B, T, V], log-probabilities.  
        enc_len: Tensor of shape [B], lengths of each sequence in log_probs.  
        asr_model: NeMo ASR model with .decoding attribute.  
  
    Returns:  
        List of decoded strings, length B.  
    """
    hyps = None
    # Check if the model has a CTC decoding attribute
    if hasattr(asr_model.decoding, 'ctc_decoder_predictions_tensor'):
        hyps = asr_model.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs,
            decoder_lengths=enc_len,
            return_hypotheses=return_hypotheses
        )
    else:
        raise AttributeError("Only CTC models are supported for the moment")

    # The above function returns a list of hypotheses
    texts = [h.text for h in hyps] if isinstance(hyps, list) else hyps
    return texts

def compute_mask(enc_len: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Utility to create a boolean mask [B, T] from enc_len.
    """
    idx = torch.arange(max_len, device=enc_len.device).unsqueeze(0)
    mask = idx < enc_len.unsqueeze(1)
    return mask


def collect_batch(
    asr_model: EncDecCTCModel | EncDecCTCModelBPE,
    reward_model: RewardModel,
    critic: CriticModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    sp_tokenizer: SentencePieceProcessor,
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Run one on-policy rollout over a single mini-batch for PPO.

    Args:
        nemo_model: NeMo ASR model.
        reward_model: Reward predictor (eval). Inputs: audio_feat, text_ids.
        critic: Value network (eval): same inputs as reward_model.
        batch: output of collate_fn with keys:
            'audio_batch': Tensor [B, C, T]
            'audio_lengths': Tensor [B]
        device: compute device for model inference.
        sp_tokenizer: SentencePieceProcessor for encoding texts.
        pad_id: padding token ID for text sequences.

    Returns:
        dict containing:
            audio_batch: [B, C, T] on CPU
            audio_lengths: [B] on CPU
            text_batch: [B, L] on CPU
            text_lengths: [B] on CPU
            log_probs_old: [B] on device
            mask: [B, T] on CPU
            greedy_ids: [B, T] on CPU
            reward: [B] on CPU
            values: [B] on CPU
    """
    # Move inputs to device
    audio = batch['audio_batch'].to(device)
    audio_lens = batch['audio_lengths'].to(device)

    # Switch all the models to eval mode
    asr_model.eval()
    reward_model.eval(); critic.eval()

    with torch.no_grad():
        # assume Nemo forward returns (log_probs, enc_len, ...)
        log_probs3d, enc_len, greedy_ids = asr_model.forward(processed_signal=audio, processed_signal_length=audio_lens)

        # Decode ASR hypotheses to text
        transcriptions = decode_batch(log_probs3d, enc_len, asr_model)

        # Get the mask for this Batch
        max_len = log_probs3d.size(1)
        mask = compute_mask(enc_len=enc_len, max_len=max_len)
        # Move mask on device
        mask = mask.to(device).float()

        # Transfrom log_probs, utterance level logits
        log_probs = log_probs3d.gather(2, greedy_ids.unsqueeze(-1)).squeeze(-1)
        log_probs = log_probs * mask

        log_probs_old = log_probs.sum(dim=1).detach()  # keep on device

        # Free up memory by deleting 3D log_probs, log_probs and mask
        del log_probs3d, log_probs

        # Tokenize with SentencePiece
        tokenized = [sp_tokenizer.EncodeAsIds(t) for t in transcriptions]

        # Ensure text_ids for empty strings will not be empty, instead they will contain the pad_id
        for text_ids in tokenized:
            if len(text_ids) == 0:
                text_ids.append(pad_id)

        # pad to max length
        text_tensors = [torch.tensor(ids, dtype=torch.long) for ids in tokenized]
        text_batch = pad_sequence(text_tensors, batch_first=True, padding_value=pad_id)
        text_lengths = torch.tensor([t.size(0) for t in text_tensors], dtype=torch.long)

        # Compute rewards and values
        reward = reward_model(audio, audio_lens, text_batch.to(device), text_lengths.to(device))
        values = critic(audio)

        reward = reward.cpu()
        values = values.cpu()

    # Prepare output dict
    return {
        'audio_batch': audio.cpu(),
        'audio_lengths': audio_lens.cpu(),
        'text_batch': text_batch.cpu(),
        'text_lengths': text_lengths.cpu(),
        'log_probs_old': log_probs_old,
        'greedy_ids': greedy_ids.cpu(),
        'mask': mask.cpu(),
        'reward': reward,
        'values': values,
    }
