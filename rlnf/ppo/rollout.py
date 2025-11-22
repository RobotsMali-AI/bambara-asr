# rlnf/ppo/rollout.py
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from torch.nn.utils.rnn import pad_sequence
from sentencepiece import SentencePieceProcessor
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from rlnf.reward.reward_model import RewardModel
from rlnf.ppo.critic_network import CriticModel


@torch.no_grad()
def decode_batch(
    log_probs: torch.Tensor,
    enc_len: torch.Tensor,
    asr_model: EncDecCTCModel | EncDecCTCModelBPE,
    return_hypotheses: bool = False,
) -> List[str]:
    """
    Decode a batch of CTC log-probs [B, T, V] to text using NeMo's decoder.
    """
    if hasattr(asr_model.decoding, "ctc_decoder_predictions_tensor"):
        hyps = asr_model.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs, decoder_lengths=enc_len, return_hypotheses=return_hypotheses
        )
    else:
        raise AttributeError("Only CTC models are supported for now.")
    return [h.text for h in hyps] if isinstance(hyps, list) else hyps


def _blank_index(asr_model: EncDecCTCModel) -> int:
    return len(asr_model.decoder.vocabulary)  # QuartzNet-style: blank is last index

def _ensure_log_softmax(logits_btv: torch.Tensor) -> torch.Tensor:
    # If already log-probs: logsumexp ≈ 0
    lse = torch.logsumexp(logits_btv.detach(), dim=-1)
    if torch.allclose(lse, torch.zeros_like(lse), atol=1e-3, rtol=1e-3):
        return logits_btv
    return logits_btv.log_softmax(dim=-1)

def _encode_texts_for_ctc(
    asr_model: EncDecCTCModel | EncDecCTCModelBPE,
    texts: List[str],
) -> Tuple[List[List[int]], List[int]]:
    """
    Map decoded strings back to label indices (no blanks), consistent with actor's output vocab.
    - For BPE models: use asr_model.tokenizer.text_to_ids()
    - For char-level CTC: map each character via decoder.vocabulary
    Returns (list_of_id_lists, list_of_lengths)
    """
    ids_list: List[List[int]] = []
    lens_list: List[int] = []

    # Prefer tokenizer if present (BPE)
    tok = getattr(asr_model, "tokenizer", None)
    if tok is not None:
        for t in texts:
            ids = tok.text_to_ids(t) if hasattr(tok, "text_to_ids") else tok.encode(t)
            # ensure non-empty for CTC
            if len(ids) == 0:
                # pick a safe non-blank symbol; take index 1 if blank=0 else 0
                blank = _blank_index(asr_model)
                fallback = 1 if blank == 0 else 0
                ids = [fallback]
            ids_list.append(ids)
            lens_list.append(len(ids))
        return ids_list, lens_list

    # Char-level fallback via vocabulary
    if hasattr(asr_model, "decoder") and hasattr(asr_model.decoder, "vocabulary"):
        vocab: List[str] = asr_model.decoder.vocabulary
        sym2idx = {s: i for i, s in enumerate(vocab)}
        for t in texts:
            # direct per-character mapping
            ids = []
            for ch in t:
                if ch in sym2idx:
                    ids.append(sym2idx[ch])
                # If char not in vocab, you could skip or map to a fallback; here we skip.
            if len(ids) == 0:
                blank = _blank_index(asr_model)
                fallback = 1 if blank == 0 else 0
                ids = [fallback]
            ids_list.append(ids)
            lens_list.append(len(ids))
        return ids_list, lens_list

    raise RuntimeError("Could not derive CTC targets: no tokenizer and no vocabulary found.")


def _pack_targets_1d(targets_padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert a padded [B, Lmax] int tensor + [B] lengths into 1D concat targets for CTCLoss.
    """
    parts = []
    for i in range(targets_padded.size(0)):
        li = int(lengths[i].item())
        parts.append(targets_padded[i, :li])
    return torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long, device=targets_padded.device)


def _seq_logprob_ctc(
    log_probs_btv: torch.Tensor,  # [B, T, V] log-probs
    input_lengths_b: torch.Tensor,  # [B] lengths in time-steps
    targets_padded_bl: torch.Tensor,  # [B, Lmax] label ids
    target_lengths_b: torch.Tensor,  # [B] target lengths
    blank_idx: int,
) -> torch.Tensor:
    """
    Compute per-sample sequence log-prob: log P(y|x) using CTC forward-backward.
    """
    # CTCLoss expects [T, B, V] and 1D concatenated targets
    log_probs_tbv = log_probs_btv.permute(1, 0, 2).float()
    flat_targets_1d = _pack_targets_1d(targets_padded_bl, target_lengths_b).to(log_probs_btv.device)

    ctc = nn.CTCLoss(blank=blank_idx, reduction="none", zero_infinity=True)
    nll = ctc(log_probs_tbv, flat_targets_1d, input_lengths_b.int(), target_lengths_b.int())  # [B]
    return -nll  # [B], sequence log-prob


@torch.no_grad()
def collect_batch(
    asr_model: EncDecCTCModel | EncDecCTCModelBPE,
    reward_model: RewardModel,
    critic: CriticModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    sp_tokenizer: SentencePieceProcessor,  # reward model tokenizer
    pad_id: int,                            # reward model pad id
) -> Dict[str, torch.Tensor | List[str]]:
    """
    One on-policy rollout over a single mini-batch for PPO.
    Stores ONLY CPU tensors needed for PPO; avoids time-major tensors.
    """
    # Move inputs to device
    audio = batch["audio_batch"].to(device)
    audio_lens = batch["audio_lengths"].to(device)

    # Eval/no-grad rollout
    asr_model.eval()
    reward_model.eval()
    critic.eval()

    with torch.no_grad():
        # Forward actor -> CTC log-probs and encoded lengths
        # NeMo CTC typically returns (log_probs[B,T,V], enc_len[B], greedy_ids[B,T]) or similar tuple
        out = asr_model(processed_signal=audio, processed_signal_length=audio_lens)
        # Be tolerant to output tuple structure
        if isinstance(out, (list, tuple)):
            logits_or_logp3d = out[0]
            enc_len = out[1]
        else:
            raise RuntimeError("Unexpected ASR forward() return; expected (log_probs, enc_len, ...).")
        
        # === ensure log-probs for both decoding & CTCLoss ===
        log_probs3d = _ensure_log_softmax(logits_or_logp3d)

        # Decode to text (for reward model & diagnostics)
        transcriptions = decode_batch(log_probs3d, enc_len, asr_model)

        # === Reward model tokenization (kept as in your original design) ===
        rm_ids = [sp_tokenizer.EncodeAsIds(t) for t in transcriptions]
        for ids in rm_ids:
            if len(ids) == 0:
                ids.append(pad_id)  # keep non-empty for batching
        rm_tensors = [torch.tensor(ids, dtype=torch.long) for ids in rm_ids]
        rm_text_batch = pad_sequence(rm_tensors, batch_first=True, padding_value=pad_id)  # [B, L_rm]
        rm_text_lens = torch.tensor([len(ids) for ids in rm_ids], dtype=torch.long)

        # Compute reward and values
        reward = reward_model(audio, audio_lens, rm_text_batch.to(device), rm_text_lens.to(device)).cpu()  # [B]
        values = critic(audio).cpu().squeeze(-1)  # [B]

        # === PPO old log-prob via CTC forward-backward ===
        # Map decoded text back to actor labels (no blanks)
        tgt_lists, tgt_lens_list = _encode_texts_for_ctc(asr_model, transcriptions)
        tgt_tensors = [torch.tensor(x, dtype=torch.long) for x in tgt_lists]
        tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=0).to(device)  # [B, Lmax] (pad won't be used)
        tgt_lens = torch.tensor(tgt_lens_list, dtype=torch.long, device=device)              # [B]

        blank_idx = _blank_index(asr_model)
        logp_old = _seq_logprob_ctc(log_probs3d, enc_len.to(device), tgt_padded, tgt_lens, blank_idx).detach()  # [B]

    # Return CPU payload only; keep raw text too (tiny memory footprint)
    return {
        "audio_batch": audio.cpu(),
        "audio_lengths": audio_lens.cpu(),
        "targets": tgt_padded.cpu(),          # [B, Lmax] (for PPO update)
        "target_lengths": tgt_lens.cpu(),     # [B]
        "input_lengths": enc_len.cpu(),       # [B] (time steps at CTC head)
        "log_probs_old": logp_old.cpu(),      # [B]
        "reward": reward.cpu(),               # [B]
        "values": values.cpu(),               # [B]
        "texts": transcriptions,              # keep raw strings for reward/debug
        # reward model text batch is not needed after reward is computed; not stored
    }
