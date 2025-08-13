# rlnf/ppo/optimizer.py
from typing import Dict
import torch
import torch.nn.functional as F
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from rlnf.ppo.loss import PPOLoss
from rlnf.ppo.critic_network import CriticModel
from rlnf.ppo.rollout import _blank_index, _seq_logprob_ctc, _ensure_log_softmax

class PPOOptimizer:
    """
    Core PPO optimizer handling actor and critic updates with CTC sequence log-probs.
    """
    def __init__(
        self,
        actor: EncDecCTCModel | EncDecCTCModelBPE,
        critic: CriticModel,
        clip_eps: float = 0.2,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-4,
        K_updates: int = 4,
        entropy_coef: float = 0.0,
        device: torch.device = torch.device("cpu"),
        amp: bool = False,
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.K_updates = K_updates
        self.entropy_coef = entropy_coef
        self.amp = amp

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.ppo_loss = PPOLoss(clip_eps)

        self.actor.train()
        self.critic.train()
        self._blank_idx = _blank_index(self.actor)

        self._scaler = torch.amp.GradScaler(enabled=amp)

    @staticmethod
    def _normalize_adv(adv: torch.Tensor) -> torch.Tensor:
        mean = adv.mean()
        std = adv.std(unbiased=False)
        return (adv - mean) / (std + 1e-8)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform K PPO updates on the provided batch.

        Expected batch keys (CPU tensors unless noted):
            audio_batch: [B, C, T]
            audio_lengths: [B]
            targets: [B, Lmax]
            target_lengths: [B]
            input_lengths: [B]
            log_probs_old: [B]
            reward: [B]
            values: [B]
        """
        # Move once per epoch to device
        audio = batch["audio_batch"].to(self.device, non_blocking=True)
        a_len = batch["audio_lengths"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        t_len = batch["target_lengths"].to(self.device, non_blocking=True)
        in_len = batch["input_lengths"].to(self.device, non_blocking=True)

        logp_old = batch["log_probs_old"].to(self.device, non_blocking=True).detach()
        reward = batch["reward"].to(self.device, non_blocking=True)
        values_old = batch["values"].to(self.device, non_blocking=True)

        # Old advantages (on-policy): use stored old values for stability
        adv = self._normalize_adv(reward - values_old).detach()

        self.actor.train()
        self.critic.train()

        for _ in range(self.K_updates):
            self.opt_actor.zero_grad(set_to_none=True)
            self.opt_critic.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=self.amp):
                # Actor forward -> current log-probs [B,T,V]
                out = self.actor(processed_signal=audio, processed_signal_length=a_len)
                if isinstance(out, (list, tuple)):
                    logits_or_logp3d = out[0]
                    in_len_new = out[1]
                    # Some NeMo models may slightly change time resolution; prefer fresh in_len
                    in_len_use = in_len_new
                else:
                    raise RuntimeError("Unexpected ASR forward() return; expected (log_probs, enc_len, ...).")
                
                # === ensure log-probs for both decoding & CTCLoss ===
                logp3d_new = _ensure_log_softmax(logits_or_logp3d)

                logp_new = _seq_logprob_ctc(
                    logp3d_new, in_len_use, targets, t_len, self._blank_idx
                )  # [B]

                # Critic forward
                V_hat = self.critic(audio).squeeze(-1)  # [B]

                # PPO actor loss (uses old log-prob & normalized advantage)
                loss_actor = self.ppo_loss(logp_old, logp_new, adv)

                # Optional entropy bonus (sequence entropy is non-trivial; we use token-level entropy proxy)
                if self.entropy_coef > 0:
                    ent = -(logp3d_new.exp() * logp3d_new).sum(dim=(1, 2)).mean()
                    loss_actor = loss_actor - self.entropy_coef * ent

                # Critic loss vs actual reward (you can also bootstrap against reward targets)
                loss_critic = F.mse_loss(V_hat, reward)

                # Diagnostics
                with torch.no_grad():
                    ratio = torch.exp(logp_new - logp_old)
                    frac_clipped = ((ratio > 1.0 + self.ppo_loss.clip_eps) | (ratio < 1.0 - self.ppo_loss.clip_eps)).float().mean()
                    diag = {
                        "adv_mean": float(adv.mean().cpu()),
                        "adv_std": float(adv.std(unbiased=False).cpu()),
                        "ratio_mean": float(ratio.mean().cpu()),
                        "frac_clipped": float(frac_clipped.cpu()),
                        "logp_old_mean": float(logp_old.mean().cpu()),
                        "logp_new_mean": float(logp_new.mean().cpu()),
                        "reward_mean": float(reward.mean().cpu()),
                        "V_hat_mean": float(V_hat.mean().cpu()),
                    }

            # Backprop (AMP-aware)
            if self._scaler.is_enabled():
                self._scaler.scale(loss_actor + 0.5 * loss_critic).backward()
                self._scaler.step(self.opt_actor)
                self._scaler.step(self.opt_critic)
                self._scaler.update()
            else:
                (loss_actor + 0.5 * loss_critic).backward()
                self.opt_actor.step()
                self.opt_critic.step()

        # Return last-iter losses + key diagnostics
        out_stats = {
            "actor_loss": float(loss_actor.detach().cpu().item()),
            "critic_loss": float(loss_critic.detach().cpu().item()),
            "mean_value": float(V_hat.detach().mean().cpu().item()),
        }
        out_stats.update(diag)
        return out_stats
