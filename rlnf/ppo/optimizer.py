from typing import Dict
import torch
import torch.nn.functional as F
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from rlnf.ppo.loss import PPOLoss
from rlnf.ppo.critic_network import CriticModel


class PPOOptimizer:
    """
    Core PPO optimizer handling actor and critic updates.
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
        device: torch.device = torch.device('cpu'),
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.K_updates = K_updates
        self.entropy_coef = entropy_coef

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.ppo_loss = PPOLoss(clip_eps)

        self.actor.train()
        self.critic.train()

    @staticmethod
    def compute_advantages(
        reward: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantage = reward - values, then z-score it and detach.
        """
        adv = reward - values
        mean = adv.mean()
        std = adv.std(unbiased=False) + 1e-8
        adv = (adv - mean) / std
        return adv.detach()

    def criticise(
        self,
        audio: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run critic in eval mode and return value predictions [B] on self.device.
        """
        self.critic.eval()
        with torch.no_grad():
            vals = self.critic(audio.to(self.device), lengths.to(self.device))
        return vals.squeeze(-1)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform K PPO updates on the provided batch.

        Args:
            batch: dict containing tensors (some on CPU):
                'audio_batch': [B, C, T]
                'audio_lengths': [B]
                'text_batch': [B, L]
                'text_lengths': [B]
                'greedy_ids': [B, T]
                'enc_len': [B]
                'mask': [B, T]
                'log_probs_old': [B]
                'reward': [B]
                'values': [B]

        Returns:
            Stats dict with floats: actor_loss, critic_loss, mean_value
        """
        # Move to device
        audio = batch['audio_batch'].to(self.device)
        a_len = batch['audio_lengths'].to(self.device)
        greedy = batch['greedy_ids'].to(self.device)
        mask = batch['mask'].to(self.device).float()

        logp_old = batch['log_probs_old'].to(self.device)
        reward = batch['reward'].to(self.device)
        values = batch['values'].to(self.device)

        # advantages
        adv = self.compute_advantages(reward, values)

        # Ensure actor and critic are in training mode
        self.actor.train()
        self.critic.train()
        print("Before attemping forward pass, actor and critic are in training mode.")

        for _ in range(self.K_updates):
            # Actor/ASR model forward
            out = self.actor(processed_signal=audio, processed_signal_length=a_len)
            logp_3d = out[0]

            # gather log-probs at greedy ids
            # logp_3d: [B, T, V], greedy: [B, T]
            lp = logp_3d.gather(2, greedy.unsqueeze(-1)).squeeze(-1)
            lp = lp * mask
            logp_new = lp.sum(dim=1)

            # Critic forward
            V_hat = self.critic(audio).squeeze(-1)

            # Losses
            loss_actor = self.ppo_loss(logp_old, logp_new, adv)
            if self.entropy_coef > 0:
                entropy = -(logp_3d.exp() * logp_3d).sum(dim=(1,2)).mean()
                loss_actor = loss_actor - self.entropy_coef * entropy

            loss_critic = F.mse_loss(V_hat, reward)

            # Backprop
            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()

            (loss_actor + 0.5 * loss_critic).backward()
            self.opt_actor.step()
            self.opt_critic.step()

        return {
            'actor_loss':  loss_actor.detach().cpu().item(),
            'critic_loss': loss_critic.detach().cpu().item(),
            'mean_value':  V_hat.detach().mean().cpu().item(),
        }
