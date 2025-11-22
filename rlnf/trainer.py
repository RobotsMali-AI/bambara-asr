from typing import Dict
import json
import contextlib

import torch
from torch.utils.data import DataLoader
from sentencepiece import SentencePieceProcessor

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
import wandb

from rlnf.dataloaders.audio_dataset import AudioDataset, collate_fn
from rlnf.ppo.rollout import collect_batch
from rlnf.ppo.optimizer import PPOOptimizer
from rlnf.reward.reward_model import RewardModel
from rlnf.ppo.critic_network import CriticModel
from rlnf.dataloaders.reward_dataset import TOKENIZER_PAD_ID


class RLNFTrainer:
    """
    Trainer for RLNF (PPO on CTC sequence log-prob).
    """

    def __init__(
        self,
        asr_model: EncDecCTCModel | EncDecCTCModelBPE,
        reward_model: RewardModel,
        critic_model: CriticModel,
        train_manifest: str,
        val_manifest: str,
        sp_tokenizer: SentencePieceProcessor,
        audio_preprocessor_config: Dict,
        device: torch.device,
        wandb_logging: bool = True,
        wandb_project: str = "Bambara-RLNF",
        run_name: str = "test-run1",
        batch_size: int = 16,
        epochs: int = 3,
        K_updates: int = 4,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-4,
        clip_eps: float = 0.2,
        val_every: int = 200,
        num_workers: int = 2,
        pin_memory: bool = True,
        amp: bool = False,  # enable if you switch to GPU
    ):
        self.asr_model = asr_model
        self.reward_model = reward_model
        self.critic_model = critic_model
        self.device = device
        self.sp_tokenizer = sp_tokenizer
        self.epochs = epochs
        self.val_manifest = val_manifest
        self.val_every = val_every

        # Data
        train_ds = AudioDataset(
            manifest_path=train_manifest, preprocessor_config=audio_preprocessor_config
        )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_ds = AudioDataset(
            manifest_path=val_manifest, preprocessor_config=audio_preprocessor_config
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.current_epoch = None

        # PPO
        self.ppo = PPOOptimizer(
            actor=asr_model,
            critic=critic_model,
            clip_eps=clip_eps,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            K_updates=K_updates,
            device=device,
            amp=amp,
        )

        # WandB
        self._use_wandb = wandb_logging
        if self._use_wandb:
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "K_updates": K_updates,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "clip_eps": clip_eps,
                    "amp": amp,
                },
            )

    def train(self):
        global_step = 0
        try:
            for epoch in range(self.epochs):
                self.current_epoch = epoch
                cfg = getattr(wandb, "config", {}) if self._use_wandb else {}
                print(f"Starting epoch {epoch+1}/{self.epochs} | config: {cfg}")

                for batch in self.train_loader:
                    # === On-policy rollout (no grad, eval mode inside) ===
                    batch_dict = collect_batch(
                        batch=batch,
                        asr_model=self.ppo.actor,
                        reward_model=self.reward_model,
                        critic=self.ppo.critic,
                        sp_tokenizer=self.sp_tokenizer,
                        device=self.device,
                        pad_id=TOKENIZER_PAD_ID,
                    )

                    # === PPO updates ===
                    stats = self.ppo.update(batch_dict)

                    # Logging
                    if self._use_wandb and wandb.run is not None:
                        wandb.log(
                            {
                                "train/actor_loss": stats["actor_loss"],
                                "train/critic_loss": stats["critic_loss"],
                                "train/value_mean": stats["mean_value"],
                                # PPO diagnostics
                                "train/adv_mean": stats.get("adv_mean", float("nan")),
                                "train/adv_std": stats.get("adv_std", float("nan")),
                                "train/ratio_mean": stats.get("ratio_mean", float("nan")),
                                "train/frac_clipped": stats.get("frac_clipped", float("nan")),
                                "train/logp_old_mean": stats.get("logp_old_mean", float("nan")),
                                "train/logp_new_mean": stats.get("logp_new_mean", float("nan")),
                                "train/reward_mean": stats.get("reward_mean", float("nan")),
                                "train/value_hat_mean": stats.get("V_hat_mean", float("nan")),
                            },
                            step=global_step,
                        )
                    else:
                        print(
                            f"Step {global_step}: "
                            f"actor={stats['actor_loss']:.4f} | "
                            f"critic={stats['critic_loss']:.4f} | "
                            f"V̄={stats['mean_value']:.4f} | "
                            f"advμ={stats.get('adv_mean', float('nan')):.3f} "
                            f"advσ={stats.get('adv_std', float('nan')):.3f} | "
                            f"ratioμ={stats.get('ratio_mean', float('nan')):.3f} "
                            f"clip%={100*stats.get('frac_clipped', float('nan')):.1f}"
                        )

                    global_step += 1

                    if (self.val_every > 0) and (global_step % self.val_every == 0):
                        self.validate(global_step)

                # end-of-epoch validation
                self.validate(global_step, end_of_epoch=True)

        except KeyboardInterrupt:
            print("Interrupted — saving checkpoints and closing WandB.")
        except MemoryError:
            print("Memory error — saving checkpoints and closing WandB.")
        finally:
            # Save final artifacts
            try:
                self.ppo.actor.save_to("actor_final.nemo")
            except Exception as e:
                print("Could not save actor:", e)
            try:
                self.critic_model.save("critic_final.ct")
            except Exception as e:
                print("Could not save critic:", e)
            with contextlib.suppress(Exception):
                if self._use_wandb:
                    wandb.finish()

    def validate(self, step: int, end_of_epoch: bool = False):
        # Eval
        self.ppo.actor.eval()
        self.ppo.critic.eval()
        with torch.no_grad():
            # WER/CER from NeMo convenience (keep batch_size small to avoid spikes)
            hyps = self.ppo.actor.transcribe(self.val_manifest, batch_size=8)
            self.ppo.actor.eval()
            hyp_texts = [h.text for h in hyps]

            # Load refs
            refs = []
            with open(self.val_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    refs.append(data.get("text", ""))

            wer = word_error_rate(hyp_texts, refs)
            cer = word_error_rate(hyp_texts, refs, use_cer=True)

            # Reward/Value on val set (no PPO updates here)
            mean_rewards, mean_values = [], []
            for batch in self.val_loader:
                val_dict = collect_batch(
                    batch=batch,
                    asr_model=self.ppo.actor,
                    reward_model=self.reward_model,
                    critic=self.ppo.critic,
                    sp_tokenizer=self.sp_tokenizer,
                    device=self.device,
                    pad_id=TOKENIZER_PAD_ID,
                )
                mean_rewards.append(float(val_dict["reward"].mean()))
                mean_values.append(float(val_dict["values"].mean()))

            mean_reward = sum(mean_rewards) / max(1, len(mean_rewards))
            mean_value = sum(mean_values) / max(1, len(mean_values))

        to_log = {
            "val/wer": wer,
            "val/cer": cer,
            "val/reward": mean_reward,
            "val/value": mean_value,
        }
        if end_of_epoch:
            to_log["epoch"] = self.current_epoch

        if self._use_wandb and wandb.run is not None:
            wandb.log(to_log, step=step)
        else:
            print(
                f"[VAL {step}] WER: {wer:.4f} | CER: {cer:.4f} | "
                f"Reward: {mean_reward:.4f} | Value: {mean_value:.4f}"
            )

        # back to train mode
        self.ppo.actor.train()
        self.ppo.critic.train()
