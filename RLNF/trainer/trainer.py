from typing import Dict
import json
import contextlib

import torch
from torch.utils.data import DataLoader
from sentencepiece import SentencePieceProcessor

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
import wandb

import datasets
from ..dataloaders.reward_dataset import RewardDataCollator
from ..Rewards.reward_processor import RewardModelProcessor

from ..utils.rollout import collect_batch
from ..optimizer.optimizer import PPOOptimizer
from ..PPO.critic_network import CriticModel

from ..Rewards.reward_model import RewardModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class RLNFTrainer:
    """
    Trainer for RLNF (PPO on CTC sequence log-prob).
    """

    def __init__(
        self,
        asr_model: EncDecCTCModel | EncDecCTCModelBPE,
        reward_model: RewardModel,
        critic_model: CriticModel,
        dataset : datasets,
        processor : RewardModelProcessor,
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
        self.processor = processor
        self.epochs = epochs
        self.val_every = val_every
        
        self.tb_writer = SummaryWriter(log_dir=f"tb_logs/{run_name}")

        
        train_ds = dataset["train"]
        val_ds = dataset["test"]
        
        collate_fn = RewardDataCollator(processor=processor, augment=False)
      
        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
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

                # tqdm pour la boucle sur le DataLoader
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
                for batch in pbar:
                    # === On-policy rollout ===
                    batch_dict = collect_batch(
                        batch=batch,
                        asr_model=self.ppo.actor,
                        reward_model=self.reward_model,
                        critic=self.ppo.critic,
                        processor=self.processor,
                        device=self.device,
                    )

                    # === PPO updates ===
                    stats = self.ppo.update(batch_dict)

                    # === Logging WandB et TensorBoard ===
                    if self._use_wandb and wandb.run is not None:
                        wandb.log(
                            {
                                "train/actor_loss": stats["actor_loss"],
                                "train/critic_loss": stats["critic_loss"],
                                "train/value_mean": stats["mean_value"],
                                "train/adv_mean": stats.get("adv_mean", float("nan")),
                                "train/adv_std": stats.get("adv_std", float("nan")),
                                "train/ratio_mean": stats.get("ratio_mean", float("nan")),
                                "train/frac_clipped": stats.get("frac_clipped", float("nan")),
                                "train/reward_mean": stats.get("reward_mean", float("nan")),
                                "train/value_hat_mean": stats.get("V_hat_mean", float("nan")),
                            },
                            step=global_step,
                        )

                        self.tb_writer.add_scalar("train/actor_loss", stats["actor_loss"], global_step)
                        self.tb_writer.add_scalar("train/critic_loss", stats["critic_loss"], global_step)
                        self.tb_writer.add_scalar("train/value_mean", stats["mean_value"], global_step)
                        self.tb_writer.add_scalar("train/adv_mean", stats.get("adv_mean", float("nan")), global_step)
                        self.tb_writer.add_scalar("train/adv_std", stats.get("adv_std", float("nan")), global_step)
                        self.tb_writer.add_scalar("train/ratio_mean", stats.get("ratio_mean", float("nan")), global_step)
                        self.tb_writer.add_scalar("train/frac_clipped", stats.get("frac_clipped", float("nan")), global_step)
                        self.tb_writer.add_scalar("train/reward_mean", stats.get("reward_mean", float("nan")), global_step)
                        self.tb_writer.add_scalar("train/value_hat_mean", stats.get("V_hat_mean", float("nan")), global_step)

                    # === Mettre à jour la barre tqdm ===
                    pbar.set_postfix({
                        "actor_loss": f"{stats['actor_loss']:.4f}",
                        "critic_loss": f"{stats['critic_loss']:.4f}",
                        "V̄": f"{stats['mean_value']:.4f}",
                        "advμ": f"{stats.get('adv_mean', float('nan')):.3f}",
                        "clip%": f"{100*stats.get('frac_clipped', float('nan')):.1f}",
                        "reward" : f"{stats.get('reward_mean', float('nan')):.4f}"
                        
                    })

                    global_step += 1

                    if (self.val_every > 0) and (global_step % self.val_every == 0):
                        self.validate(global_step)
                        self.tb_writer.flush()

                # Validation en fin d'époque
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
                self.critic_model.save_pretrained("critic_final")
            except Exception as e:
                print("Could not save critic:", e)
            with contextlib.suppress(Exception):
                if self._use_wandb:
                    wandb.finish()
            self.tb_writer.close()


    def validate(self, step: int, end_of_epoch: bool = False):
        # Eval mode
        self.ppo.actor.eval()
        self.ppo.critic.eval()
        
        mean_rewards, mean_values = [], []
        wer, cer = 0.0, 0.0

        # tqdm pour la validation
        pbar_val = tqdm(self.val_loader, desc=f"Validation at step {step}", leave=False)
        
        with torch.no_grad():
            for batch in pbar_val:
                # Transcription batch audio
                
                self.ppo.actor.sample_rate = 16000
                self.ppo.actor.spec_augmentation = None

                
                audio = [aud.cpu() for aud in batch["nemo_audio"]] #
                
                hyps = self.ppo.actor.transcribe(audio, batch_size=8)

                hyp_texts = [h.text for h in hyps]

                # Décodage batch texte de référence
                tokenizer = self.processor.tokenizer
                refs = tokenizer.batch_decode(batch["text"], skip_special_tokens=True)

                # WER/CER
                wer = word_error_rate(hyp_texts, refs)
                cer = word_error_rate(hyp_texts, refs, use_cer=True)
                

                # Reward/Value
                val_dict = collect_batch(
                    batch=batch,
                    asr_model=self.ppo.actor,
                    reward_model=self.reward_model,
                    critic=self.ppo.critic,
                    processor=self.processor,
                    device=self.device,
                )
                mean_rewards.append(float(val_dict["reward"].mean()))
                mean_values.append(float(val_dict["values"].mean()))

                # Mise à jour dynamique de la barre
                pbar_val.set_postfix({
                    "WER": f"{wer:.4f}",
                    "CER": f"{cer:.4f}",
                    "Reward": f"{float(val_dict['reward'].mean()):.4f}",
                    "Value": f"{float(val_dict['values'].mean()):.4f}"
                })

        # Moyennes sur tout le set de validation
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

        # Logging WandB + TensorBoard
        if self._use_wandb and wandb.run is not None:
            wandb.log(to_log, step=step)
            self.tb_writer.add_scalar("val/wer", to_log["val/wer"], step)
            self.tb_writer.add_scalar("val/cer", to_log["val/cer"], step)
            self.tb_writer.add_scalar("val/reward", to_log["val/reward"], step)
            self.tb_writer.add_scalar("val/value", to_log["val/value"], step)
            if "epoch" in to_log:
                self.tb_writer.add_scalar("val/epoch", to_log["epoch"], step)

        # Back to train mode
        self.ppo.actor.train()
        self.ppo.critic.train()
