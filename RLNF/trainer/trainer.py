from typing import Dict
import json
import contextlib
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    Trainer RLNF compatible SINGLE GPU & MULTI-GPU (DDP) & Best checkpoints saving.
    """

    def __init__(
        self,
        asr_model: EncDecCTCModel | EncDecCTCModelBPE,
        reward_model: RewardModel,
        critic_model: CriticModel,
        dataset: datasets,
        processor: RewardModelProcessor,
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
        amp: bool = False,

        # ===== BEST CHECKPOINT =====
        save_dir: str = "checkpoints",
        save_best_by: str = "val/wer",
        save_best_mode: str = "min",
    ):
        # ================= DDP =================
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.is_main = self.rank == 0

        self.device = device
        self.processor = processor
        self.reward_model = reward_model
        self.epochs = epochs
        self.val_every = val_every
        self.current_epoch = None
        
        self.batch_size = batch_size

    
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_best_by = save_best_by
        self.save_best_mode = save_best_mode
        self.best_val = float("inf") if save_best_mode == "min" else -float("inf")

      
        self.tb_writer = SummaryWriter(log_dir=f"tb_logs/{run_name}") if self.is_main else None

        self._use_wandb = wandb_logging and self.is_main
        if self._use_wandb:
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=locals(),
            )


        collate_fn = RewardDataCollator(processor=processor, augment=False)

        train_sampler = DistributedSampler(dataset["train"]) if self.is_distributed else None

        self.train_loader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.val_loader = DataLoader(
            dataset["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    
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


    def train(self):
        global_step = 0

        try:
            for epoch in range(self.epochs):
                self.current_epoch = epoch

                if self.is_distributed:
                    self.train_loader.sampler.set_epoch(epoch)

                if self.is_main:
                    print(f"Epoch {epoch+1}/{self.epochs}")

                pbar = tqdm(
                    self.train_loader,
                    leave=False,
                    disable=not self.is_main,
                    desc=f"Epoch {epoch+1}/{self.epochs}"
                )

                for batch in pbar:
                    actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor
                    critic = self.ppo.critic.module if self.is_distributed else self.ppo.critic

                    batch_dict = collect_batch(
                        batch=batch,
                        asr_model=actor,
                        reward_model=self.reward_model,
                        critic=critic,
                        processor=self.processor,
                        device=self.device,
                    )

                    stats = self.ppo.update(batch_dict)

                    if self.is_main:
                        if self._use_wandb:
                            wandb.log({f"train/{k}": v for k, v in stats.items()}, step=global_step)

                        for k, v in stats.items():
                            self.tb_writer.add_scalar(f"train/{k}", v, global_step)

                        #pbar.set_postfix(
                        #    actor_loss=f"{stats['actor_loss']:.3f}",
                        #    reward=f"{stats['reward_mean']:.3f}",
                        #)
                        
                        pbar.set_postfix({
                            "actor_loss": f"{stats['actor_loss']:.3f}",
                            "critic_loss": f"{stats['critic_loss']:.3f}",
                            "V": f"{stats['mean_value']:.3f}",
                            "adv": f"{stats['adv_mean']:.3f}",
                            "clip%": f"{100*stats['frac_clipped']:.1f}",
                            "reward" : f"{stats['reward_mean']:.3f}",
                            "ratio_mean": f"{stats['ratio_mean']:.3f}",
                            "frac_clipped": f"{stats['frac_clipped']:.3f}"
                        
                    })

                    global_step += 1

                    if self.is_main and self.val_every > 0 and global_step % self.val_every == 0:
                        self.validate(global_step)

                if self.is_main:
                    self.validate(global_step, end_of_epoch=True)

        finally:
            if self.is_main:
                self.save_final()
                if self._use_wandb:
                    wandb.finish()
                self.tb_writer.close()

   
    def validate(self, step: int, end_of_epoch: bool = False):
        actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor
        critic = self.ppo.critic.module if self.is_distributed else self.ppo.critic

        actor.eval()
        critic.eval()

        wers, cers, rewards, values = [], [], [], []

        with torch.no_grad():
            
            pbar_val = tqdm(
                    self.val_loader,
                    leave=False,
                    disable=not self.is_main,
                    desc=f"Validation at step {step}"
            )
            for batch in pbar_val:
                
                actor.spec_augmentation = None
                actor.sample_rate = 16000
                actor.preprocessor.featurizer.to(self.device)
                
                
                
                audio = [aud for aud in batch["_audio"]] 

                hyps = actor.transcribe(audio, batch_size=self.batch_size)
                hyp_texts = [h.text for h in hyps]

                refs = self.processor.tokenizer.batch_decode(
                    batch["text"], skip_special_tokens=True
                )

                wers.append(word_error_rate(hyp_texts, refs))
                cers.append(word_error_rate(hyp_texts, refs, use_cer=True))

                val_dict = collect_batch(
                    batch=batch,
                    asr_model=actor,
                    reward_model=self.reward_model,
                    critic=critic,
                    processor=self.processor,
                    device=self.device,
                )

                rewards.append(val_dict["reward"].mean().item())
                values.append(val_dict["values"].mean().item())
                
            pbar_val.set_postfix({
                    "WER": f"{sum(wers) / len(wers):.4f}",
                    "CER": f"{sum(cers) / len(cers):.4f}",
                    "Reward": f"{ sum(rewards) / len(rewards):.4f}",
                    "Value": f"{sum(values) / len(values):.4f}"
                })

        to_log = {
            "val/wer": sum(wers) / len(wers),
            "val/cer": sum(cers) / len(cers),
            "val/reward": sum(rewards) / len(rewards),
            "val/value": sum(values) / len(values),
        }

        if end_of_epoch:
            to_log["epoch"] = self.current_epoch

        cur = to_log[self.save_best_by]
        is_better = (
            cur < self.best_val if self.save_best_mode == "min"
            else cur > self.best_val
        )

        if is_better:
            print(f"New best {self.save_best_by}: {cur:.4f}")
            self.best_val = cur
            self.save_best(step)

        if self._use_wandb:
            wandb.log(to_log, step=step)

        for k, v in to_log.items():
            self.tb_writer.add_scalar(k, v, step)

        actor.train()
        critic.train()

    def save_best(self, step: int):
        actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor
        critic = self.ppo.critic.module if self.is_distributed else self.ppo.critic

        actor.save_to(os.path.join(self.save_dir, f"best_step{step}_actor.nemo"))
        torch.save(critic.state_dict(), os.path.join(self.save_dir, f"best_step{step}_critic.pt"))

    def save_final(self):
        actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor
        critic = self.ppo.critic.module if self.is_distributed else self.ppo.critic

        actor.save_to("actor_final.nemo")
        torch.save(critic.state_dict(), "critic_final.pt")
