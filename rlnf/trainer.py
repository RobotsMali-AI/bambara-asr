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
    A trainer class for Reinforcement Learning from Nouhoum Feedback (RLNF) for Automatic Speech Recognition (ASR) models.
    This class implements training and validation loops using Proximal Policy Optimization (PPO) to optimize the ASR model
    based on rewards provided by a reward model.
    Attributes:
        asr_model (EncDecCTCModel | EncDecCTCModelBPE): The ASR model to be trained.
        reward_model (RewardModel): The reward model used to compute rewards for the ASR outputs.
        critic_model (CriticModel): The critic model used in PPO for value estimation.
        train_manifest (str): Path to the training manifest file containing audio file paths and metadata.
        val_manifest (str): Path to the validation manifest file containing audio file paths and metadata.
        sp_tokenizer (SentencePieceProcessor): SentencePiece tokenizer for tokenizing text data.
        audio_preprocessor_config (Dict): Configuration for the audio preprocessing pipeline.
        device (torch.device): The device (CPU or GPU) to run the training on.
        run_name (str): Name of the training run for logging purposes. Defaults to "test-run1".
        batch_size (int): Batch size for training and validation. Defaults to 16.
        epochs (int): Number of training epochs. Defaults to 3.
        K_updates (int): Number of PPO updates per batch. Defaults to 4.
        actor_lr (float): Learning rate for the ASR model (actor). Defaults to 1e-5.
        critic_lr (float): Learning rate for the critic model. Defaults to 1e-4.
        clip_eps (float): Clipping epsilon for PPO. Defaults to 0.2.
        val_every (int): Frequency (in steps) of validation during training. Defaults to 200.
        num_workers (int): Number of workers for data loading. Defaults to 2.
    Methods:
        train():
            Executes the training loop for the specified number of epochs. Logs training metrics and performs
            validation at regular intervals.
        validate(step: int, end_of_epoch: bool = False):
            Performs validation on the validation dataset. Computes metrics such as Word Error Rate (WER),
            Character Error Rate (CER), mean reward, and mean value. Logs validation metrics.
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
        wandb_project: str = 'Bambara-RLNF',
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
    ):
        # save models and config
        self.asr_model = asr_model
        self.reward_model = reward_model
        self.critic_model = critic_model
        self.device = device
        self.sp_tokenizer = sp_tokenizer
        self.epochs = epochs
        self.val_manifest = val_manifest
        self.val_every = val_every

        # Training DataLoader
        train_ds = AudioDataset(manifest_path=train_manifest,
                                preprocessor_config=audio_preprocessor_config)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        # Validation DataLoader
        val_ds = AudioDataset(manifest_path=val_manifest,
                              preprocessor_config=audio_preprocessor_config)
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.current_epoch = None

        # PPO Optimizer
        self.ppo = PPOOptimizer(
            actor=asr_model,
            critic=critic_model,
            clip_eps=clip_eps,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            K_updates=K_updates,
            device=device,
        )

        if wandb_logging:
            # W&B init
            wandb.init(
                project='Bambara-RLNF',
                name=run_name,
                config={
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'K_updates': K_updates,
                    'actor_lr': actor_lr,
                    'critic_lr': critic_lr,
                    'clip_eps': clip_eps,
                }
            )

    def train(self):
        global_step = 0
        try:
            for epoch in range(self.epochs):
                self.current_epoch = epoch
                print("Starting epoch:", epoch + 1, "of", self.epochs, "with config:",
                      wandb.config)
                for batch in self.train_loader:
                    batch_dict = collect_batch(
                        batch=batch,
                        asr_model=self.ppo.actor,
                        reward_model=self.reward_model,
                        critic=self.ppo.critic,
                        sp_tokenizer=self.sp_tokenizer,
                        device=self.device,
                        pad_id=TOKENIZER_PAD_ID
                    )

                    stats = self.ppo.update(batch_dict)
                    if wandb.run is not None:
                        # Log training stats to WandB
                        wandb.log({
                            'train/actor_loss': stats['actor_loss'],
                            'train/critic_loss': stats['critic_loss'],
                            'train/value_mean': stats['mean_value'],
                        }, step=global_step)
                    else:
                        print(f"Step {global_step}: Actor Loss: {stats['actor_loss']}, "
                              f"Critic Loss: {stats['critic_loss']}, "
                              f"Mean Value: {stats['mean_value']}")

                    global_step += 1

                    if global_step % self.val_every == 0:
                        self.validate(global_step)

                # end of epoch validation
                self.validate(global_step, end_of_epoch=True)
        except KeyboardInterrupt:
            print("Interrupted by user — saving checkpoints and closing WandB.")
        except MemoryError:
            print("Memory error encountered — saving checkpoints and closing WandB.")

        finally:
            # save final models
            self.ppo.actor.save_to('actor_final.nemo')
            self.critic_model.save('critic_final.ct')
            with contextlib.suppress(Exception):
                wandb.finish()

    def validate(self, step: int, end_of_epoch: bool = False):
        # Put actor and critic in eval mode
        self.ppo.actor.eval()
        self.ppo.critic.eval()
        with torch.no_grad():
            hyps = self.ppo.actor.transcribe(self.val_manifest, batch_size=16)
            # Ensure the actor stays in eval mode after transcribing
            self.ppo.actor.eval()

            # The above line returns a list of Hypothesis objects.
            hyps = [hyp.text for hyp in hyps]
            # Load references from the validation manifest
            refs = []
            with open(self.val_manifest, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    refs.append(data.get('text', ''))
            # Calculate WER
            wer = word_error_rate(hyps, refs)
            # Calculate CER
            cer = word_error_rate(hyps, refs, use_cer=True)

            mean_rewards = []
            mean_values = []
            for batch in self.val_loader:
                val_dict = collect_batch(
                        batch=batch,
                        asr_model=self.ppo.actor,
                        reward_model=self.reward_model,
                        critic=self.ppo.critic,
                        sp_tokenizer=self.sp_tokenizer,
                        device=self.device,
                        pad_id=TOKENIZER_PAD_ID
                    )
                mean_reward = float(val_dict['reward'].mean())
                mean_value = float(val_dict['values'].mean())
                mean_rewards.append(mean_reward)
                mean_values.append(mean_value)
            mean_reward = sum(mean_rewards) / len(mean_rewards)
            mean_value = sum(mean_values) / len(mean_values)

        # Log validation metrics
        to_log = {
            'val/wer': wer,
            'val/cer': cer,
            'val/reward': mean_reward,
            'val/value': mean_value,
        }
        if end_of_epoch:
            to_log['epoch'] = self.current_epoch

        if wandb.run is not None:
            wandb.log(to_log, step=step)
        else:
            print(f"Validation at step {step}: WER: {wer}, CER: {cer}, "
                  f"Mean Reward: {mean_reward}, Mean Value: {mean_value}")

        self.ppo.actor.train()
