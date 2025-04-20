"""
Copyright 2025 RobotsMali AI4D Lab.

Licensed under the MIT License; you may not use this file except in compliance with the License.  
You may obtain a copy of the License at:

https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.
"""
import sys
# from utils package
from utils.preprocessing import check_and_convert_audio_channels
from utils.helpers import load_config, enable_bn_se
from utils.wandb import MyWandbLogger as WandbLogger
# Lightning imports
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# Import Weight and Biases
import wandb
# Nemo  imports
import nemo.collections.asr as nemo_asr
import nemo.lightning as nl
from nemo.lightning import AutoResume

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("Usage: python fine_tuning_quartznet_20M_ctc.py <config_path>")

    # Load YAML configuration
    config_path = sys.argv[1]
    config = load_config(config_path)

    print(f"Fine tuning {config.model.name}...\nDownloading the checkpoint")

    # Load QuartzNet model
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=config.model.name)

    # Ensure all audio files have only 1 channel
    check_and_convert_audio_channels(config.data_loaders.train.manifest_filepath)
    check_and_convert_audio_channels(config.data_loaders.valid.manifest_filepath)
    check_and_convert_audio_channels(config.data_loaders.test.manifest_filepath)

    # The new vocabulary for the model (These are the characters its gonna output now)
    new_vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
             'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
             'w', 'x', 'y', 'z', ' ', "'", '-', 'ŋ', 'ɔ', 'ɛ', 'ɲ', 'ɓ', 'ɾ']

    # Do not change the vocabulary that would reset the decoder's weights
    # model.change_vocabulary(new_vocabulary=new_vocab)

    # Freeze encoder if specified
    if config.training.freeze_encoder:
        model.encoder.freeze()
        model.encoder.apply(enable_bn_se)
        print("Model encoder has been frozen")
    else:
        model.encoder.unfreeze()
        print("Model encoder has been unfrozen")

    # Setup optimization
    model.setup_optimization(optim_config=config.optim)

    # Update the labels of the dataloaders
    config.data_loaders.train.labels = new_vocab
    config.data_loaders.test.labels = new_vocab
    config.data_loaders.valid.labels = new_vocab

    # Setup training, validation, and test data
    model.setup_training_data(train_data_config=config.data_loaders.train)
    model.setup_validation_data(val_data_config=config.data_loaders.valid)
    model.setup_test_data(test_data_config=config.data_loaders.test)

    # Increase SpectAugment for larger models to prevent overfitting
    model.cfg.spec_augment.rect_freq = 50
    model.cfg.spec_augment.rect_time = 120
    model.cfg.spec_augment.rect_masks = 10 # Increased

    print(model.cfg.spec_augment)
    model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

    # Setup logger and callbacks
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        name=config.wandb.name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.training.checkpoint_dir,
        save_weights_only=True,
        save_last=True,
        monitor="val_wer",
        mode="min",
        save_top_k=config.training.save_top_k
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_wer",
        mode="min",
        patience=config.training.patience,
        verbose=True
    )

    # Define trainer
    trainer = nl.Trainer(
        devices=1,
        accelerator='gpu',
        precision=config.training.precision,
        max_epochs=config.training.epochs,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        logger=wandb_logger,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Auto resume policy
    resume = AutoResume(
        resume_if_exists=config.training.resume_if_exists,
	resume_from_directory=config.training.checkpoint_dir,
        resume_ignore_no_checkpoint=config.training.resume_ignore_no_checkpoint
    )
    resume.setup(trainer)

    # Start training
    try:
        trainer.fit(model)
    except Exception:
        print("Training interrupted, finishing logging...")
        wandb.finish()

    # Save trained model
    model.save_to(config.training.save_model_path)

    # Run testing if test set is available
    if hasattr(model.cfg, 'test_ds') and model.cfg.test_ds.manifest_filepath is not None:
        if model.prepare_test(trainer):
            trainer.test(model)

    print(f"Fine-tuning completed successfully...\nNeMo model saved to: {config.training.save_model_path}")
