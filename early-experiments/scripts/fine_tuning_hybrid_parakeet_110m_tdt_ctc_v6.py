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
        raise ValueError("Usage: python fine_tuning_hybrid_parakeet_110m_tdt_ctc.py <config_path>")

    # Load YAML configuration
    config_path = sys.argv[1]
    config = load_config(config_path)

    print(f"Fine tuning {config.model.name}...\nDownloading the checkpoint")

    # Load Parakeet-110M-tdt-ctc model
    model = nemo_asr.models.ASRModel.restore_from(restore_path=config.model.name)

    # pretrained_ctc_decoder = None
    # pretrained_decoder = None

    # if config.training.warm_decoder:
        # Preserve the decoder parameters in case weight matching can be done later to restore them later
        # For the restauration to be possible, the new vocab size of the model should be equal to the vocab size of the pretraining dataset
     #   pretrained_decoder = model.decoder.state_dict()
        # This Hybrid arcihtecture has a CTC decoder as well
      #  pretrained_ctc_decoder = model.ctc_decoder.state_dict()

    # Ensure all audio files have only 1 channel
    check_and_convert_audio_channels(config.data_loaders.train.manifest_filepath)
    check_and_convert_audio_channels(config.data_loaders.valid.manifest_filepath)
    check_and_convert_audio_channels(config.data_loaders.test.manifest_filepath)

    # Continuing another training do not change vocabulary
    # model.change_vocabulary(
    #    new_tokenizer_dir=config.tokenizer.path,
    #    new_tokenizer_type=config.tokenizer.type
    #)

    # Freeze encoder if specified
    if config.training.freeze_encoder:
        model.encoder.freeze()
        model.encoder.apply(enable_bn_se)
        print("Model encoder has been frozen")
    else:
        model.encoder.unfreeze()
        print("Model encoder has been unfrozen")

    #if pretrained_ctc_decoder is not None and pretrained_decoder is not None:
        # Restore preserved model weights if shapes match
     #   if model.decoder.prediction.dec_rnn.lstm.weight_hh_l0.shape == pretrained_decoder['prediction.dec_rnn.lstm.weight_hh_l0'].shape:
     #       model.decoder.load_state_dict(pretrained_decoder)

      #      if not model.decoder.training:
      #          # Ensure the decoder is still in training mode
      #          model.decoder.train()
      #      print("Decoder shapes matched - restored weights from pre-trained model")

      #  if model.ctc_decoder.decoder_layers[0].weight.shape == pretrained_ctc_decoder['decoder_layers.0.weight'].shape:
      #      model.ctc_decoder.load_state_dict(pretrained_ctc_decoder)

      #      if not model.ctc_decoder.training:
                # Ensure the CTC decoder is still in training mode
      #          model.ctc_decoder.train()
      #      print("CTC Decoder shapes matched - restored weights from pre-trained model")

    # Setup optimization
    model.setup_optimization(optim_config=config.optim)

    # Setup training, validation, and test data
    model.setup_training_data(train_data_config=config.data_loaders.train)
    model.setup_validation_data(val_data_config=config.data_loaders.valid)
    model.setup_test_data(test_data_config=config.data_loaders.test)

    # Increase SpectAugment for larger models to prevent overfitting
    model.cfg.spec_augment.freq_masks = 4 # Increase the number of frequency masks
    model.cfg.spec_augment.freq_width = 27
    model.cfg.spec_augment.time_masks = 10
    model.cfg.spec_augment.time_width = 0.1 # Increase time width

    print(model.cfg.spec_augment)
    model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

    # Decrease the impact of the auxilary decoder on the total loss (since it is performing better)
    model.cfg.aux_ctc.ctc_loss_weight = 0.6 # Was set to 0.15 for V5

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
