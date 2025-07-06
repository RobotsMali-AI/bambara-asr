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
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from rlnf.dataloaders.reward_dataset import get_dataloaders
from rlnf.reward.reward_model import RewardModel
from rlnf.reward.train_utils import fit, evaluate
from sentencepiece import SentencePieceProcessor


def load_tokenizer(model_path: str) -> SentencePieceProcessor:
    sp = SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def main():
    parser = argparse.ArgumentParser(description="Train a reward model on audio+text data")
    parser.add_argument('--train_manifest', type=str, required=True)
    parser.add_argument('--test_manifest', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to SentencePiece .model file')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./training_archives')
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--scheduler_step_size', type=int, default=30)
    parser.add_argument('--scheduler_gamma', type=float, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--plot', action='store_true',
                        help='Plot predictions vs. targets after eval')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print config
    print("Configuration:")
    print(json.dumps(vars(args), indent=2))

    ### Audio preprocessor Config
    preprocessor_config = {
        'normalize': 'per_feature',
        'window_size': 0.02,
        'sample_rate': 16000,
        'window_stride': 0.01,
        'window': 'hann',
        'features': 64,
        'n_fft': 512,
        'frame_splicing': 1,
        'dither': 1e-05,
        'stft_conv': False
    }
    n_mel = preprocessor_config['features']


    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.GetPieceSize()
    print(f"Text Tokenizer Vocabulary size: {vocab_size}")

    # DataLoaders
    print("Preparing data loaders...")
    train_loader, test_loader = get_dataloaders(
        args.train_manifest,
        args.test_manifest,
        args.tokenizer_path,
        preprocessor_config=preprocessor_config,
        batch_size=args.batch_size,
        audio_transform=None,
        num_workers=4,    #  Note: if yuou are using a GPU, set this to 0, before we fix the issue
    )

    # Model
    print("Instantiating model...")
    model = RewardModel(
        n_mel=n_mel,
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        lstm_hidden=args.hidden_dim,
        lstm_layers=1,
        audio_conv_channels=128,
        audio_conv_layers=3,
        head_hidden=args.hidden_dim,
        dropout=args.dropout,
    )
    model.to(device)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        print(f"Using scheduler: StepLR with step_size={args.scheduler_step_size}, gamma={args.scheduler_gamma}")

    # Training
    checkpoint_dir = args.save_dir + '/checkpoints'
    history = fit(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=test_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler,
    )

    # Final model
    final_path = os.path.join(args.save_dir, 'final_model.ckpt')
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    # Save training logs
    logs_path = os.path.join(args.save_dir, 'training_logs.json')
    with open(logs_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training logs to {logs_path}")

    # Final evaluation
    print("Final evaluation on test set:")
    _ = evaluate(model, test_loader, criterion, device)

    # Optional plot predictions vs targets
    if args.plot:
        preds, targets = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                audio = batch['audio_batch'].to(device)
                text = batch['text_batch'].to(device)
                labels = batch['score_batch'].numpy()
                audio_lens = batch['audio_lengths'].to(device)
                text_lens = batch['text_lengths'].to(device)

                out = model(audio, audio_lens, text, text_lens).cpu().numpy()
                preds.append(out)
                targets.append(labels)
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        # Loss curves
        plt.figure()
        plt.plot(range(1, args.epochs+1), history['train_loss'], label='Train Loss')
        plt.plot(range(1, args.epochs+1), history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Predictions vs Targets
        plt.figure()
        plt.scatter(targets, preds, alpha=0.5)
        plt.xlabel('Targets')
        plt.ylabel('Predictions')
        plt.title('Pred vs Target')
        plt.plot([0,1],[0,1], 'r--')
        plt.show()

if __name__ == '__main__':
    main()
