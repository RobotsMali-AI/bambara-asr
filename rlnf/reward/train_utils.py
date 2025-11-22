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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rlnf.reward.reward_model import RewardModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def fit(
    model: RewardModel,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    checkpoint_dir: str,
    scheduler: torch.optim.lr_scheduler.StepLR = None,
):
    """
    Train model for a number of epochs.

    Optionally plots training loss per epoch.
    """
    model.to(device)

    print("Starting training...")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_dataloader:
            audio = batch['audio_batch'].to(device)
            text = batch['text_batch'].to(device)
            labels = batch['score_batch'].to(device)
            audio_lens = batch['audio_lengths'].to(device)
            text_lens = batch['text_lengths'].to(device)

            preds = model(audio, audio_lens, text, text_lens)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * audio.size(0)
        epoch_loss = running_loss / len(train_dataloader.dataset)
        history['train_loss'].append(epoch_loss)
        # Validation
        val_metrics = evaluate(model, valid_dataloader, criterion, device)
        val_loss = val_metrics['loss']
        history['val_loss'].append(val_loss)

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"{checkpoint_dir}/best_model.ckpt"
            model.save(save_path=save_path)
            print(f"Saved best model to {save_path}")

        print(f"Epoch {epoch}/{epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation R2: {val_metrics['r2']:.4f}")

    return history

def evaluate(
    model: RewardModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """
    Evaluate the model on given dataloader.

    Returns MSE, MAE, R2, and accuracy@threshold.
    """
    model.eval()
    preds_all = []
    labels_all = []
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            audio = batch['audio_batch'].to(device)
            text = batch['text_batch'].to(device)
            labels = batch['score_batch'].to(device)
            audio_lens = batch['audio_lengths'].to(device)
            text_lens = batch['text_lengths'].to(device)

            preds = model(audio, audio_lens, text, text_lens)
            loss = criterion(preds, labels)
            losses.append(loss.item() * audio.size(0))

            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()
    mse = mean_squared_error(labels_all, preds_all)
    mae = mean_absolute_error(labels_all, preds_all)
    r2 = r2_score(labels_all, preds_all)
    avg_loss = sum(losses) / len(dataloader.dataset)
    return {'loss': avg_loss, 'mse': mse, 'mae': mae, 'r2': r2}
