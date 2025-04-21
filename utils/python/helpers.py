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
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
import torch.nn as nn

def load_config(config_path: str):
    """Load fine-tuning configuration from a YAML file."""
    return OmegaConf.load(config_path)

def enable_bn_se(m):
    """Enable batch normalization and squeeze-excite layers for training."""
    if isinstance(m, nn.BatchNorm1d):
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)
    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

def analyse_ctc_failures_in_model(model):
    """
    Analyzes CTC (Connectionist Temporal Classification) failures in a given model.

    This function evaluates the model on its training data to count how many times
    the CTC loss computation would fail due to the length of the input sequence being
    less than or equal to the length of the target sequence. It also records the lengths
    of the acoustic model sequences and the target sequences.

    Args:
        model (torch.nn.Module): The model to be analyzed.

    Returns:
        tuple: A tuple containing:
            - count_ctc_failures (int): The number of CTC loss computation failures.
            - am_seq_lengths (list): A list of lengths of the acoustic model sequences.
            - target_seq_lengths (list): A list of lengths of the target sequences.
    """
    count_ctc_failures = 0
    am_seq_lengths = []
    target_seq_lengths = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    mode = model.training

    train_dl = model.train_dataloader()

    with torch.no_grad():
        model = model.eval()
        for batch in tqdm(train_dl, desc='Checking for CTC failures'):
            x, x_len, y, y_len = batch
            x, x_len = x.to(device), x_len.to(device)
            x_logprobs, x_len = model(input_signal=x, input_signal_length=x_len)

            # Find how many CTC loss computation failures will occur
            for xl, yl in zip(x_len, y_len):
                if xl <= yl:
                    count_ctc_failures += 1

            # Record acoustic model lengths=
            am_seq_lengths.extend(x_len.to('cpu').numpy().tolist())

            # Record target sequence lengths
            target_seq_lengths.extend(y_len.to('cpu').numpy().tolist())

            del x, x_len, y, y_len, x_logprobs

    if mode:
        model = model.train()

    return count_ctc_failures, am_seq_lengths, target_seq_lengths

# num_ctc_failures, am_seq_lengths, target_seq_lengths = analyse_ctc_failures_in_model(parakeet_100)
# print(f"CTC loss will fail for {num_ctc_failures} samples ({num_ctc_failures * 100./ float(len(am_seq_lengths))} % of samples)!\n"
#                  f"Increase the vocabulary size of the tokenizer so that this number becomes close to zero !")