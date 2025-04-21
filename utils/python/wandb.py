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
from argparse import Namespace
from typing import Any, Dict, Union
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.logger import (
    _convert_json_serializable,
    _convert_params,
    _sanitize_callable_params,
)
from .preprocessing import _omegaconf_to_container

class MyWandbLogger(WandbLogger):
    """
    Custom WandbLogger for logging hyperparameters with additional filtering and transformations.

    Methods Changed
    -------
    log_hyperparams(params: Union[Dict[str, Any], Namespace]) -> None
        Logs hyperparameters to WandB with optional filtering of large configuration sections.
        
        Parameters
        ----------
        params : Union[Dict[str, Any], Namespace]
            The hyperparameters to log. Can be a dictionary or a Namespace object.
    """
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        # 1) Convert Namespace -> dict
        if isinstance(params, Namespace):
            params = vars(params)

        # 2) Standard Lightning transformations
        params = _convert_params(params)
        params = _sanitize_callable_params(params)

        # 3) (Optional) Filter out big Nemo config sections you don’t want
        filtered = {}
        for k, v in params.items():
            if k in {"cfg", "decoding", "encoder", "decoder"}:
                # skip these big / complex configs (not relevant on WandB)
                continue
            filtered[k] = v

        # In Case there would be any simpler object that are DictConfig Types
        filtered = _omegaconf_to_container(filtered)

        # 4) JSON-safe conversion on the remainder
        filtered = _convert_json_serializable(filtered)

        # 5) Update wandb.config
        self.experiment.config.update(filtered, allow_val_change=True)
