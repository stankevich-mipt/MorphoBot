#    Copyright 2025, Stankevich Andrey, stankevich.as@phystech.edu

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""MLFlow logging utilities for structured experiment tracking."""

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Optional, Union

import mlflow
from mlflow.models import infer_signature
from mlflow.pytorch import log_model as mlflow_torch_log_model
from mlflow_registry import (
    build_artifact_s3_uri,
    configure_mlflow,
    ensure_experiment,
)
from mlflow_registry.tags import TagKeys, TagValues
import torch

# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(levelname)s] %(message)s'
# )
# logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Comprehensive MLFlow logger for ML experiments with Poetry integration."""

    def __init__(self, experiment_name: str, **start_kwargs):
        """Initalize the experiment logger.

        Args:
            experiment_name: experiment entry for the registry.
        """
        self.experiment_name = experiment_name
        self.start_kwargs = start_kwargs
        self.logger = logging.getLogger()

        configure_mlflow()
        self.artifact_location = build_artifact_s3_uri(
            experiment_name
        )
        self.experiment_id = ensure_experiment(
            self.experiment_name, self.artifact_location
        )
        self._poetry_requirements = None
        self._current_run = None

    def __enter__(self):
        """Capture current run in class attribute."""
        if self.experiment_id:
            mlflow.set_experiment(experiment_id=self.experiment_id)
        self._current_run = mlflow.start_run(**self.start_kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the run, free the reference."""
        mlflow.end_run()
        self._current_run = None

    def get_poetry_requirements(self) -> list[str]:
        """Extract requirements.txt directly with poetry export."""
        if self._poetry_requirements is not None:
            return self._poetry_requirements

        try:
            result = subprocess.run(
                ["poetry", "export", "-f", "requirements.txt", "--without-hashes"],
                capture_output=True, text=True, check=True
            )
            self._poetry_requirements = [
                line.strip() for line in result.stdout.split("\n")
                if line.strip() and not line.startswith("#")
            ]
            return self._poetry_requirements
        except subprocess.CalledProcessError as e:
            self.logger.info(f"Warning: Failed to export Poetry requirements: {e}")
            return []

    def _flatten_dataclass_dict(
        self, data: dict[str, Any], parent_key: str = ''
    ) -> dict[str, Any]:
        """Flatten nested dictionaries for parameter logging."""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dataclass_dict(v, new_key).items())
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                items.append((new_key, str(v)))
            elif v is not None:
                items.append((new_key, v))
        return dict(items)

    def log_dataclass_configs(self, **configs):
        """Log dataclass configurations as MLFlow parameters."""
        for config_name, config_obj in configs.items():
            if config_obj is None:
                continue

            if is_dataclass(config_obj):
                config_dict = asdict(config_obj)  # type: ignore
                flat_config = self._flatten_dataclass_dict(
                    config_dict, config_name
                )

                mlflow_params = {}
                for key, value in flat_config.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow_params[key] = value
                    else:
                        mlflow_params[key] = str(value)

                mlflow.log_params(mlflow_params)

                config_file = f"{config_name}_config.json"
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
                mlflow.log_artifact(config_file)
                Path(config_file).unlink()

            else:
                pass
                self.logger.info(
                    f"Warning: {config_name} is not a dataclass; "
                    "skipping structured logging"
                )

    def log_experiment_tags(self, tags: dict[str | TagKeys, str | TagValues]):
        """Log experiment tag profile with additional metadata."""
        stringifed_tags = {str(k): str(v) for k, v in tags.items()}

        enhanced_tags = {
            **stringifed_tags,
            "timestamp": datetime.now().isoformat(),
            "mlflow_version": mlflow.__version__,
            "python_version": subprocess.run(
                ["python", "--version"], capture_output=True, text=True
            ).stdout.strip()
        }

        mlflow.set_tags(enhanced_tags)

    def log_pytorch_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_example: Optional[torch.Tensor] = None,
    ):
        """Log Pytorch model using dedicated MLFlow API."""
        pip_requirements = self.get_poetry_requirements()

        signature = None
        if input_example is not None:
            with torch.inference_mode():
                model_output = model(input_example)
                signature = infer_signature(
                    input_example.cpu().numpy(),
                    model_output.cpu().numpy()
                )

        if input_example is not None and signature is not None:
            mlflow_torch_log_model(
                pytorch_model=model,
                registered_model_name=model_name,
                artifact_path=model_name,
                pip_requirements=pip_requirements,
                input_example=input_example.cpu().numpy(),
                signature=signature
            )
        else:
            mlflow_torch_log_model(
                pytorch_model=model,
                registered_model_name=model_name,
                artifact_path=model_name,
                pip_requirements=pip_requirements,
            )
