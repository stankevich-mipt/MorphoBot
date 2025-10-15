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

"""Script to load evaluation metric models in MLFlow registry.

Downloads and registers the following models:
- Inception-v3 for FID computation
- AlexNet/VGG/SqueezeNet for LPIPS computation
- Inception-v3 classifier for Inception Score
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema, TensorSpec
from mlflow_registry.tags import TagKeys, TagValues
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvmodels
from workflows.common.mlflow_experiment_logger import ExperimentLogger


try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRegistrar:
    """Handles registration of evaluation models with MLFlow logger."""

    def __init__(self, experiment_name: str = "evaluation_models_registry"):
        """Setup MLFlow experiment that will aggregate the model data."""
        self.experiment_name = experiment_name
        self.experiment_logger = ExperimentLogger(experiment_name)
        logger.info(
            "Initialized evaluation model registrar;"
            f" experiment name is {experiment_name}"
        )

    def register_fid_model(self) -> None:
        """Register Inception-v3 model for FID computation."""
        logger.info("Registering FID Inception-v3 model...")
        model = tvmodels.inception_v3(pretrained=True, transform_input=False)

        model.fc = nn.Identity()  # type: ignore
        model.eval()

        with self.experiment_logger:

            signature, input_example = self._get_fid_model_io()
            # pass dummy image as input_example to infer signature
            self.experiment_logger.log_pytorch_model(
                model=model,
                model_name="inception_v3_fid",
                signature=signature,
                input_example=input_example,
            )

            self.experiment_logger.log_experiment_tags({
                "metric_type": "fid",
                "feature_dim": "2048",
                "input_size": "299x299",
                "pretrained": "true",
                "output_layer": "removed_fc",
                "usage": "batch_feature_extraction"
            })

            # additionally log configuration as json
            model_config = {
                "model_type": "inception_v3_fid",
                "feature_dim": 2048,
                "input_size": [299, 299],
                "num_parameters": sum(
                    p.numel() for p in model.parameters()
                )
            }

            with open("fid_model_config.json", "w") as f:
                json.dump(model_config, f, indent=2)

            mlflow.log_artifact("fid_model_config.json", "model_config")
            Path("fid_model_config.json").unlink()

            logger.info("FID model registered successfully")

    def _get_fid_model_io(self) -> tuple[ModelSignature, torch.Tensor]:
        """Helper for inputs and signature of FID Inception-V3 models."""
        input_schema = Schema([
            TensorSpec(type=np.dtype(np.float32), shape=(-1, 3, 299, 299), name="images"),
        ])
        output_schema = Schema([
            TensorSpec(type=np.dtype(np.float32), shape=(-1, 2048), name="features"),
        ])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = torch.randn(1, 3, 299, 299, dtype=torch.float32)

        return signature, input_example

    def register_lpips_model(self, networks: Optional[list[str]] = None) -> None:
        """Register LPIPS models for perceptual similarity computation.

        Args:
            networks: List of networks to register ("alex", "vgg", "squeeze").
            If not provided, all will be registered.
        """
        if not LPIPS_AVAILABLE:
            logger.warning("LPIPS library not available; skipping LPIPS model registration")
            return

        if networks is None:
            networks = ["alex", "vgg", "squeeze"]

        for net_type in networks:
            logger.info(f"Registering LPIPS {net_type} model...")

            with self.experiment_logger:

                lpips_model = lpips.LPIPS(net=net_type)  # type: ignore
                lpips_model.eval()

                signature, input_example = self._get_lpips_model_io()

                self.experiment_logger.log_pytorch_model(
                    model=lpips_model,
                    model_name=f"lpips_{net_type}",
                    signature=signature,
                    input_example=input_example
                )

                self.experiment_logger.log_experiment_tags({
                    "metric_type": "lpips",
                    "network_type": net_type,
                    "input_range": "[-1, 1]",
                    "min_input_size": "64x64",
                    "pretrained": "true",
                    "output_type": "similariy distance",
                    "usage": "image_pair_comparison"
                })

                model_config = {
                    "model_type": f"lpips_{net_type}",
                    "network_backbone": net_type,
                    "input_range": [-1, 1],
                    "minimum_size": [64, 64],
                    "num_parameters": sum(p.numel() for p in lpips_model.parameters()),
                    "spatial_average": True
                }

                config_file = f"lpips_{net_type}_config.json"
                with open(config_file, "w") as f:
                    json.dump(model_config, f, indent=2)

                mlflow.log_artifact(config_file, "model_config")
                Path(config_file).unlink()

                logger.info(f"LPIPS {net_type} model registered successfully")

    def _get_lpips_model_io(self) -> tuple[ModelSignature, torch.Tensor]:
        """Helper for inputs and signature of LPIPS models."""
        input_schema = Schema([
            TensorSpec(type=np.dtype(np.float32), shape=(-1, 3, 64, 64), name="img1"),
            TensorSpec(type=np.dtype(np.float32), shape=(-1, 3, 64, 64), name="img2"),
        ])
        output_schema = Schema([
            TensorSpec(type=np.dtype(np.float32), shape=(-1, 1), name="lpips_distance"),
        ])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = torch.randn(1, 3, 64, 64, dtype=torch.float32) * 2. - 1.

        return signature, input_example

    def register_inception_score_model(self):
        """Register Inception-v3 classifier for InceptionScore computation."""
        logger.info("Registering InceptionScore model...")

        with self.experiment_logger:

            model = tvmodels.inception_v3(pretrained=True, transform_input=False)
            model.eval()

            signature, input_example = self._get_is_model_io()
            self.experiment_logger.log_pytorch_model(
                model=model,
                model_name="inception_v3_classifier",
                signature=signature,
                input_example=input_example
            )

            self.experiment_logger.log_experiment_tags({
                "metric_type": "inception_score",
                "num_classes": "1000",
                "input_size": "299x299",
                "pretrained": "true",
                "dataset": "imagenet",
                "splits": "10",
                "usage": "quality_diversity_measurement"
            })

            model_config = {
                "model_type": "inception_v3_classifier",
                "num_classes": 1000,
                "input_size": [299, 299],
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "default_splits": 10,
                "pretrained_dataset": "imagenet",
                "output_type": "class_probabilities"
            }

            with open("inception_score_config.json", "w") as f:
                json.dump(model_config, f, indent=2)

            mlflow.log_artifact("inception_score_config.json")
            Path("inception_score_config.json").unlink()

            logger.info("Inception Score model registered successfully")

    def _get_is_model_io(self) -> tuple[ModelSignature, torch.Tensor]:
        """Helper for inputs and signature of Inception-V3 Score model."""
        input_schema = Schema([
            TensorSpec(type=np.dtype(np.float32), shape=(-1, 3, 299, 299), name="images"),
        ])
        output_schema = Schema([
            ColSpec("float", name="is_mean"),
            ColSpec("float", name="is_std"),
        ])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = torch.randn(1, 3, 299, 299, dtype=torch.float32)

        return signature, input_example

    def register_all_models(
        self, lpips_networks: Optional[list[str]] = None
    ) -> None:
        """Push each model for network-dependend metrics into registry."""
        models_registered = []

        try:
            self.register_fid_model()
            models_registered.append("FID")
        except Exception as e:
            logger.error(f"Failed to register FID model: {e}")

        try:
            self.register_lpips_model(lpips_networks)
            models_registered.append("LPIPS")
        except Exception as e:
            logger.error(f"Failed to register LPIPS models: {e}")

        try:
            self.register_inception_score_model()
            models_registered.append("Inception Score")
        except Exception as e:
            logger.error(f"Failed to register Inception Score model: {e}")

        # Log registration summary in a separate run
        with self.experiment_logger:
            mlflow.log_params({
                "models_registered":  ", ".join(models_registered),
                "total_models_count": len(models_registered),
                "lpips_networks": ", ".join(lpips_networks or ["alex", "vgg", "squeeze"])
            })

        summary = {
            "models_registered": models_registered,
            "total_models_count": len(models_registered),
            "lpips_networks": lpips_networks or ["alex", "vgg", "squeeze"]
        }

        with open("registration_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact("registration_summary.json", "summary")
        Path("registration_summary.json").unlink()

        logger.info(f"Model registration complete! Registered: {', '.join(models_registered)}")

    def list_registered_models(self) -> None:
        """List all registered evaluation models using MLFlow client."""
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            models = client.search_registered_models()

            eval_models = [
                model for model in models
                if any(
                    keyword in model.name.lower() for keyword in
                    ["fid", "lpips", "inception", "evaluation"]
                )
            ]

            if eval_models:
                logger.info("Registered evaluation models:")
                for model in eval_models:
                    latest_version = model.latest_versions[0] if model.latest_versions else None
                    version_info = f"v{latest_version.version}" if latest_version else "No versions"
                    logger.info(f"  -{model.name} ({version_info})")

                    if latest_version:
                        try:
                            version_details = client.get_model_version(
                                model.name, latest_version.version
                            )
                            if version_details.tags:
                                logger.info(
                                    f"    Tags: {dict(version_details.tags)}")
                        except Exception:
                            pass
            else:
                logger.info("No evaluation models found in registry")

        except Exception as e:
            logger.error(f"Failed to list models: {e}")


def main():  # noqa: D103

    parser = argparse.ArgumentParser(
        description="Register evaluation metric models using ExperimentLogger"
    )

    parser.add_argument(
        "--experiment-name",
        default="evaluation_models_registry",
        help="MLFlow experiment name for model registration"
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Models to register: all, fid, lpips, is, or comma-separated list"
    )
    parser.add_argument(
        "--lpips-networks",
        default="alex,vgg,squeeze",
        help="LPIPS networks to register (comma-separated)"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list existing models, don't register new ones"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        registrar = ModelRegistrar(args.experiment_name)
    except Exception as e:
        logger.error(f"Failed to initialize ModelRegistrar: {e}")
        sys.exit(1)

    if args.list_only:
        registrar.list_registered_models()
        return

    models_to_register = [m.strip().lower() for m in args.models.split(",")]
    lpips_networks = [n.strip().lower() for n in args.lpips_networks.split(",")]

    if "all" in models_to_register:
        registrar.register_all_models(lpips_networks)
    else:
        if "fid" in models_to_register:
            registrar.register_fid_model()
        if "lpips" in models_to_register:
            registrar.register_lpips_model(lpips_networks)
        if "is" in models_to_register:
            registrar.register_inception_score_model()

    registrar.list_registered_models()


if __name__ == "__main__":
    main()
