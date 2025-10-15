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

"""Register CycleGAN generators in MLFlow with explicit signature.

Usage:
    poetry run python -m workflows.cyclegan_utkfaces.register_cyclegan
    --ckpt-dir /path/to/cyclegan/checkpoint/
    --use-best --alias=champion
"""

import argparse

from configs.cyclegan_utkfaces.model import (
    ResNetGeneratorConfig,
    TAGS_RESNET_GENERATOR_DEV,
)
from mlflow.client import MlflowClient
from workflows.common.config import get_config_instance
from workflows.common.mlflow_experiment_logger import ExperimentLogger

from models.cyclegan.generator import ResNetGenerator
from .io import CycleGANCheckpointManager


def main():  # noqa: D103

    parser = argparse.ArgumentParser(
        description="Register CycleGAN generator in MLflow")
    parser.add_argument(
        "--ckpt-dir", type=str, required=True,
        help="directory with checkpoints produced with training.py script"
    )
    parser.add_argument(
        "--experiment_name", type=str,
        default="register_cyclegan_generators", help="Path to checkpoint file"
    )
    parser.add_argument(
        "--model-name-prefix", type=str,
        default="cyclegan_generator",
        help="name prefix for a pair of cyclegan generators"
    )
    parser.add_argument(
        "--use-best", action="store_true",
        help="If set, use best checkpoint; use last otherwise"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="device for checkpointed model weights"
    )
    parser.add_argument(
        "--alias", type=str, default="champion",
        help="model alias in registry; useful for lookup"
    )

    args = parser.parse_args()

    experiment_logger = ExperimentLogger(args.experiment_name)

    checkpoint_manager = CycleGANCheckpointManager(str(args.ckpt_dir))

    if args.use_best:
        checkpoint = checkpoint_manager.get_best_checkpoint()
    else:
        checkpoint = checkpoint_manager.get_latest_checkpoint()

    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoints found in directory: {args.ckpt_dir}")

    config_generator_M2F = get_config_instance(ResNetGeneratorConfig)
    config_generator_F2M = get_config_instance(ResNetGeneratorConfig)

    generator_M2F = ResNetGenerator(config_generator_M2F)
    generator_F2M = ResNetGenerator(config_generator_F2M)

    checkpoint_manager.load_inference_models(
        generator_M2F, generator_F2M,
        checkpoint_path=str(checkpoint.resolve()), map_location=str(args.device)
    )

    m2f_model_name = str(args.model_name_prefix) + "_m2f"
    f2m_model_name = str(args.model_name_prefix) + "_f2m"

    with experiment_logger:

        experiment_logger.log_experiment_tags({
            **TAGS_RESNET_GENERATOR_DEV,
            "m2f_model_name": m2f_model_name,
            "f2m_model_name": f2m_model_name,
        })

        model_info_m2f = experiment_logger.log_pytorch_model(
            generator_M2F, m2f_model_name,
            config_generator_M2F.get_model_signature()
        )
        model_info_f2m = experiment_logger.log_pytorch_model(
            generator_F2M, f2m_model_name,
            config_generator_F2M.get_model_signature()
        )

        if model_info_m2f is None:
            raise RuntimeError(
                f"Unable to register model with name {m2f_model_name}")

        if model_info_f2m is None:
            raise RuntimeError(
                f"Unable to register model with name {f2m_model_name}"
            )

        client = MlflowClient()
        client.set_registered_model_alias(
            m2f_model_name, args.alias,
            str(model_info_m2f.registered_model_version)
        )

        client.set_registered_model_alias(
            f2m_model_name, args.alias,
            str(model_info_f2m.registered_model_version)
        )


if __name__ == "__main__":
    main()
