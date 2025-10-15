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

"""TranslationModel and ModelBackend interfaces with CycleGAN."""

import logging
from pathlib import Path
from typing import Any

from configs.cyclegan_utkfaces.model import (
    GeneratorSignature,
    TAGS_RESNET_GENERATOR_DEV
)
import mlflow.models as mlflow_models
import mlflow.pytorch as mlfpt
from mlflow_registry import (
    configure_mlflow,
    get_latest_run_by_tags,
)
from serde import from_dict
import torch
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F


from ..interfaces import ModelBackend, TranslationModel
from ..schema import (
    GenderType,
    ModelNotFoundError,
    TranslationConfig
)

logger = logging.getLogger(__name__)


class CycleGANTranslationModel(TranslationModel):
    """CycleGAN implementation of TranslationModel interface."""
    def __init__(
        self,
        signature: GeneratorSignature,
        G_M2F: torch.nn.Module,
        G_F2M: torch.nn.Module,
        device: str,
        config: TranslationConfig
    ):
        """Initialize with both generator instances."""
        self.G_M2F, self.G_F2M = G_M2F, G_F2M
        self.signature = signature
        self.device = torch.device(device)
        self.config = config

        self.G_M2F.to(self.device).eval()
        self.G_F2M.to(self.device).eval()

    def align_to_signature(
        self, input_tensor: torch.Tensor
    ):
        """Interpolate input tensor to match the model signature."""
        return F.interpolate(input_tensor, size=self.signature.input_shape[2:])

    def translate(
        self,
        input_tensor: torch.Tensor,
        source_gender: GenderType
    ) -> torch.Tensor:
        """Determine generator from source_gender argument."""
        input_tensor = input_tensor.to(self.device)
        H, W = input_tensor.shape[2:]

        generator = self.G_M2F if source_gender == "male" else self.G_F2M
        input_tensor = self.align_to_signature(input_tensor)

        with torch.inference_mode():
            if self.config.use_mixed_precision:
                with autocast(device_type=str(self.device)):
                    translated = generator(input_tensor)
            else:
                translated = generator(input_tensor)

        return F.interpolate(translated, size=(H, W))

    def get_model_info(self) -> dict[str, Any]:  # noqa: D102
        return {
            "backend": "cyclegan",
            "architecture": "resnet_generator",
            "generators": ["male_to_female", "female_to_male"],
            "m2f_parameters": sum(p.numel() for p in self.G_M2F.parameters()),
            "f2m_parameters": sum(p.numel() for p in self.G_F2M.parameters()),
            "device": str(self.device),
            "mixed_precision": self.config.use_mixed_precision
        }

    def is_ready(self) -> bool:
        """Model is ready when both generator attributes are present."""
        return self.G_M2F is not None and self.G_F2M is not None


class CycleGANBackend(ModelBackend):
    """CycleGAN model backend implementation."""

    def load_models(self, config: TranslationConfig, output_path: Path) -> TranslationModel:
        """Load CycleGAN generators from MLFlow registry."""
        try:

            configure_mlflow()
            logger.info("Loading CycleGAN generators from MLFlow registry...")

            mlflow_run = get_latest_run_by_tags({
                **TAGS_RESNET_GENERATOR_DEV,
            })

            if "m2f_model_name" not in mlflow_run.tags:
                # this is technically a lie - you could still fetch the model name
                # using run metadata; however, such soulution is fragile
                raise ModelNotFoundError("M2F model name not found in run parameters.")

            m2f_model_name = mlflow_run.tags["m2f_model_name"]
            (output_path / "m2f_model").mkdir(exist_ok=True, parents=True)

            g_m2f_uri = f"models:/{m2f_model_name}@{config.male_to_female_alias}"
            G_M2F = mlfpt.load_model(g_m2f_uri, dst_path=str(output_path / "m2f_model"))

            if "f2m_model_name" not in mlflow_run.tags:
                raise ModelNotFoundError("F2M model name not found in run parameters.")

            f2m_model_name = mlflow_run.tags["f2m_model_name"]
            (output_path / "f2m_model").mkdir(exist_ok=True, parents=True)

            g_f2m_uri = f"models:/{f2m_model_name}@{config.female_to_male_alias}"
            G_F2M = mlfpt.load_model(g_f2m_uri, dst_path=str(output_path / "f2m_model"))

            signature_dict = getattr(mlflow_models.get_model_info(g_m2f_uri), "_signature_dict", None)
            if signature_dict is None:
                raise AttributeError("Could not fetch signature dict from model info")
            signature = from_dict(GeneratorSignature, signature_dict)

            logger.info("CycleGAN models loaded successfully.")

            return CycleGANTranslationModel(
                signature=signature,
                G_M2F=G_M2F, G_F2M=G_F2M,
                device=config.device, config=config
            )

        except Exception as e:
            logger.error(f"Failed to load CycleGAN models: {e}")
            raise ModuleNotFoundError

    def get_backend_info(self) -> dict[str, Any]:  # noqa: D102
        return {
            "name": "cyclegan",
            "description":  "CycleGAN-based gender translation",
            "input_format": "tensor [-1, 1]",
            "output_format": "tensor [-1, 1]"
        }
