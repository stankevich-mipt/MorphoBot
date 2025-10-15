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

"""Interfaces for backend-agnostic image translation service.

Provides abstract interfaces for:
- TranslationModel: Generic image-to-image translation interface
- ModelBackend: Backend-specific model loading and inference
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from .schema import GenderType, TranslationConfig

class TranslationModel(ABC):
    """Abstract interface for gender translation models."""

    @abstractmethod
    def translate(
        self,
        input_tensor: torch.Tensor,
        source_gender: GenderType
    ) -> torch.Tensor:
        """Translate gender in the input tensor.

        Args:
            input_tensor: Input image tensor (B,C,H,W) in range [-1, 1]
            source_gender: "male" or "female"

        Returns:
            Translated tensor with same shape and range as input
        """
        ...

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded models."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if models are loaded and ready for inference."""


class ModelBackend(ABC):
    """Abstract backend for loading and managing translation models."""

    @abstractmethod
    def load_models(
        self,
        config: TranslationConfig,
        output_path: Path
    ) -> TranslationModel:
        """Load translation model according to configuration."""
        ...

    @abstractmethod
    def get_backend_info(self) -> dict[str, Any]:
        """Get information about this backend."""
        ...
