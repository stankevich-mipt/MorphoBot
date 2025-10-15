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

"""Middleware for registering and managing translation models."""

import logging
from typing import Any

from .interfaces import ModelBackend
from .models.cyclegan import CycleGANBackend
from .schema import (
    ModelBackendType,
    UnsupportedBackendError
)

logger = logging.getLogger(__name__)

class BackendRegistry:
    """Registry for managing different model backends."""
    _backends: dict[ModelBackendType, ModelBackend] = {
        ModelBackendType.CYCLEGAN: CycleGANBackend(),
    }

    @classmethod
    def get_backend(cls, backend_type: ModelBackendType) -> ModelBackend:
        """Pick proper class for backend model by backend type."""
        if backend_type not in cls._backends:
            raise UnsupportedBackendError(f"Backend {backend_type.value} not supported")
        return cls._backends[backend_type]

    @classmethod
    def list_backends(cls) -> list[dict[str, Any]]:
        """Reveal all available backends with their info."""
        return [
            {"type": backend_type.value, **backend.get_backend_info()}
            for backend_type, backend in cls._backends.items()
        ]

    @classmethod
    def register_backend(cls, backend_type: ModelBackendType, backend: ModelBackend):
        """Register a new backend."""
        cls._backends[backend_type] = backend
        logger.info(f"Registered backend: {backend_type.value}")
