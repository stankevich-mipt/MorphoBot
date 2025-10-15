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

"""Defines gender_translation service constants schema and contracts."""


from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


GenderType = Literal["male", "female"]


class ModelBackendType(Enum):
    """Supported model backend types."""
    CYCLEGAN = "cyclegan"


@dataclass
class TranslationConfig:
    """Configuration for translation models and image processing."""
    backend_type: ModelBackendType
    male_to_female_alias: str = "champion"
    female_to_male_alias: str = "champion"
    device: str = "auto"
    batch_size: int = 1
    use_mixed_precision: bool = True


@dataclass
class TranslationResult:
    """Result of gender translation operation."""
    source_gender: str
    target_gender: str
    bbox: tuple[int, int, int, int]
    processing_time_ms: float
    model_info: dict[str, Any]
    status: str = "success"


class TranslationError(Exception):
    """Base exception for translation errors."""
    def __init__(  # noqa: D107
        self,
        message: str,
        code: str = "TRANSLATION_ERROR",
        status: int = 500
    ):
        super().__init__(message)
        self.code = code
        self.status = status


class ModelNotFoundError(TranslationError):
    """Raised when required models cannot be found or loaded."""
    def __init__(  # noqa: D107
        self, message: str = "Required translation models could not be found."
    ):
        super().__init__(message, "MODEL_NOT_FOUND", 503)


class InvalidImageError(TranslationError):
    """Raised when image is invalid or cannot be processed."""
    def __init__(  # noqa: D107
        self, message: str = "Invalid or corrupted image is provided."
    ):
        super().__init__(message, "INVALID_IMAGE", 400)


class NoFacesDetectedError(TranslationError):
    """Raised when no faces are detected in the image."""
    def __init__(  # noqa: D107
        self, message: str = "No faces detected in the provided image"
    ):
        super().__init__(message, "NO_FACE_DETECTED", 400)


class UnsupportedBackendError(TranslationError):
    """Raised when unsupported backend is requested."""
    def __init__(  # noqa: D107
        self, message: str = "Unsupported model backend."
    ):
        super().__init__(message, "UNSUPPORTED_BACKEND", 501)
