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

"""Implements inference logic for TranslationModel interface."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any, Optional

import cv2
from mlflow_registry import (
    configure_mlflow,
    find_and_fetch_artifacts_by_tags,
)
from mlflow_registry.tag_profiles import TAG_PROFILES
import numpy as np
from PIL import Image
import torch
from vision.dlib68_detector import Dlib68Detector
from vision.procrustes_aligner_5pt import (
    FivePointAligner, Template, WarpContext,
)

from .backend_registry import BackendRegistry
from .schema import (
    GenderType,
    InvalidImageError,
    ModelBackendType,
    ModelNotFoundError,
    NoFacesDetectedError,
    TranslationConfig,
    TranslationError,
    TranslationResult
)
from .utils import (
    create_alignment_template,
    pil_to_bytes,
    rect_to_xyxy,
)

logger = logging.getLogger(__name__)


@dataclass
class FaceProcessingContext:
    """Dataclass aggregating info about forward step of face processing."""
    bbox: Optional[tuple[int, int, int, int]] = None
    face_tensor: Optional[torch.Tensor] = None
    mask: Optional[np.ndarray] = None
    warp_context: Optional[WarpContext] = None
    alignment_template: Optional[Template] = None


class FaceDetector:
    """Face detection and alignment utility."""

    def __init__(self, output_path: Path):
        """Initalize by creating detector and aligner objects."""
        self.detector = self._load_detector(output_path)
        self.aligner = self._create_aligner()

    def _load_detector(self, output_path: Path):
        """Load face detector from MLflow artifacts."""
        try:
            artifacts_path = find_and_fetch_artifacts_by_tags(
                dst_dir=str(output_path),
                tags=TAG_PROFILES["vision_landmarks_detector"],
                unique=True
            )

            predictor_files = list(artifacts_path.glob("*.dat"))
            if not predictor_files:
                raise ModelNotFoundError("Face detector weights not found")

            return Dlib68Detector(
                predictor_path=str(predictor_files[0]),
                detector_kind='hog',
                upsample_times=1
            )
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load face detector: {str(e)}")

    def _create_aligner(self):
        """Use generic template to initialize FivePointAligner instance."""
        template = create_alignment_template()
        return FivePointAligner(
            male_template=template,
            female_template=template
        )

    def process_image(
        self, image: np.ndarray
    ) -> tuple[Optional[torch.Tensor], FaceProcessingContext]:
        """Detect, align, and preprocess face from image.

        Returns:
            Tuple (crop, bbox, WarpContext) where:
                - crop is part of the image containing face
                - bbox is an integer coordinates of face bounding box
                - WarpContext is a data structure that contains parameters
                of affine warp which produced alignment
        """
        try:
            # detect face
            bbox = self.detector.detect_one(image)
            if bbox is None:
                return None, FaceProcessingContext()

            # crop with margin to get neck/hair
            bbox = self.detector.expand_with_margin(bbox, image)
            image = image[bbox.top():bbox.bottom(), bbox.left(): bbox.right()]

            # get alignment template of size given by min(H,W)
            template = create_alignment_template(size=(min(image.shape[0], image.shape[1])))

            landmarks = self.detector.landmarks68(
                image, self.detector._xyxy_to_rect(0, 0, image.shape[1], image.shape[0]))
            if landmarks is None:
                return None, FaceProcessingContext()

            aligned_face, warp_context, _ = self.aligner.align_to_template(image, landmarks, template)
            if aligned_face is None:
                return None, FaceProcessingContext()

            # get mask to fill the blanks with original pixels
            mask_warp, _, _ = self.aligner.align_to_template(
                np.ones_like(image), landmarks, template)
            mask = self.aligner.reverse_warp(mask_warp, warp_context).astype(np.bool_)  # type: ignore

            face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype(np.float32) / 255.0

            face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1))
            face_tensor = (face_tensor * 2.0 - 1).unsqueeze(0)

            return face_tensor, FaceProcessingContext(
                alignment_template=template,
                bbox=rect_to_xyxy(bbox),
                mask=mask,
                warp_context=warp_context
            )

        except Exception as e:
            logger.error(f"Error in face processing: {e}")
            return None, FaceProcessingContext()

    def postprocess_image(
        self,
        translated_image: torch.Tensor,
        original_image: np.ndarray,
        context: FaceProcessingContext
    ) -> np.ndarray:
        """Invert the stack of transforms done with process_image."""
        translated_image = torch.clamp(
            translated_image.squeeze(0).permute(1, 2, 0), -1., 1.
        )

        np_image = (translated_image.detach().cpu().numpy() + 1.) / 2.
        np_image = (np_image * 255.).astype(np.uint8)

        np_image = cv2.resize(
            np_image, (
                context.alignment_template.target_size,   # type: ignore
                context.alignment_template.target_size    # type: ignore
            )
        )

        np_image = self.aligner.reverse_warp(
            np_image, ctx=context.warp_context)  # type: ignore
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

        bbox = context.bbox

        result = np.copy(original_image)
        result[bbox[1]:bbox[3], bbox[0]:bbox[2]] = (  # type: ignore
            context.mask * np_image  # type: ignore
            + (1 - context.mask) * original_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # type: ignore
        )

        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

class GenderTranslationEngine:
    """Gender translation orchestractor with pluggable backends."""

    def __init__(self, config: TranslationConfig, output_path: Optional[Path] = None):
        """Initialize translation engine.

        Args:
            config: Translation configuration
            output_path: Path for storing downloaded models
        """
        self.config = config
        self.output_path = output_path or Path(
            os.getenv("ARTIFACTS_OUTPUT_DIR", ".artifacts")
        )
        self.output_path.mkdir(parents=True, exist_ok=True)

        configure_mlflow()
        self._initialize()

    def _initialize(self):
        """Setup face detector and translation models."""
        logger.info("Initializing gender translation engine...")

        self.face_detector = FaceDetector(self.output_path)
        backend = BackendRegistry.get_backend(self.config.backend_type)
        self.translation_model = backend.load_models(self.config, self.output_path)

        logger.info(f"Engine initialized with {self.config.backend_type.value} backend")

    def translate_gender(
        self, image_bytes: bytes, source_gender: GenderType
    ) -> tuple[bytes, TranslationResult]:
        """Translate gender in the provided image.

        Args:
            image_bytes: Input image as bytes
            source_gender: domain to translate from ("male" or "female")

        Returns:
            Tuple of (translated_image_bytes, tranlation_result)
        """
        start_time = time.time()

        try:
            if source_gender not in ["male", "female"]:
                raise TranslationError(f"Invalid source gender: {source_gender}")

            target_gender = "female" if source_gender == "male" else "male"

            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                raise InvalidImageError("Failed to decode image bytes")

            input_tensor, processing_context = self.face_detector.process_image(image)
            if input_tensor is None or any(processing_context.__dataclass_fields__) is None:
                raise NoFacesDetectedError("No valid faces detected in image")

            result = self.translation_model.translate(input_tensor, source_gender)

            result = self.face_detector.postprocess_image(result, image, processing_context)
            result_bytes = pil_to_bytes(Image.fromarray(result), format="JPEG", quality=95)
            processing_time = (time.time() - start_time) * 1000

            result = TranslationResult(
                source_gender=source_gender,
                target_gender=target_gender,
                bbox=processing_context.bbox,  # type: ignore
                processing_time_ms=processing_time,
                model_info=self.translation_model.get_model_info()
            )

            logger.info(
                f"Translation completed: {source_gender} -> {target_gender} "
                f"in {processing_time:.1f}ms using {self.config.backend_type.value}"
            )

            return result_bytes, result

        except TranslationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during translation: {e}")
            raise TranslationError(f"Translation failed: {str(e)}")

    def health_check(self) -> dict[str, Any]:
        """Check if engine and model are up."""
        return {
            "engine_ready": True,
            "face_detection_ready": self.face_detector is not None,
            "translation_model_ready": (
                self.translation_model is not None and self.translation_model.is_ready()
            ),
            "backend_type": self.config.backend_type.value,
            "backend_info": BackendRegistry.get_backend(self.config.backend_type).get_backend_info(),
            "model_info": self.translation_model.get_model_info() if self.translation_model else {},
            "device": self.config.device
        }


def create_translation_engine(
    backend_type: ModelBackendType = ModelBackendType.CYCLEGAN,
    male_to_female_alias: str = "champion",
    female_to_male_alias: str = "champion",
    output_path: Optional[Path] = None,
    **kwargs
) -> GenderTranslationEngine:
    """Factory function to create a GenderTranslationEngine.

    Args:
        backend_type: Type of model backend to use
        male_to_female_alias: MLflow alias for M2F model
        female_to_male_alias: MLflow alias for F2M model
        output_path: Path for model storage
        **kwargs: additional configuration options

    Returns:
        Configured GenderTranslationEngine instance
    """
    config = TranslationConfig(
        backend_type=backend_type,
        male_to_female_alias=male_to_female_alias,
        female_to_male_alias=female_to_male_alias,
        **kwargs
    )

    return GenderTranslationEngine(config, output_path)
