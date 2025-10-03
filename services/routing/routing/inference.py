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

"""Inference logic for routing microservice."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import dlib
import mlflow.pytorch as mlfpt
from mlflow_registry import (
    configure_mlflow,
    find_and_fetch_artifacts_by_tags,
    get_latest_run_by_tags,
)
from mlflow_registry.tag_profiles import TAG_PROFILES
import numpy as np
import numpy.typing as npt
import torch
from vision.dlib68_detector import Dlib68Detector
from vision.procrustes_aligner_5pt import (
    FivePointAligner, Template
)


logger = logging.getLogger(__name__)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_LABEL_MAP = {0: "male", 1: "female"}

# get the constants from env, if possible
OUTPUT_DIR = Path(os.getenv("ARTIFACTS_OUTPUT_DIR", "./artifacts"))
TEMPLATE_SIZE = int(os.getenv("TEMPLATE_SIZE", "256"))
PREDICTOR_WEIGHTS_FILENAME = os.getenv(
    "PREDICTOR_WEIGHTS_FILENAME",
    "shape_predictor_68_face_landmarks.dat"
)
MLFLOW_CLASSIFIER_ALIAS = os.getenv(
    "MLFLOW_CLASSIFIER_ALIAS", "champion"
)
MLFLOW_CLASSIFIER_NAME = os.getenv(
    "MLFLOW_CLASSIFIER_NAME",
    "resnet_backbone_classifier"
)


@dataclass
class ClassificationResult:
    """Schema for ImageProcessor output signature."""
    predicted_class: str
    confidence: float
    bbox: tuple[int, int, int, int]
    status: str = "success"


class InferenceError(Exception):
    """Base class for inference pipeline errors."""
    def __init__(  # noqa: D107
        self,
        message: str,
        code: str = "INFERENCE_ERROR",
        status: int = 500
    ):
        super().__init__(message)
        self.code, self.status = code, status


class ModelNotFoundError(InferenceError):
    """Raises when required models cannot be loaded."""
    def __init__(  # noqa: D107
        self,
        detail: str = "Required model could not be found."
    ):
        super().__init__(detail, code="MODEL_NOT_FOUND", status=503)


class InvalidImageError(InferenceError):
    """Raised when image cannot be processed."""
    def __init__(  # noqa: D107
        self,
        detail: str = "Invalid or corrupted image"
    ):
        super().__init__(detail, "INVALID_IMAGE", status=400)


class NoFacesDetectedError(InferenceError):
    """Raised when no faces were detected on the image."""
    def __init__(  # noqa: D107
        self,
        detail: str = "No faces detected on the image"
    ):
        super().__init__(detail, "NO_FACE_DETECTED", status=400)


def create_alignment_template(size: int = TEMPLATE_SIZE):
    """Provide the default template for face alignment."""
    points = np.array([
        [0.35 * size, 0.38 * size],
        [0.65 * size, 0.38 * size],
        [0.50 * size, 0.52 * size],
        [0.40 * size, 0.70 * size],
        [0.60 * size, 0.70 * size],
    ], dtype=np.float32)

    return Template(points=points, target_size=size)

def normalize_imagenet(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Apply standard ImageNet1K image scaling."""
    return (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def rect_to_xyxy(rect: dlib.rectangle) -> tuple[int, int, int, int]:  # type: ignore
    """Rectange object from dlib to tuple of ints."""
    return rect.left(), rect.top(), rect.right(), rect.bottom()


class ModelLoader:
    """Handles loading of detector and classifier models from MLFlow."""

    @staticmethod
    def load_detector(output_path: Path) -> Dlib68Detector:
        """Load dlib face detector from MLFlow artifacts."""
        try:
            predictor_path = find_and_fetch_artifacts_by_tags(
                dst_dir=str(output_path),
                tags=TAG_PROFILES["vision_landmarks_detector"],
                unique=True
            ) / PREDICTOR_WEIGHTS_FILENAME

            if not predictor_path.exists():
                raise ModelNotFoundError(
                    f"Predictor weights not found at {predictor_path}")

            detector = Dlib68Detector(
                predictor_path=str(predictor_path),
                detector_kind='hog',
                upsample_times=1
            )

            logger.info(f"Loaded face detector from {predictor_path}")
            return detector

        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            raise ModelNotFoundError(f"Failed to load face detector: {e}")

    @staticmethod
    def load_classifier(
        output_path: Path,
        alias: str = MLFLOW_CLASSIFIER_ALIAS
    ) -> torch.nn.Module:
        """Load classifier model from MLFlow registry."""
        try:
            classifier_run = get_latest_run_by_tags(
                tags=TAG_PROFILES["routing_model"]
            )

            model_name = classifier_run.params.get("model.model_name")
            if not model_name:
                raise ModelNotFoundError("Model name not found in run parameters")

            model_uri = f"models:/{model_name}@{alias}"
            classifier = mlfpt.load_model(model_uri, dst_path=str(output_path))

            classifier.eval()
            classifier = classifier.to(torch.device('cpu'))

            logger.info(f"Loaded classifier from {model_uri}")
            return classifier

        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            raise ModelNotFoundError(f"Failed to load classifier: {e}")


class ImageProcessor:
    """Handles the complete face detection, alignment and classification pipeline."""

    def __init__(
        self,
        output_path: Optional[Path] = None,
        classifier_alias: str = MLFLOW_CLASSIFIER_ALIAS
    ):
        """Initialize from MLFlow registry."""
        self.output_path = output_path or OUTPUT_DIR

        configure_mlflow()

        logger.info("Loading models...")
        self.detector = ModelLoader.load_detector(self.output_path)
        self.classifier = ModelLoader.load_classifier(
            self.output_path, alias=classifier_alias
        )

        template = create_alignment_template()
        self.aligner = FivePointAligner(
            male_template=template,
            female_template=template
        )

        logger.info("ImageProcessor initialized successfully")

    def preprocess_face(
        self,
        face_img: npt.NDArray[np.uint8],
        target_size: int = TEMPLATE_SIZE
    ) -> tuple[Optional[torch.tensor], Optional[dlib.rectangle]]:  # type: ignore
        """Detect, align, and preprocess face from image.

        Returns:
            Tuple of (preprocessed_tensor, detected_bbox) or (None, None)
            if no faces found.

        """
        try:
            bbox = self.detector.detect_one(face_img)
            if bbox is None:
                return None, None

            landmarks = self.detector.landmarks68(face_img, bbox)
            if landmarks is None:
                return None, None

            aligned_face, _, _ = self.aligner.align(face_img, landmarks)
            if aligned_face is None:
                return None, None

            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            aligned_face = cv2.resize(aligned_face, (target_size, target_size))

            face_tensor = np.transpose(aligned_face, (2, 0, 1)).astype(np.float32) / 255.
            face_tensor = normalize_imagenet(face_tensor)
            face_tensor = torch.from_numpy(face_tensor).unsqueeze(0)

            return face_tensor, bbox

        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            return None, None

    def classify_face(
        self,
        image_bytes: bytes,
        label_map: Optional[dict[int, str]] = None
    ) -> ClassificationResult:
        """Complete pipeline: decode -> detect -> align -> classify.

        Args:
            image_bytes: raw image data
            label_map: optional custom label mapping

        Returns:
            Dictionary with prediction results

        Raises:
            InvalidImageError: if image cannot be decoded
            NoFaceDetectedError: if no face is found
        """
        if label_map is None:
            label_map = DEFAULT_LABEL_MAP

        # decode image
        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if img is None:
            raise InvalidImageError("Failed to decode image bytes")

        processed_tensor, bbox = self.preprocess_face(img)  # type: ignore

        if processed_tensor is None or bbox is None:
            raise NoFacesDetectedError("No valid face detected in image")

        with torch.inference_mode():
            logits = self.classifier(processed_tensor)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)

        predicted_idx = int(top_idx.item())
        confidence = float(top_prob.item())
        predicted_label = label_map.get(
            predicted_idx, f"class_{predicted_idx}"
        )

        return ClassificationResult(
            predicted_class=predicted_label,
            confidence=confidence,
            bbox=rect_to_xyxy(bbox)
        )

def create_image_processor(**kwargs) -> ImageProcessor:
    """Factory function to create ImageProcessor with optional configuration."""
    return ImageProcessor(**kwargs)
