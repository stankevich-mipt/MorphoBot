#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


"""Unit tests for the vision.dlib68_detector module.

Covers:
    - detector initialization
    - face detection
    - landmark extraction
    - bounding box expansion and clipping

Notes:
    - test fetch the dlib shape predictor from the MLFlow artifact
    registry to ensure consistency with tracked artifacts
"""

import os
from pathlib import Path

import cv2
import dlib
from dotenv import load_dotenv
from mlflow_registry import (
    configure_mlflow,
    find_and_fetch_artifacts_by_tags
)
from mlflow_registry.tag_profiles import TAG_PROFILES
import numpy as np
import pytest
from vision.dlib68_detector import Dlib68Detector


SAMPLE_IMAGE_PATH = (
    Path(__file__).parent / "image_data" / "640px-Neil_Armstrong_official.jpg"
)
DOTENV_FILE_PATH = (
    Path(__file__).parent.parent.parent / "mlflow_registry" / ".env"
)


@pytest.fixture()
def predictor_path(tmp_path_factory, monkeypatch):
    """Fetches dlib predictor weights into tmp dir."""
    load_dotenv(DOTENV_FILE_PATH)

    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    )
    monkeypatch.setenv(
        "AWS_ACCESS_KEY_ID",
        os.getenv("AWS_ACCESS_KEY_ID", "")
    )
    monkeypatch.setenv(
        "AWS_SECRET_ACCESS_KEY",
        os.getenv("AWS_SECRET_ACCESS_KEY", "")
    )
    monkeypatch.setenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9002"
    )

    configure_mlflow()
    dst_dir = tmp_path_factory.mktemp("dlib_predictor")
    path = find_and_fetch_artifacts_by_tags(
        dst_dir=str(dst_dir),
        tags=TAG_PROFILES["vision_landmarks_detector"],
        unique=True
    ) / "shape_predictor_68_face_landmarks.dat"
    yield str(path)


@pytest.fixture()
def detector(predictor_path):
    """Sets test up by creating detector instance from fetched weights."""
    return Dlib68Detector(
        predictor_path=predictor_path,
        detector_kind="hog",
        upsample_times=1
    )


def test_detector_initialization(detector):
    """Instance is created with required attribute set."""
    assert detector is not None
    assert detector.detector is not None
    assert detector.shape_predictor is not None


def test_detect_one_returns_none_for_blank_image(detector):
    """Blank image results in zero detections."""
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = detector.detect_one(blank)
    assert bbox is None


@pytest.fixture(scope="module")
def test_image():
    """Test image is reachable and valid."""
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img is not None, "Test image is missing or could not be loaded"
    return img


@pytest.fixture()
def detected_bbox(detector, test_image):
    """Detector finds face on the test image."""
    bbox = detector.detect_one(test_image)
    assert bbox is not None
    assert isinstance(bbox, dlib.rectangle)  # type: ignore
    return bbox


def test_landmarks68_return_shape_and_type(
    detector, test_image, detected_bbox
):
    """Predicted landmark array has proper type and shape."""
    landmarks = detector.landmarks68(test_image, detected_bbox)
    assert landmarks.shape == (68, 2)
    assert landmarks.dtype == np.float32


def test_expand_with_margin_expands_bbox(
    detector, test_image, detected_bbox
):
    """Expanded bbox is larger than the original."""
    expanded = detector.expand_with_margin(detected_bbox, test_image)

    old_w = detected_bbox.right() - detected_bbox.left()
    old_h = detected_bbox.bottom() - detected_bbox.top()

    new_w = expanded.right() - expanded.left()
    new_h = expanded.bottom() - expanded.top()

    assert new_w >= old_w
    assert new_h >= old_h
