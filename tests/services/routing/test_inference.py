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

"""Unit tests for inference.py module.

Coverage:
- Exception classes and error handling
- ModelLoader static methods with MLflow integration
- ImageProcessor complete pipeline
- Edge cases and error conditions
"""

from dataclasses import asdict
import dlib
import numpy as np
import torch
import cv2
from pathlib import Path
import pytest
from unittest.mock import patch

from routing.inference import (
    ClassificationResult,
    create_image_processor,
    ImageProcessor,
    InvalidImageError,
    ModelLoader,
    ModelNotFoundError,
    NoFacesDetectedError,
    TEMPLATE_SIZE
)


@pytest.fixture
def mock_dlib_detector(mocker):  # noqa: D103
    return mocker.patch("routing.inference.Dlib68Detector")


@pytest.fixture
def mock_fetch_artifacts(mocker):
    """Provides isolation by mocking MLflow artifact fetching."""
    return mocker.patch(
        "routing.inference.find_and_fetch_artifacts_by_tags"
    )

@pytest.fixture
def mock_run_lookup(mocker):
    """Provides isolation by mocking MLflow run lookup."""
    return mocker.patch(
        "routing.inference.get_latest_run_by_tags"
    )

@pytest.fixture
def mock_load_model(mocker):
    """Provides isolation by mocking MLflow Pytorch model loading."""
    return mocker.patch(
        "routing.inference.mlfpt.load_model"
    )


class TestModelLoader:
    """Test suite for ModelLoader class."""

    def test_load_detector_success(
        self, mock_dlib_detector, mock_fetch_artifacts,
        tmp_path, mocker
    ):
        """Detector loading executes the expected call sequence."""
        # setup
        artifact_path = tmp_path / "artifacts"
        weights_file = artifact_path / "shape_predictor_68_face_landmarks.dat"
        artifact_path.mkdir()
        weights_file.touch()

        mock_fetch_artifacts.return_value = artifact_path
        mock_detector_instance = mocker.Mock()
        mock_dlib_detector.return_value = mock_detector_instance

        # act
        results = ModelLoader.load_detector(tmp_path / "output")

        # assert
        mock_fetch_artifacts.assert_called_once()
        mock_dlib_detector.assert_called_once_with(
            predictor_path=str(weights_file),
            detector_kind='hog',
            upsample_times=1
        )
        assert results == mock_detector_instance

    def test_load_detector_weights_not_found(
        self, mock_fetch_artifacts, tmp_path
    ):
        """ModelLoader raises when detector weights are missing."""
        artifact_path = tmp_path / "artifacts"
        artifact_path.mkdir()

        mock_fetch_artifacts.return_value = artifact_path

        with pytest.raises(
            ModelNotFoundError,
            match="Predictor weights not found"
        ):
            ModelLoader.load_detector(tmp_path / "output")

    def test_load_detector_mlflow_error(
        self, mock_fetch_artifacts, tmp_path
    ):
        mock_fetch_artifacts.side_effect = Exception("MLflow error")
        """ModelLoader raises when MLflow fetch fails."""
        with pytest.raises(Exception, match="MLflow error"):
            ModelLoader.load_detector(tmp_path / "output")

    def test_load_classifier_success(
        self, mock_run_lookup, mock_load_model, mocker, tmp_path
    ):
        """Classifier loading executes expected call sequence."""
        # setup
        mock_run = mocker.Mock()
        mock_run.params = {
            "model.model_name": "test_classifier"
        }
        mock_run_lookup.return_value = mock_run

        mock_classifier = mocker.Mock()
        mock_classifier_to = mocker.Mock(
            return_value=mock_classifier
        )
        mock_classifier.to = mock_classifier_to

        mock_load_model.return_value = mock_classifier

        # act
        result = ModelLoader.load_classifier(
            tmp_path / "output", "test_alias"
        )

        # assert
        mock_run_lookup.assert_called_once()
        mock_load_model.assert_called_once_with(
            "models:/test_classifier@test_alias",
            dst_path=str(tmp_path / "output")
        )
        mock_classifier.eval.assert_called_once()
        mock_classifier_to.assert_called_once()

        assert result == mock_classifier

    def test_load_classifier_no_model_name(
        self, mock_run_lookup, tmp_path, mocker
    ):
        """ModelLoader raises if model name is not in run parameters."""
        mock_run = mocker.Mock()
        mock_run.params = {}
        mock_run_lookup.return_value = mock_run

        with pytest.raises(
            ModelNotFoundError,
            match="Model name not found in run parameters"
        ):
            ModelLoader.load_classifier(tmp_path / "output")

    def test_load_classifier_mlflow_error(
        self, mock_run_lookup, tmp_path
    ):
        """ModelLoader raises when MLflow run lookup fails."""
        mock_run_lookup.side_effect = Exception("MLflow error")

        with pytest.raises(Exception, match="MLflow error"):
            ModelLoader.load_classifier(tmp_path / "output")


@pytest.fixture
def mock_detector(mocker):
    """Mock object mirroring DlibDetector68 interface."""
    detector = mocker.Mock()
    detector.detect_one.return_value = mocker.Mock()
    detector.landmarks68.return_value = (
        np.random.randn(68, 2).astype(np.float32)
    )
    return detector

@pytest.fixture
def mock_classifier(mocker):
    """Mock object mirroring ResNetClassifier interface."""
    classifier = mocker.Mock()
    # higher confidence for class 0
    logits = torch.tensor([[2.0, 1.0]])
    classifier.return_value = logits
    return classifier

@pytest.fixture
def mock_aligner(mocker):
    """Mock object mirroring ResNetClassifier interface."""
    aligner = mocker.Mock()
    aligned_face = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    aligner.align.return_value = (aligned_face, None, None)
    return aligner


class TestImageProcessor:
    """ImageProcessor class testing suite."""

    @patch('routing.inference.configure_mlflow')
    @patch('routing.inference.ModelLoader.load_detector')
    @patch('routing.inference.ModelLoader.load_classifier')
    @patch('routing.inference.create_alignment_template')
    @patch('routing.inference.FivePointAligner')
    def test_init_success(
        self, mock_aligner_class, mock_template, mock_load_classifier,
        mock_load_detector, mock_configure_mlflow, tmp_path, mocker
    ):
        """ImageProcessor init executes the expected call sequence."""
        mock_load_detector.return_value = mock_detector
        mock_load_classifier.return_value = mock_classifier
        mock_aligner_class.return_value = mock_aligner
        mock_template.return_value = mock_template_obj = mocker.Mock()

        processor = ImageProcessor(tmp_path, "test_alias")

        mock_configure_mlflow.assert_called_once()
        mock_load_detector.assert_called_once_with(tmp_path)
        mock_load_classifier.assert_called_once_with(tmp_path, alias="test_alias")
        mock_template.assert_called_once()
        mock_aligner_class.assert_called_once_with(
            male_template=mock_template_obj,
            female_template=mock_template_obj
        )

        assert processor.detector == mock_detector
        assert processor.classifier == mock_classifier
        assert processor.aligner == mock_aligner
        assert processor.output_path == tmp_path

    def test_preprocess_face_success(
        self, mock_detector, mock_aligner, mocker
    ):
        """Sanity check for face preprocessing logic."""
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.detector = mock_detector
        processor.aligner = mock_aligner

        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        mock_bbox = mocker.Mock()
        mock_detector.detect_one.return_value = mock_bbox

        landmarks = np.random.rand(68, 2).astype(np.float32)
        mock_detector.landmarks68.return_value = landmarks

        aligned_face = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        mock_aligner.align.return_value = (aligned_face, None, None)

        tensor, bbox = processor.preprocess_face(test_image)

        mock_detector.detect_one.assert_called_once_with(test_image)
        mock_detector.landmarks68.assert_called_once_with(
            test_image, bbox
        )
        mock_aligner.align.assert_called_once_with(test_image, landmarks)

        assert tensor is not None
        assert bbox == mock_bbox
        assert tensor.shape == (1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE)

    def test_preprocess_face_no_detection(
        self, mock_detector, mock_aligner
    ):
        """Preprocessing fails early when no face is detected."""
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.detector = mock_detector
        processor.aligner = mock_aligner

        mock_detector.detect_one.return_value = None

        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor, bbox = processor.preprocess_face(test_image)

        assert tensor is None
        assert bbox is None
        mock_detector.landmarks68.assert_not_called()
        mock_aligner.align.assert_not_called()

    def test_preprocess_face_no_landmarks(
        self, mock_detector, mock_aligner, mocker
    ):
        """Preprocessing fails early when landmarks cannot be extracted."""
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.detector = mock_detector
        processor.aligner = mock_aligner

        mock_detector.detect_one.return_value = mocker.Mock()
        mock_detector.landmarks68.return_value = None

        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor, bbox = processor.preprocess_face(test_image)

        assert tensor is None
        assert bbox is None
        mock_aligner.align.assert_not_called()

    def test_preprocess_face_alignment_failure(
        self, mock_detector, mock_aligner, mocker
    ):
        """Alignment failure results in tensor and bbox both being None."""
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.detector = mock_detector
        processor.aligner = mock_aligner

        mock_detector.detect_one.return_value = mocker.Mock()
        mock_detector.landmarks68.return_value = np.random.rand(68, 2).astype(np.float32)
        mock_aligner.align.return_value = (None, None, None)

        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor, bbox = processor.preprocess_face(test_image)

        assert tensor is None
        assert bbox is None

    def test_classify_face_success(
        self, mock_detector, mock_classifier, mock_aligner, mocker
    ):
        """Sanity check for face classification logic."""
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.detector = mock_detector
        processor.classifier = mock_classifier
        processor.aligner = mock_aligner

        test_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

        with (
            patch('routing.inference.np.frombuffer') as mock_frombuffer,
            patch('routing.inference.cv2.imdecode') as mock_imdecode,
            patch.object(processor, 'preprocess_face') as mock_preprocess
        ):
            mock_frombuffer.return_value = np.array([1, 2, 3], dtype=np.uint8)
            mock_imdecode.return_value = test_image

            mock_tensor = torch.randn(1, 3, 256, 256)
            mock_bbox = mocker.Mock()
            mock_bbox.left.return_value = 10
            mock_bbox.top.return_value = 20
            mock_bbox.right.return_value = 100
            mock_bbox.bottom.return_value = 150
            mock_preprocess.return_value = (mock_tensor, mock_bbox)

            image_bytes = b"fake_jpeg_data"
            result = processor.classify_face(image_bytes)

            mock_frombuffer.assert_called_once_with(
                image_bytes, dtype=np.uint8
            )
            mock_imdecode.assert_called_once()
            mock_preprocess.assert_called_once_with(test_image)
            mock_classifier.assert_called_once_with(mock_tensor)

            assert isinstance(result, ClassificationResult)
            assert result.predicted_class == "male"
            assert 0.0 <= result.confidence <= 1.0
            assert result.bbox == (10, 20, 100, 150)
            assert result.status == "success"

    def test_classify_face_invalid_image(
        self, mock_detector, mock_classifier, mock_aligner
    ):
        """ImageProcessor raises when image data is invalid."""
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.detector = mock_detector
        processor.classifier = mock_classifier
        processor.aligner = mock_aligner

        with (
            patch('routing.inference.np.frombuffer') as mock_frombuffer,
            patch('routing.inference.cv2.imdecode') as mock_imdecode
        ):
            mock_frombuffer.return_value = np.array([1, 2, 3], dtype=np.uint8)
            mock_imdecode.return_value = None

            with pytest.raises(InvalidImageError, match="Failed to decode image bytes"):
                processor.classify_face(b"invalid_image_data")

    def test_classify_face_no_face_detected(
        self, mock_detector, mock_classifier, mock_aligner
    ):
        """ImageProcessor raises when no face is detected."""
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.detector = mock_detector
        processor.classifier = mock_classifier
        processor.aligner = mock_aligner

        test_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

        with (
            patch('routing.inference.np.frombuffer') as mock_frombuffer,
            patch('routing.inference.cv2.imdecode') as mock_imdecode,
            patch.object(processor, 'preprocess_face') as mock_preprocess
        ):
            mock_frombuffer.return_value = np.array([1, 2, 3], dtype=np.uint8)
            mock_imdecode.return_value = test_image
            mock_preprocess.return_value = (None, None)

            with pytest.raises(NoFacesDetectedError, match="No valid face detected"):
                processor.classify_face(b"image without face")


def test_create_image_processor(mocker):
    """Factory function behaves as expected."""
    mock_processor_class = mocker.patch(
        'routing.inference.ImageProcessor'
    )
    mock_instance = mocker.Mock()
    mock_processor_class.return_value = mock_instance

    result = create_image_processor()
    mock_processor_class.assert_called_once_with()
    assert result == mock_instance

    # reset mock and test with args
    mock_processor_class.reset_mock()
    result = create_image_processor(
        output_path=Path("test"), classifier_alias="test_alias"
    )
    mock_processor_class.assert_called_once_with(
        output_path=Path("test"),
        classifier_alias="test_alias"
    )
    assert result == mock_instance


# TODO: write meaningful behavioural tests for module-level
# constants and environment variables.

if __name__ == "__main__":
    pytest.main([__file__])
