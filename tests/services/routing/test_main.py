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

"""Unit tests for main.py FastAPI application.

Coverage:
- FastAPI application initialization and lifespan management
- All endpoints: root, health, classify
- Exception handling and error responses
- File upload validation and processing
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient
from fastapi import UploadFile

from routing.main import app, lifespan
from routing.inference import (
    ClassificationResult,
    InferenceError,
    ModelNotFoundError,
    InvalidImageError,
    NoFacesDetectedError
)

class TestLifespanManager:
    """Test suite for lifespan context manager."""

    @pytest.mark.asyncio
    @patch('routing.main.create_image_processor')
    async def test_lifespan_success(self, mock_create_processor, mocker):
        """Successfully entering lifespan guarantees processor in app state."""
        mock_processor = mocker.Mock()
        mock_create_processor.return_value = mock_processor

        mock_app = mocker.Mock()
        mock_app.state = mocker.Mock()

        async with lifespan(mock_app):
            mock_create_processor.assert_called_once()
            assert mock_app.state.processor == mock_processor

    @pytest.mark.asyncio
    @patch('routing.main.create_image_processor')
    async def test_lifespan_initialization_failure(
        self, mock_create_processor, mocker
    ):
        """Init failure does not fail silenlty."""
        mock_create_processor.side_effect = Exception("Model loading failed")

        mock_app = mocker.Mock()
        mock_app.state = mocker.Mock()

        with pytest.raises(Exception, match="Model loading failed"):
            async with lifespan(mock_app):
                pass


@pytest.fixture
def client(monkeypatch):
    """Mock client with noop lifespan."""
    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    monkeypatch.setattr("routing.main.lifespan", noop_lifespan, raising=True)
    return TestClient(app)


@pytest.fixture
def mock_processor(mocker):  # noqa: D101
    processor = mocker.Mock()
    return processor


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    image_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        b'\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    return image_data


class TestEndpoints:
    """Test suite for FastAPI endpoints using TestClient."""

    def test_root_endpoint(self, client):
        """Root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["service"] == "Routing API"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data
        assert data["endpoints"]["health"] == "/health"
        assert data["endpoints"]["classify"] == "/classify"

    def test_health_check_healthy(self, client, mock_processor, mocker):
        """Service is healthy when processor is loaded."""
        client.app.state.processor = mock_processor

        response = client.get("/health")
        data = response.json()

        assert response.status_code == 200
        assert data["status"] == "healthy"
        assert data["service"] == "routing-api"
        assert "model_alias" in data

    def test_health_check_not_ready_no_processor(self, client):
        """Response with no processor provides 503 status."""
        if hasattr(client.app.state, 'processor'):
            delattr(client.app.state, 'processor')

        response = client.get("/health")
        data = response.json()

        assert response.status_code == 503
        assert data["status"] == "not_ready"
        assert data["detail"] == "Models not loaded"

    def test_classify_success(self, client, mock_processor, sample_image_bytes):
        """Successful image classification yields status code 200 + expected data."""
        mock_result = ClassificationResult(
            predicted_class="female",
            confidence=0.85,
            bbox=(10, 20, 100, 150),
            status="success"
        )
        mock_processor.classify_face.return_value = mock_result

        client.app.state.processor = mock_processor

        files = {"file": ("test_image.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/classify", files=files)
        data = response.json()

        assert response.status_code == 200
        expected_data = asdict(mock_result)
        expected_data["bbox"] = list(expected_data["bbox"])
        assert data == expected_data

        mock_processor.classify_face.assert_called_once_with(sample_image_bytes)

    @pytest.mark.parametrize(
        "upload, expected_status, detail_check, expect_called",
        [
            (
                ("test.txt", b"text content", "text/plain"),
                400, lambda d: "Invalid file type" in d, False
            ),
            (
                ("empty.jpg", b"", "image/jpeg"),
                400, lambda d: d == "Empty file uploaded", False
            ),
        ],
    )
    def test_classify_invalid_inputs(
        self, client, mock_processor, upload, expected_status, detail_check, expect_called
    ):
        """Invalid inputs yields 400 with detail."""
        client.app.state.processor = mock_processor
        filename, payload, content_type = upload
        files = {"file": (filename, io.BytesIO(payload), content_type)}
        response = client.post('/classify', files=files)

        assert response.status_code == expected_status
        data = response.json()
        assert detail_check(data["detail"])
        if not expect_called:
            mock_processor.classify_face.assert_not_called()

    def test_classify_model_not_found_error(
        self, client, mock_processor, sample_image_bytes
    ):
        """NodelNotFoundError on server side yields 503."""
        mock_processor.classify_face.side_effect = ModelNotFoundError(
            "Model not available"
        )
        client.app.state.processor = mock_processor

        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/classify", files=files)
        data = response.json()

        assert response.status_code == 503
        assert data["error"] == "MODEL_NOT_FOUND"
        assert data["detail"] == "Model not available"
        assert data["status"] == "error"

    def classify_invalid_image_error(
        self, client, mock_processor, sample_image_bytes
    ):
        """InvalidImageError on server side yields 400."""
        mock_processor.classify_face.side_effect = InvalidImageError(
            "Corrupted image"
        )
        client.app.state.processor = mock_processor
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/classify", files=files)
        data = response.json()

        assert response.status_code == 400
        assert data["error"] == "INVALID_IMAGE"
        assert data["detail"] == "Corrupted image"
        assert data["status"] == "error"

    def classify_no_faces_detected_error(
        self, client, mock_processor, sample_image_bytes
    ):
        """NoFacesDetected on server side yields 400."""
        mock_processor.classify_face.side_effect = NoFacesDetectedError(
            "Corrupted image"
        )
        client.app.state.processor = mock_processor
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/classify", files=files)
        data = response.json()

        assert response.status_code == 400
        assert data["error"] == "NO_FACE_DETECTGED"
        assert data["detail"] == "No face found"
        assert data["status"] == "error"

    def test_classify_unexpected_exception(self, client, mock_processor, sample_image_bytes):
        """Unexpected exception is treatead as server-side issue."""
        mock_processor.classify_face.side_effect = RuntimeError("Unexpected error")

        client.app.state.processor = mock_processor
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/classify", files=files)
        data = response.json()

        assert response.status_code == 500
        assert data["detail"] == "Internal server error"


if __name__ == "__main__":
    pytest.main([__file__])
