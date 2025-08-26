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


"""Unit tests for RegistryConfig class in mlflow_registry.config module.

Tests cover:
    - Proper reading and application of environment variables
    - Correct fallback to configured defaults
    - Idempotency of configuration calls
    - Handling of partial environment variable setup
"""

import os
from unittest.mock import patch

import mlflow
from mlflow_registry import RegistryConfig
import pytest


def test_registry_config_calls_set_tracking_uri(monkeypatch):
    """Check if mlflow.set_tracking_uri is called by cfg class."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://test-mlflow-server:5000")

    with patch("mlflow.set_tracking_uri") as mock_set_uri:

        cfg = RegistryConfig()
        cfg.configure_mlflow()
        mock_set_uri.assert_called_once_with("http://test-mlflow-server:5000")


def test_registry_config_sets_s3_env(monkeypatch):
    """Apply env vars and check whether cfg sets them accordingly."""
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://minio-test:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")

    cfg = RegistryConfig()

    with patch.dict(os.environ, {}, clear=True) as mock_env:

        cfg.configure_mlflow()

        assert mock_env.get("MLFLOW_S3_ENDPOINT_URL") == "http://minio-test:9000"
        assert mock_env.get("AWS_ACCESS_KEY_ID") == "test-access-key"
        assert mock_env.get("AWS_SECRET_ACCESS_KEY") == "test-secret-key"


def test_registry_config_defaults(monkeypatch):
    """Defaults are used when env. variables are missing."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    cfg = RegistryConfig()
    cfg.configure_mlflow()

    assert mlflow.get_tracking_uri() == "http://localhost:5000"
    assert os.environ.get("MLFLOW_S3_ENDPOINT_URL", "") == ""
    assert os.environ.get("AWS_ACCESS_KEY_ID", "") == ""
    assert os.environ.get("AWS_SECRET_ACCESS_KEY", "") == ""


def test_config_idempotent_calls(monkeypatch):
    """Multiple configure calls retain consistent MLFlow state."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://test-mlflow-server:5000")

    with patch("mlflow.set_tracking_uri") as mock_set_uri:

        cfg = RegistryConfig()
        cfg.configure_mlflow()
        cfg.configure_mlflow()

        assert mock_set_uri.call_count == 2
        mock_set_uri.assert_any_call("http://test-mlflow-server:5000")


def test_partial_env_vars(monkeypatch):
    """Partial env vars configure MLFlow without errors."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://partial-server:5000")
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "partial-access")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    cfg = RegistryConfig()

    with (
        patch.dict(os.environ, {}, clear=True) as mock_env,
        patch("mlflow.set_tracking_uri") as mock_set_uri
    ):

        cfg.configure_mlflow()

        mock_set_uri.assert_called_once_with("http://partial-server:5000")

        assert mock_env.get("MLFLOW_S3_ENDPOINT_URL", "") == ""
        assert mock_env.get("AWS_ACCESS_KEY_ID", "") == "partial-access"
        assert mock_env.get("AWS_SECRET_ACCESS_KEY", "") == ""
