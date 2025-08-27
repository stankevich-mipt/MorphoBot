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


"""Unit tests for mlflow_registry.uri_mapping module.

Tests cover:
    - URI building with different schemes
    - Environment variable handling
    - Decorator functionality for path transformation
"""

from unittest.mock import patch

from mlflow_registry import uri_mapping
import pytest


def test_build_artifact_s3_uri_with_s3_scheme(monkeypatch):
    """Build URI uses s3:// scheme when env var has s3 root."""
    monkeypatch.setenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT",
        "s3://my-bucket/artifacts"
    )

    result = uri_mapping.build_artifact_s3_uri("models/my_model")
    assert result == "s3://my-bucket/artifacts/models/my_model"


def test_build_artifat_s3_uri_with_file_scheme(monkeypatch):
    """Build URI uses file:// scheme when env var has file root."""
    monkeypatch.setenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT",
        "file:///local/artifacts"
    )

    result = uri_mapping.build_artifact_s3_uri("models/my_model")
    assert result == "file:///local/artifacts/models/my_model"


def test_build_artifact_s3_uri_handles_trailing_slashes(monkeypatch):
    """Build URI normalizes trailing and leading slashes in path and subpath."""
    monkeypatch.setenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT",
        "s3://bucket/artifacts/"
    )

    result = uri_mapping.build_artifact_s3_uri("/models/my_model")
    assert result == "s3://bucket/artifacts/models/my_model"


def test_build_artifact_s3_uri_handles_empty_subpath(monkeypatch):
    """Build URI works correctly with empty artifact subpath."""
    monkeypatch.setenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT",
        "s3://bucket/artifacts/"
    )
    result = uri_mapping.build_artifact_s3_uri("")
    assert result == "s3://bucket/artifacts/"


@pytest.mark.parametrize(
    "root,subpath,expected", [
        (
            "s3://bucket/path",
            "file.txt",
            "s3://bucket/path/file.txt"
        ),
        (
            "file:///tmp/mlflow",
            "data/model.pkl",
            "file:///tmp/mlflow/data/model.pkl"
        ),
        (
            "s3://my_artifacts",
            "nested/deep/file",
            "s3://my_artifacts/nested/deep/file"
        )
    ]
)
def test_build_artifact_s3_uri_parametrized(
    monkeypatch, root, subpath, expected
):
    """Build URI correctly handles various root and subpath combinations."""
    monkeypatch.setenv("MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT", root)

    result = uri_mapping.build_artifact_s3_uri(subpath)
    assert result == expected


def test_build_artifacts_s3_uri_handles_special_characters(monkeypatch):
    """Build URI correctly handles special chars in subpath."""
    monkeypatch.setenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT",
        "s3://bucket/artifacts"
    )

    result = uri_mapping.build_artifact_s3_uri("models/model-v1.0_final.pk")
    assert result == "s3://bucket/artifacts/models/model-v1.0_final.pk"


def test_with_artifact_root_decorator(monkeypatch):
    """Decorator transforms first string argument into full artifact URI."""
    monkeypatch.setenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT",
        "s3://bucket/artifacts"
    )

    @uri_mapping.with_artifact_root
    def mock_function(uri, extra_arg):
        return f"processed: {uri}, extra: {extra_arg}"

    result = mock_function("models/test", "additional")
    assert result == (
        "processed: s3://bucket/artifacts/models/test, "
        "extra: additional"
    )


def test_with_artifact_root_decorator_preserves_function_metadata():
    """Decorator preserves original function name and docstring."""
    @uri_mapping.with_artifact_root
    def sample_function(path):
        """Sample function docstring."""
        return path

    assert sample_function.__name__ == "sample_function"
    assert sample_function.__doc__ == "Sample function docstring."


def test_with_artifact_root_decorator_with_kwargs(monkeypatch):
    """Decorator works with functions that accept keyword arguments."""
    monkeypatch.setenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT",
        "file:///local/artifacts"
    )

    @uri_mapping.with_artifact_root
    def mock_function(uri, flag=False):
        return f"uri: {uri}, flag: {flag}"

    result = mock_function("data.csv", flag=True)
    assert result == "uri: file:///local/artifacts/data.csv, flag: True"


def test_with_artifact_root_decorator_kwonly_args(monkeypatch):
    """Decorator should work for kword-only functions."""
    monkeypatch.setenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT",
        "file:///local/artifacts"
    )

    @uri_mapping.with_artifact_root
    def func(*, artifact_subpath=None, flag=False):
        return f"artifact_subpath: {artifact_subpath}, flag: {flag}"

    result = func(artifact_subpath="data.csv", flag=True)
    excepted_uri = "file:///local/artifacts/data.csv"
    assert result == f"artifact_subpath: {excepted_uri}, flag: True"
