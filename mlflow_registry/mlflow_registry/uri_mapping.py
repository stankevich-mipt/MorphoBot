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

"""Utilites for maintaining correct artifact URIs."""

import functools
import os
from urllib.parse import urlparse


_DEFAULT_ROOT = "files:///mlflow/artifacts"


def build_artifact_s3_uri(artifact_subpath: str) -> str:
    """Map subpath into valid registry URI using env.

    Composes a full artifact URI pointing to the S3 storage
    backing the MLFlow artifact root, i.e., prefixes the
    artifact subpath with the configured root.

    Reads the artifact root from MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT
    environment variable, falling back to "file:///mlflow/artifacts"
    if unset.

    Supports switching between file:// and s3:// schemes transparently.

    Args:
        artifact_subpath: relative path to the artifact inside the
        artifact root

    Returns:
        Full URI string, e.g., "s3://mlflow_artifacts/my/path" or
        "file:///mlflowartifacts/my/path"
    """
    default_root = _DEFAULT_ROOT
    artifact_root = os.getenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT", default_root
    )

    parsed = urlparse(artifact_root)
    scheme = parsed.scheme
    root_path = parsed.netloc + parsed.path
    normalized_root = root_path.rstrip("/").lstrip("/")
    normalized_subpath = artifact_subpath.rstrip("/").lstrip("/")

    if scheme == "s3":
        full_uri = f"s3://{normalized_root}/{normalized_subpath}"
    else:
        full_uri = f"file:///{normalized_root}/{normalized_subpath}"

    return full_uri


def with_artifact_root(func):
    """Transform the first string argument into proper artifact URI."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        if len(args) == 0:
            if kwargs:
                first_k, first_v = next(iter(kwargs.items()))
                kwargs[first_k] = build_artifact_s3_uri(first_v)
            else:
                return func()
        else:
            args = list(args)
            args[0] = build_artifact_s3_uri(args[0])
            args = tuple(args)
        return func(*args, **kwargs)

    return wrapper
