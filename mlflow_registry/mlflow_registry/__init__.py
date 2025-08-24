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


"""Package mlflow/mlflow_registry.

Initialization file provides API for registry-related operations
available to other services.
"""

from .config import RegistryConfig
from .search import (
    find_and_fetch_artifacts_by_tags,
    get_latest_runs_by_tags,
    get_unique_run_by_tags,
    RegistrySearchError
)
from .uri_mapping import with_artifact_root

__all__ = [
    "RegistryConfig",
    "RegistrySearchError",
    "find_and_fetch_artifacts_by_tags",
    "get_latest_runs_by_tags",
    "get_unique_run_by_tags",
    "with_artifact_root",
]

def configure_mlflow() -> None:
    """Read env variables and configure MLFlow registry."""
    RegistryConfig().configure_mlflow()
