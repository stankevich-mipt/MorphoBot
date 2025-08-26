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


"""Unit tests for mlflow_registry.search module.

Tests cover:
    - Searching MLFlow runs by tags
    - uniqueness constraints
    - artifact fetching

"""

from dataclasses import dataclass
from pathlib import Path
import re
from unittest.mock import MagicMock, patch

from mlflow_registry import search
from mlflow_registry.search import RegistrySearchError, RunRecord
import pytest


@dataclass
class MockInfo:  # noqa
    run_id: str
    experiment_id: str
    start_time: int


@dataclass
class MockData:  # noqa
    tags: dict[str, str]
    params: dict[str, str]


class MockRun:
    """Imitates MLFlow run object data access structure."""
    def __init__(self, run_id, experiment_id, start_time, tags, params):  # noqa
        self.info = MockInfo(run_id, experiment_id, start_time)
        self.data = MockData(tags, params)


# Sample MLFlow run record mocks used for filtering in tests
SAMPLE_RUNS = [
    MockRun(
        run_id="run1",
        experiment_id="exp1",
        start_time=1000,
        tags={"tagA": "val1", "tagB": "val2"},
        params={"param1": "x"}
    ),
    MockRun(
        run_id="run2",
        experiment_id="exp1",
        start_time=2000,
        tags={"tagA": "valX", "tagB": "val2"},
        params={"param1": "y"}
    ),
    MockRun(
        run_id="run3",
        experiment_id="exp2",
        start_time=3000,
        tags={"tagA": "val1", "tagC": "val3"},
        params={"param1": "z"}
    ),
]


@dataclass
class MockExperiment:
    """Imitates MLFlow experiment data structure."""
    experiment_id: str
    lifecycle_stage: str


# Predefined experiment mock objects keyed by experiment name
MOCK_EXPERIMENTS = {
    "exp1": MockExperiment(experiment_id="exp1", lifecycle_stage="active"),
    "exp2": MockExperiment(experiment_id="exp2", lifecycle_stage="active"),
}


@pytest.fixture
def mlflow_client_mock():
    """Mocks mlflow_tracking.MLflowClient behaviour.

    Patches search_runs, search_experiments,and get_experiment_by_name
    methods with side effects that provide simple lookup logic.
    """
    with (
        patch("mlflow_registry.search.mlflow_tracking.MlflowClient")
        as client_class_mock
    ):
        client_mock = MagicMock()
        client_class_mock.return_value = client_mock

        def search_runs_side_effect(
            experiment_ids, filter_string,
            max_results, order_by
        ):
            matches = re.findall(
                r"tags\.([^\s=]+) = '([^']+)'", filter_string
            )
            tags = {k: v for k, v in matches}

            filtered = [
                run for run in SAMPLE_RUNS
                if run.info.experiment_id in experiment_ids
                and all(run.data.tags.get(k) == v for k, v in tags.items())
            ]

            if order_by:
                order_expr = order_by[0].strip()
                # Parse field and direction
                if order_expr.startswith("attribute.start_time"):
                    reverse = order_expr.endswith("DESC")
                    filtered.sort(key=lambda r: r.info.start_time, reverse=reverse)

            return filtered

        client_mock.search_runs.side_effect = search_runs_side_effect
        client_mock.get_experiment_by_name.side_effect = (
            lambda name: MOCK_EXPERIMENTS.get(name, None)
        )

        client_mock.search_experiments.return_value = list(
            MOCK_EXPERIMENTS.values()
        )

        yield client_mock


def test_search_runs_by_tags_returns_expected_runs(mlflow_client_mock):
    """Search returns runs matching given tag filters."""
    runs = search.search_runs_by_tags({"tagA": "valX"})
    assert isinstance(runs, list)
    assert len(runs) == 1
    assert runs[0].run_id == "run2"


def test_search_run_by_tags_empty_on_no_match(mlflow_client_mock):
    """Search returns empty list when no runs match tags."""
    runs = search.search_runs_by_tags({"tagA": "nonexistent"})
    assert runs == []


def test_get_unique_run_by_tags_success(mlflow_client_mock):
    """Unique run search retuns exactly one matching run."""
    run = search.get_unique_run_by_tags({"tagA": "valX"})
    assert isinstance(run, RunRecord)
    assert run.run_id == "run2"


def test_get_unique_run_by_tags_raises_on_zero(mlflow_client_mock):
    """Unique run search raises error if multiple matches found."""
    with pytest.raises(RegistrySearchError, match="No MLFlow runs found"):
        search.get_unique_run_by_tags({"tagA": "nonexistent"})


def test_get_unique_runs_by_tags_raises_on_duplicates(mlflow_client_mock):
    """Latest run search returns most recent run matching tags."""
    with pytest.raises(RegistrySearchError, match="Non-unique runs"):
        search.get_unique_run_by_tags({"tagB": "val2"})


def test_get_latest_run_by_tags_success(mlflow_client_mock):
    """Latest run search returns most recent run matching tags."""
    run = search.get_latest_run_by_tags({"tagA": "val1"})
    assert isinstance(run, RunRecord)
    assert run.run_id == "run3"


def test_get_latest_run_by_tags_raises_on_zero(mlflow_client_mock):
    """Latest run search raises error if no runs found."""
    with pytest.raises(RegistrySearchError, match="No MLFlow runs found"):
        search.get_latest_run_by_tags({"tagA": "nonexistent"})


@pytest.mark.parametrize(
    "unique_flag, expected_run_id", [
        (True, "run2"),
        (False, "run2")
    ]
)
def test_find_and_fetch_artifacts_by_tags(
    mlflow_client_mock, unique_flag, expected_run_id
):
    """Find run by tags and fetch artifact path (resolve download is mocked)."""
    path_mock = Path("/tmp/fake_artifact")

    with patch(
        "mlflow_registry.search.resolve_artifact_to_local",
        return_value=path_mock
    ):

        result = search.find_and_fetch_artifacts_by_tags(
            tags={"tagA": "valX"},
            artifact_subpath="artifact.png",
            experiment_names=["exp1"],
            unique=unique_flag,
            dst_dir="/tmp"
        )

        assert isinstance(result, Path)
        assert result == path_mock
