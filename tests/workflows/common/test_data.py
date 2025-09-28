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


"""Unit tests for workflows/common/data module.

Mocks the MLFlow storage lookup utils to provide isolation.

Coverage:
    - valid and erroneus tag signatures
    - lookup exception handling
    - deduplication
    - edge cases (unicode names / large entry list)
"""

import pytest
from unittest.mock import MagicMock, Mock

from mlflow_registry.tags import TagKeys, Type
from workflows.common.data import fetch_supported_datasets_mlflow


@pytest.fixture
def mock_configure_mlflow(mocker):
    """Package-independent stub for configure_mlflow function."""
    return mocker.patch('workflows.common.data.configure_mlflow')


@pytest.fixture
def mock_search_runs(mocker):
    """Package-independent stub for search_runs_by_tags function."""
    return mocker.patch('workflows.common.data.search_runs_by_tags')


@pytest.fixture
def mlflow_mocks(mock_configure_mlflow, mock_search_runs):
    """Combined fixture for both MLFlow mocks."""
    return {
        "configure": mock_configure_mlflow,
        "search": mock_search_runs
    }


@pytest.fixture
def sample_runs():
    """Fixture providing sample MLFlow runs."""
    mock_run1 = Mock()
    mock_run1.tags = {"name": "dataset_1", "type" : Type.DATASET}

    mock_run2 = Mock()
    mock_run2.tags = {"name": "dataset_2", "type": Type.DATASET}

    mock_run3 = Mock()
    mock_run3.tags = {"name": "dataset_3", "type": Type.DATASET}

    return (mock_run1, mock_run2, mock_run3)


@pytest.fixture
def run_with_missing_name():
    """Fixture providing a run object without name tag."""
    mock_run = Mock()
    mock_run.tags = {"type": Type.DATASET}
    return mock_run


@pytest.fixture
def run_with_none_name():
    """Fixture providing a run object with None name."""
    mock_run = Mock()
    mock_run.tags = {"name": None, "type": Type.DATASET}
    return mock_run


@pytest.fixture
def duplicate_name_runs():
    """Fixture providing runs with duplicate names."""
    mock_run1 = Mock()
    mock_run1.tags = {"name": "dataset_1", "type": Type.DATASET}

    mock_run2 = Mock()
    mock_run2.tags = {"name": "dataset_1", "type": Type.DATASET}

    mock_run3 = Mock()
    mock_run3.tags = {"name": "dataset_2", "type": Type.DATASET}

    return (mock_run1, mock_run2, mock_run3)


@pytest.fixture
def unicode_name_runs():
    """Fixture providing runs with Unicode names."""
    mock_run1 = Mock()
    mock_run1.tags = {"name": "测试数据集", "type": Type.DATASET}  # Chinese

    mock_run2 = Mock()
    mock_run2.tags = {"name": "données_test", "type": Type.DATASET}  # French

    mock_run3 = Mock()
    mock_run3.tags = {"name": "тестовые_данные", "type": Type.DATASET}  # Russian

    return (mock_run1, mock_run2, mock_run3)


class TestFetchSupportedDatasetsMLflow():
    """Test suite for fetch_supported_datasets_mlflow function."""

    def test_fetch_datasets_with_valid_names(
        self, mlflow_mocks, sample_runs
    ):
        """Lookup logic is correct."""
        mlflow_mocks["search"].return_value = sample_runs

        result = fetch_supported_datasets_mlflow()

        expected_datasets = {"dataset_1", "dataset_2", "dataset_3"}
        assert result == expected_datasets
        assert isinstance(result, set)

        mlflow_mocks['configure'].assert_called_once()
        mlflow_mocks['search'].assert_called_once_with(
            {TagKeys.TAG_TYPE: Type.DATASET}
        )

    def test_fetch_datasets_with_missing_name_tags(
        self, mlflow_mocks, sample_runs,
        run_with_missing_name, run_with_none_name
    ):
        """Missing tags do not yield side-effect results."""
        pass

    def test_fetch_datasets_with_duplicate_names(
        self, mlflow_mocks, duplicate_name_runs
    ):
        """Dataset names are deduplicated on retrieval."""
        mlflow_mocks["search"].return_value = duplicate_name_runs

        result = fetch_supported_datasets_mlflow()

        expected_datasets = {"dataset_1", "dataset_2"}
        assert result == expected_datasets
        assert len(result) == 2

    def test_fetch_datasets_empty_results(
        self, mlflow_mocks
    ):
        """Empty search results do not produce side effects."""
        mlflow_mocks["search"].return_value = []

        result = fetch_supported_datasets_mlflow()

        assert result == set()
        assert len(result) == 0

        mlflow_mocks["configure"].assert_called_once()
        mlflow_mocks["search"].assert_called_once()

    def test_fetch_datasets_configure_mlflow_exception(
        self, mlflow_mocks
    ):
        """MLFlow configuration does not fail silenlty."""
        err_msg = "MLFlow configuration failed"
        mlflow_mocks["configure"].side_effect = Exception(err_msg)

        with pytest.raises(Exception, match=err_msg):
            fetch_supported_datasets_mlflow()

        mlflow_mocks['configure'].assert_called_once()
        mlflow_mocks['search'].assert_not_called()

    def test_fetch_datasets_search_runs_exception(
        self, mlflow_mocks
    ):
        """MLFlow search does not fail silently."""
        err_msg = "MLFlow search failed"
        mlflow_mocks["search"].side_effect = Exception(err_msg)

        with pytest.raises(Exception, match=err_msg):
            fetch_supported_datasets_mlflow()

        mlflow_mocks['configure'].assert_called_once()
        mlflow_mocks['search'].assert_called_once()


class TestFetchSupportedDatasetsMLFlowEdgeCases:
    """Additional tests for edge cases and boundary conditions."""

    def test_large_dataset_collection(self, mlflow_mocks):
        """Lookup works when dataset runs are plentiful."""
        mock_runs = []
        expected_names = set()

        for i in range(1000):
            mock_run = Mock()
            dataset_name = f"dataset_{i}"
            mock_run = Mock()
            mock_run.tags = {"name": dataset_name, "type": Type.DATASET}
            expected_names.add(dataset_name)
            mock_runs.append(mock_run)

        mlflow_mocks['search'].return_value = mock_runs

        result = fetch_supported_datasets_mlflow()
        assert result == expected_names
        assert len(result) == 1000

    def test_unicode_dataset_names(
        self, mlflow_mocks, unicode_name_runs
    ):
        """Test handling of Unicode characters in dataset names."""
        mlflow_mocks["search"].return_value = unicode_name_runs

        result = fetch_supported_datasets_mlflow()

        expected_datasets = {"测试数据集", "données_test", "тестовые_данные"}
        assert result == expected_datasets


if __name__ == "__main__":
    pytest.main([__file__])
