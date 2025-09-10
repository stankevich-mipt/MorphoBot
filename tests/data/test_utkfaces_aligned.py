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


"""Unit test for data/utkfaces_aligned module.

Coverage:
    TestUTKFacesImageFolder - tests for the image folder dataset
        - initialization with/without augmentation pipelines
        - dataset length calculation
        - item retrieval with proper data flow
        - configuration-based instantiation via from_config()
        - error handling for missing files and directories

    TestUTKFacesDataset - test for paired male/female dataset
        - Balanced and unbalanced data scenarios
        - Even/odd index logic for gender selection
        - Label assignment verification
        - configuration loading and error handling
"""

from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, Mock

from data.utkfaces_aligned import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    normalize_imagenet,
    UTKFacesDataset,
    UTKFacesImageFolder,
)
import numpy as np
import numpy.typing as npt
import pytest


class TestNormalizeImagenet:
    """Test suite for normalize_imagenet function."""

    def test_normalize_imagenet_basic(self):
        """Normalization result for simple image matches expectation."""
        img = np.ones((3, 224, 224), dtype=np.float32)
        result = normalize_imagenet(img)

        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

        expected = (1.0 - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        expected = np.tile(expected, (1, 224, 224))
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_imagenet_different_sizes(self):
        """Normalization does not interfere with (H, W) dimensions."""
        sizes = [(3, 32, 32), (3, 128, 128), (3, 512, 512)]

        for size in sizes:
            img = np.random.rand(*size).astype(np.float32)
            result = normalize_imagenet(img)
            assert result.shape == size
            assert result.dtype == np.float32


class TestUTKFacesImageFolder:
    """Test suite for UTKFacesImageFolder class."""

    @pytest.fixture
    def mock_image_paths(self) -> list[Path]:
        """Mock image paths for testing."""
        return [
            Path("/fake/path/image1.jpg"),
            Path("/fake/path/image2.jpg"),
            Path("/fake/path/image3.jpg"),
        ]

    @pytest.fixture
    def mock_image_data(self):
        """Create test image stubs with numpy."""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_init_basic(self, mock_image_paths):
        """Basic init correctly handles image_paths argument."""
        dataset = UTKFacesImageFolder(image_paths=mock_image_paths)

        assert len(dataset.image_paths) == 3
        assert dataset.aug_stack is None
        assert dataset.image_paths == sorted(mock_image_paths)

    def test_init_with_augmentation(self, mock_image_paths):
        """Augmentation pipeline is present if provided."""
        mock_aug = Mock()
        dataset = UTKFacesImageFolder(
            image_paths=mock_image_paths, aug=mock_aug)

        assert dataset.aug_stack == mock_aug

    def test_len(self):
        """Dataset len is fetched from image_paths __len__ attribute."""
        class LenSpy(list):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.len_called = 0

            def __len__(self):
                self.len_called += 1
                return super().__len__()

        mock_image_paths = LenSpy([1, 2, 3])
        dataset = UTKFacesImageFolder(image_paths=mock_image_paths)

        assert len(dataset) == len(mock_image_paths)
        assert mock_image_paths.len_called >= 1

    def test_getitem_without_augmentation(
        self, mock_image_paths, mock_image_data, mocker
    ):
        """Fetched dataset element is (3, H, W) np.array of float32."""
        mock_read_image = mocker.patch("data.utkfaces_aligned.read_image_rgb")
        mock_read_image.return_value = mock_image_data

        dataset = UTKFacesImageFolder(image_paths=mock_image_paths)
        result = dataset[0]

        mock_read_image.assert_called_once_with(mock_image_paths[0])

        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

        # reasonable range for ImageNet normalization
        assert np.min(result) >= -3.0
        assert np.max(result) <= 3.0

    def test_getitem_with_augmentation(
        self, mock_image_paths, mock_image_data, mocker
    ):
        """Albumentations pipeline is called if present."""
        mock_read_image = mocker.patch("data.utkfaces_aligned.read_image_rgb")
        mock_read_image.return_value = mock_image_data
        mock_aug = mocker.Mock()
        mock_aug.return_value = {"image": mock_image_data}

        dataset = UTKFacesImageFolder(image_paths=mock_image_paths, aug=mock_aug)
        result = dataset[0]

        mock_aug.assert_called_once_with({"image": mock_image_data})

        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32
        assert np.min(result) >= -3.0
        assert np.max(result) <= 3.0

    def test_getitem_image_read_failure(
        self, mock_image_paths, mock_image_data, mocker
    ):
        """Raises ValueError if image reading fails."""
        mock_read_image = mocker.patch("data.utkfaces_aligned.read_image_rgb")
        mock_read_image.return_value = None

        dataset = UTKFacesImageFolder(image_paths=mock_image_paths)

        with pytest.raises(ValueError, match="Could not get image at"):
            dataset[0]

    def test_from_config_success(
        self, mock_image_paths, mocker
    ):
        """Instantiation with valid config yields proper dataset."""
        mock_find_images = mocker.patch(
            "data.utkfaces_aligned.find_images", return_value=mock_image_paths)
        mock_path_exists = mocker.patch("pathlib.Path.exists", return_value=True)

        config = {
            "root": "/fake/root/path",
            "aug": None
        }
        dataset = UTKFacesImageFolder.from_config(config)

        assert len(dataset.image_paths) == 3
        assert dataset.aug_stack is None
        mock_find_images.assert_called_once()
        mock_path_exists.assert_called_once()

    def test_from_config_with_augmentation(
        self, mock_image_paths, mocker
    ):
        """Config with aug stack yields proper dataset."""
        mocker.patch(
            "data.utkfaces_aligned.find_images", return_value=mock_image_paths)
        mocker.patch("pathlib.Path.exists", return_value=True)
        mock_aug = mocker.Mock()

        config = {
            "root": "/fake/root/path",
            "aug": mock_aug
        }
        dataset = UTKFacesImageFolder.from_config(config)

        assert dataset.aug_stack is mock_aug

    def test_from_config_nonexistent_root(self, mocker):
        """Classmethod from_config raises on invalid image folder root."""
        mocker.patch("pathlib.Path.exists", return_value=False)

        config = {"root": "/nonexistent/path"}

        with pytest.raises(
            FileNotFoundError, match="Root directory does not exist"
        ):
            UTKFacesImageFolder.from_config(config)


class TestUTKFacesDataset:
    """Test suite for UTKFacesDataset class."""

    @pytest.fixture
    def mock_male_paths(self):
        """Create mock male image paths."""
        return [
            Path("/fake/male/image1.jpg"),
            Path("/fake/male/image2.jpg"),
            Path("/fake/male/image3.jpg")
        ]

    @pytest.fixture
    def mock_female_paths(self):
        """Create mock female image paths."""
        return [
            Path("/fake/female/image1.jpg"),
            Path("/fake/female/image2.jpg"),
        ]

    @pytest.fixture
    def mock_image_data(self):
        """Create test image stubs with numpy."""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_init_basic(self, mock_male_paths, mock_female_paths):
        """Basic init correctly handles path arguments."""
        dataset = UTKFacesDataset(
            image_paths_male=mock_male_paths,
            image_paths_female=mock_female_paths,
        )

        assert len(dataset.image_paths_male) == 3
        assert len(dataset.image_paths_female) == 2
        assert dataset.aug_stack is None

    def test_len(self, mock_male_paths, mock_female_paths):
        """Len is correctly evaluated in balanced/unbalanced cases."""
        dataset_unbalanced = UTKFacesDataset(
            image_paths_male=mock_male_paths,
            image_paths_female=mock_female_paths,
        )
        assert len(dataset_unbalanced) == 4

        mock_female_paths.append(Path("/fake/female/img3.jpg"))
        dataset_balanced = UTKFacesDataset(
            image_paths_male=mock_male_paths,
            image_paths_female=mock_female_paths
        )
        assert len(dataset_balanced) == 6

    def test_len_calls_on_attributes(self):
        """Dataset len estimate assesses both folder sizes."""
        class LenSpy(list):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.len_called = 0

            def __len__(self):
                self.len_called += 1
                return super().__len__()

        male_spy = LenSpy([1, 2, 3])
        female_spy = LenSpy([1, 2])

        dataset = UTKFacesDataset(
            image_paths_male=male_spy,
            image_paths_female=female_spy
        )

        result = len(dataset)

        assert male_spy.len_called >= 1
        assert female_spy.len_called >= 1
        assert result == 4

    def test_getitem_even_index_male(
        self, mock_male_paths, mock_female_paths, mock_image_data, mocker
    ):
        """Even subscripts yield male images."""
        mock_read_image = mocker.patch("data.utkfaces_aligned.read_image_rgb")
        mock_read_image.return_value = mock_image_data

        dataset = UTKFacesDataset(
            image_paths_male=mock_male_paths,
            image_paths_female=mock_female_paths
        )
        img, label = dataset[0]

        mock_read_image.assert_called_once_with(mock_male_paths[0])
        assert label == 0
        assert img.shape == (3, 224, 224)
        assert img.dtype == np.float32

    def test_getitem_even_index_female(
        self, mock_male_paths, mock_female_paths, mock_image_data, mocker
    ):
        """Even subscripts yield male images."""
        mock_read_image = mocker.patch("data.utkfaces_aligned.read_image_rgb")
        mock_read_image.return_value = mock_image_data

        dataset = UTKFacesDataset(
            image_paths_male=mock_male_paths,
            image_paths_female=mock_female_paths
        )
        img, label = dataset[1]

        mock_read_image.assert_called_once_with(mock_female_paths[0])
        assert label == 1
        assert img.shape == (3, 224, 224)
        assert img.dtype == np.float32

    def test_getitem_with_augmentation(
        self, mock_male_paths, mock_female_paths, mock_image_data, mocker
    ):
        """Aug stack is properly involved in getitem logic."""
        mocker.patch(
            'data.utkfaces_aligned.read_image_rgb', return_value=mock_image_data
        )
        mock_aug = mocker.Mock()
        mock_aug.return_value = {"image": mock_image_data}

        dataset = UTKFacesDataset(
            image_paths_male=mock_male_paths,
            image_paths_female=mock_female_paths,
            aug=mock_aug
        )
        dataset[0]

        mock_aug.assert_called_once_with({"image": mock_image_data})

    def test_getitem_image_read_failure(
        self, mock_male_paths, mock_female_paths, mocker
    ):
        """Getitem raises when image reading fails."""
        mocker.patch("data.utkfaces_aligned.read_image_rgb", return_value=None)

        dataset = UTKFacesDataset(
            image_paths_male=mock_male_paths,
            image_paths_female=mock_female_paths,
        )

        with pytest.raises(ValueError, match="Could not get image at"):
            dataset[0]

    def test_from_config_success(
        self, mock_male_paths, mock_female_paths, mocker
    ):
        """Instantiation with valid config yields proper dataset."""
        mock_find_images = mocker.patch("data.utkfaces_aligned.find_images")
        mock_find_images.side_effect = [mock_male_paths, mock_female_paths]
        mocker.patch('pathlib.Path.exists', return_value=True)

        config = {
            "root_male": "/fake/male/path",
            "root_female": "/fake/female/path"
        }
        dataset = UTKFacesDataset.from_config(config)

        assert len(dataset.image_paths_male) == 3
        assert len(dataset.image_paths_female) == 2
        assert dataset.aug_stack is None
        assert mock_find_images.call_count == 2

    def test_from_config_male_root_not_exists(self, mocker):
        """Nonexistent male root in config raises."""
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_path_exists.side_effect = [False, True]

        config = {
            "root_male": "/nonexistent/male/path",
            "root_female": "/fake/female/path"
        }

        with pytest.raises(
            FileNotFoundError, match="Root directory does not exist"
        ):
            UTKFacesDataset.from_config(config)

    def test_from_config_female_root_not_exists(self, mocker):
        """Nonexistent male root in config raises."""
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_path_exists.side_effect = [True, False]

        config = {
            "root_male": "/fake/male/path",
            "root_female": "/nonexistent/female/path",
        }

        with pytest.raises(
            FileNotFoundError, match="Root directory does not exist"
        ):
            UTKFacesDataset.from_config(config)
