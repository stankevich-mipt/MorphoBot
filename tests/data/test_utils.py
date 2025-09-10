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


"""Unit tests for data/utils module.

Coverage:
    TestFindImages:
        - empty directory / non-existent path handling
        - multiple supported extensions
        - mixed files (images and non-images)
        - files with unsupported extensions

    TestGetTrainTestSplitIndices:
        - basic train/test splits with different ratios
        - index completeness and non-overlap verification
        - shuffle behaviour, reproducibility with different seeds

"""

import os
from pathlib import Path
import tempfile
from unittest.mock import Mock

from data.utils import find_images, get_train_test_split_indices
import numpy as np
import pytest


class TestFindImages:
    """Test suite for find_images function."""

    def test_find_images_empty_directory(self, tmp_path):
        """Empty directory produces empty list."""
        result = find_images(tmp_path)
        assert result == []

    def test_find_images_no_images(self, tmp_path):
        """No files with supported extensions -> empty list."""
        (tmp_path / "document.txt").write_text("content")
        (tmp_path / "script.py").write_text("print('hello')")
        (tmp_path / "data.csv").write_text("a,b,c")

        result = find_images(tmp_path)
        assert result == []

    def test_find_images_multiple_extensions(self, tmp_path):
        """Images with various supported extension are discoverable."""
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        expected_files = []

        for i, ext in enumerate(extensions):
            image_file = tmp_path / f"image{i}{ext}"
            image_file.write_bytes(b"fake image data")
            expected_files.append(image_file)

        result = find_images(tmp_path)
        assert len(result) == len(extensions)
        assert sorted(result) == sorted(expected_files)

    def test_find_images_case_insensitive(self, tmp_path):
        """Uppercase extensions are supported."""
        files = [
            tmp_path / "photo.JPG",
            tmp_path / "image.PNG",
            tmp_path / "pic.JPEG"
        ]

        for file in files:
            file.write_bytes(b"fake_image_data")

        result = find_images(tmp_path)
        assert len(result) == 3
        assert sorted(result) == sorted(files)

    def test_find_images_recursive(self, tmp_path):
        """Images in subdirectories are also discovered."""
        subdir1 = tmp_path / "subdir1"
        subdir2 = tmp_path / "subdir1" / "nested"
        subdir1.mkdir()
        subdir2.mkdir()

        root_image = tmp_path / "root.jpg"
        sub_image = subdir1 / "sub.png"
        nested_image = subdir2 / "nested.bmp"

        root_image.write_bytes(b"fake")
        sub_image.write_bytes(b"fake")
        nested_image.write_bytes(b"fake")

        result = find_images(tmp_path)
        assert len(result) == 3
        assert sorted(result) == sorted([root_image, sub_image, nested_image])

    def test_find_images_mixed_files(self, tmp_path):
        """Pick images from a mix of different files."""
        (tmp_path / "image.jpg").write_bytes(b"fake")
        (tmp_path / "document.txt").write_text("content")
        (tmp_path / "photo.png").write_bytes(b"fake")
        (tmp_path / "script.py").write_text("print('hello')")
        (tmp_path / "pic.gif").write_bytes(b"fake")

        result = find_images(tmp_path)
        found_extensions = {p.suffix.lower() for p in result}

        assert len(result) == 2
        assert found_extensions == {".jpg", ".png"}

    def test_find_images_nonexistent_path(self):
        """Rglob on non-existent paths returns empty."""
        nonexistent_path = Path("/this/path/does/not/exist")
        result = find_images(nonexistent_path)

        assert result == []


class TestGetTrainTestSplitIndices:
    """Test suite for get_train_test_split_indices function."""

    @pytest.fixture
    def mock_sized_dataset(self):
        """Create mock dataset with __len__ method."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        return dataset

    def test_basic_split_no_shuffle(self, mock_sized_dataset):
        """Split 0.8 w/o shuffle yields two disjoint sets of proper sizes."""
        train_indices, test_indices = get_train_test_split_indices(
            mock_sized_dataset, split_ratio=0.8, shuffle=False
        )

        assert len(train_indices) == 80
        assert len(test_indices) == 20

        assert train_indices == list(range(80))
        assert test_indices == list(range(80, 100))

        assert set(train_indices).isdisjoint(set(test_indices))

        # check if all indices are covered
        all_indices = set(train_indices + test_indices)
        assert all_indices == set(range(100))

    def test_different_split_ratios_w_edge_cases(self, mock_sized_dataset):
        """Splitting in ratio works correctly."""
        cases = [
            (0.0, 0, 100),
            (0.3, 30, 70),
            (0.5, 50, 50),
            (0.7, 70, 30),
            (0.9, 90, 10),
            (1.0, 100, 0)
        ]

        for ratio, expected_train, expected_test in cases:
            train_indices, test_indices = get_train_test_split_indices(
                mock_sized_dataset, split_ratio=ratio, shuffle=False
            )

            assert len(train_indices) == expected_train
            assert len(test_indices) == expected_test

    def test_shuffle_behaviour(self, mock_sized_dataset, mocker):
        """Shuffle is indeed called when shuffle=True is set."""
        mock_shuffle = mocker.patch("numpy.random.shuffle")

        train_indices, test_indices = get_train_test_split_indices(
            mock_sized_dataset, split_ratio=0.8, shuffle=True
        )
        shuffled_args = mock_shuffle.call_args[0][0]

        mock_shuffle.assert_called_once()
        assert shuffled_args == list(range(100))
        assert len(train_indices) == 80
        assert len(test_indices) == 20

    def test_shuffle_variance(self, mock_sized_dataset):
        """Shuffle results differ depending on a seed."""
        np.random.seed(42)
        train1, test1 = get_train_test_split_indices(
            mock_sized_dataset, split_ratio=0.8, shuffle=True
        )

        np.random.seed(123)
        train2, test2 = get_train_test_split_indices(
            mock_sized_dataset, split_ratio=0.8, shuffle=True
        )

        assert train1 != train2 or test1 != test2
        assert len(train1) == len(train2) == 80
        assert len(test1) == len(test2) == 20

    def test_shuffle_reproducibility(self, mock_sized_dataset):
        """Same RNG seeds yield equal splits."""
        np.random.seed(42)
        result1 = get_train_test_split_indices(
            mock_sized_dataset, split_ratio=0.8, shuffle=True
        )
        np.random.seed(123)
        np.random.seed(42)
        result2 = get_train_test_split_indices(
            mock_sized_dataset, split_ratio=0.8, shuffle=True
        )

        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__])
