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


"""ImageFolder-type datasets with custom handling logic.

This module provides classes that implement torch.utils.data.Dataset
interface for single-domain and two-domain image collections.
Both provide the way to hook up Albumentation compose stack into
image instance retrieval, as well as a way to instantiate them from
plain python dictionary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from vision.utils import read_image_rgb

from .types import (
    ConfigurableDataset,
    ImageOnlyTransform,
    SizedDataset
)
from .utils import find_images


class ImageFolder(SizedDataset):
    """ImageFolder implementation tailored to UTKFaces."""

    def __init__(
        self,
        image_paths: list[Path],
        transforms: Optional[ImageOnlyTransform] = None,
    ):
        """Construct from a list of image paths.

        Attributes:
            image_paths: absolute paths to aligned RGB face images
            transforms: Optional, a prebuild Albumentations Compose injected
            from outside
        """
        self.image_paths = list(sorted(image_paths))  # reproducibility
        self.transforms: ImageOnlyTransform | None = transforms

    def __len__(self) -> int:  # noqa: D105
        return len(self.image_paths)

    def __getitem__(self, index):
        """Read->[transform|rescale to {-1;1}]->channel-first transpose.

        Returns:
            (C, H, W) numpy array.
        """
        img_path = self.image_paths[index]
        img = read_image_rgb(img_path)
        if img is None:
            raise ValueError(f"Could not get image at {img_path}")

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        else:
            img = 2. * (img.astype(np.float32) / 255.) - 1.

        return np.transpose(img, (2, 0, 1))

    @classmethod
    def from_config(cls, cfg_dict: dict[str, Any]) -> ImageFolder:
        """Build dataset from config dict.

        Expected dict keys:
        - root:  str | Path - directory containing images
        - aug: ImageOnlyTransform | None - prebuilt Albumentations pipeline
        """
        root = Path(cfg_dict["root"]).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"Root directory does not exist: {root}"
            )

        image_paths = find_images(root)

        transforms: Optional[ImageOnlyTransform] = cfg_dict.get("transforms")
        return cls(image_paths=image_paths, transforms=transforms)


class TwoDomainImageFolder(ConfigurableDataset, SizedDataset):
    """Unpaired dataset that interleaves images from two domains."""
    def __init__(
        self,
        image_paths_A: list[Path],
        image_paths_B: list[Path],
        transforms: Optional[ImageOnlyTransform] = None,
    ):
        """Construct from two lists of image paths.

        Attributes:
            image_paths_A: absolute paths to images from domain A
            image_paths_B: absolute paths to images from domain Bages.
            transforms: Optional, a prebuild Albumentations Compose injected
            from outside
        """
        self.image_paths_A = sorted(image_paths_A)
        self.image_paths_B = sorted(image_paths_B)
        self.transforms: ImageOnlyTransform | None = transforms

    def __len__(self):
        """Length is double the size of a smaller subset."""
        return 2 * min(
            len(self.image_paths_A), len(self.image_paths_B)
        )

    def __getitem__(self, index: int) -> tuple[npt.NDArray[np.float32], int]:
        """Even indices -> (A, 0); odd -> (B, 1).

        Preprocessing is organized as follows:
            Read->[transform|rescale to {-1;1}]->channel-first transpose.
        """
        if index % 2:
            img_path = self.image_paths_B[index // 2]
            label = int(1)
        else:
            img_path = self.image_paths_A[index // 2]
            label = int(0)

        img = read_image_rgb(img_path)
        if img is None:
            raise ValueError(f"Could not get image at {img_path}")

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        else:
            img = 2. * (img.astype(np.float32) / 255.) - 1.

        return np.transpose(img, (2, 0, 1)), label

    @classmethod
    def from_config(cls, cfg_dict: dict[str, Any]) -> TwoDomainImageFolder:
        """Build dataset from config dict.

        Expected dict keys:
        - root_A: str | Path - directory containing images from domain A
        - root_B: str | Path - directory containing images from domain B
        - aug: ImageOnlyTransform | None - prebuilt Albumentations pipeline
        """
        root_A = Path(cfg_dict["root_A"]).expanduser().resolve()
        root_B = Path(cfg_dict["root_B"]).expanduser().resolve()

        if not root_A.exists():
            raise FileNotFoundError(
                f"Root directory does not exist: {root_A}"
            )

        if not root_B.exists():
            raise FileNotFoundError(
                f"Root directory does not exist: {root_B}"
            )

        image_paths_A = find_images(root_A)
        image_paths_B = find_images(root_B)
        transforms: Optional[ImageOnlyTransform] = cfg_dict.get("transforms")

        return cls(
            image_paths_A=image_paths_A,
            image_paths_B=image_paths_B,
            transforms=transforms
        )
