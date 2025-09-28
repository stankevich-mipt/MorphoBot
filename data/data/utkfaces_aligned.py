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


"""Interface for preprocessed UTKFaces dataset.

Implementation of torch.utils.data.Dataset
interface for aligned and cropped images from UTKFaces
separated by gender. Full preprocessing routine is
implemented in the series of scripts provided in the current
package. They are executed in following order
    1) download_utkfaces.py
    2) build_utkfaces_manifest_dlib_detector.py
    3) build_alignment_templates_utkfaces.py
    4) build_aligned_dataset_utkfaces.py

For more details on preprocessing steps refer to the original scripts
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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Apply standard ImageNet1K image scaling."""
    return (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


class UTKFacesImageFolder(SizedDataset):
    """ImageFolder implementation tailored to UTKFaces."""

    def __init__(
        self,
        image_paths: list[Path],
        aug: Optional[ImageOnlyTransform] = None,
    ):
        """Construct from a list of image paths.

        Attributes:
            image_paths: absolute paths to aligned RGB face images
            aug: Optional, a prebuild Albumentations Compose injected
            from outside
        """
        self.image_paths = list(sorted(image_paths))  # reproducibility
        self.aug_stack: ImageOnlyTransform | None = aug

    def __len__(self) -> int:  # noqa: D105
        return len(self.image_paths)

    def __getitem__(self, index):
        """Read->[augment]->channel-first normalize and rescale."""
        img_path = self.image_paths[index]
        img = read_image_rgb(img_path)
        if img is None:
            raise ValueError(f"Could not get image at {img_path}")

        if self.aug_stack is not None:
            img = self.aug_stack({"image": img})['image']

        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.
        return normalize_imagenet(img)

    @classmethod
    def from_config(cls, cfg_dict: dict[str, Any]) -> UTKFacesImageFolder:
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

        aug: Optional[ImageOnlyTransform] = cfg_dict.get("aug")
        return cls(image_paths=image_paths, aug=aug)


class UTKFacesDataset(ConfigurableDataset, SizedDataset):
    """UTKFaces paired dataset: interleaves male and female faces."""
    def __init__(
        self,
        image_paths_male: list[Path],
        image_paths_female: list[Path],
        aug: Optional[ImageOnlyTransform] = None,
    ):
        """Construct from two lists of image paths.

        Attributes:
            image_paths_male: absolute paths to aligned male
            face images.
            image_paths_female: absolute paths to aligned female
            face images.
            train: if True, apply augmentations
        """
        self.image_paths_male = sorted(image_paths_male)
        self.image_paths_female = sorted(image_paths_female)
        self.aug_stack: ImageOnlyTransform | None = aug

    def __len__(self):
        """Length is double the size of a smaller class."""
        return 2 * min(
            len(self.image_paths_male), len(self.image_paths_female)
        )

    def __getitem__(self, index: int) -> tuple[npt.NDArray[np.float32], int]:
        """Even indices -> (male, 0); odd -> (female, 1)."""
        if index % 2:
            img_path = self.image_paths_female[index // 2]
            label = int(1)
        else:
            img_path = self.image_paths_male[index // 2]
            label = int(0)

        img = read_image_rgb(img_path)
        if img is None:
            raise ValueError(f"Could not get image at {img_path}")

        if self.aug_stack is not None:
            img = self.aug_stack({"image": img})['image']

        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.
        return normalize_imagenet(img), label

    @classmethod
    def from_config(cls, cfg_dict: dict[str, Any]) -> UTKFacesDataset:
        """Build dataset from config dict.

        Expected dict keys:
        - root_male:  str | Path - directory containing male images
        - root_female: str | Path - directory containing female images
        - aug: ImageOnlyTransform | None - prebuilt Albumentations pipeline
        """
        root_male = Path(cfg_dict["root_male"]).expanduser().resolve()
        root_female = Path(cfg_dict["root_female"]).expanduser().resolve()

        if not root_male.exists():
            raise FileNotFoundError(
                f"Root directory does not exist: {root_male}"
            )

        if not root_female.exists():
            raise FileNotFoundError(
                f"Root directory does not exist: {root_female}"
            )

        image_paths_male = find_images(root_male)
        image_paths_female = find_images(root_female)
        aug: Optional[ImageOnlyTransform] = cfg_dict.get("aug")

        return cls(
            image_paths_male=image_paths_male,
            image_paths_female=image_paths_female,
            aug=aug
        )
