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


"""Common utilites for all used datasets."""

from pathlib import Path

import numpy as np

from .types import SizedDataset


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(root: Path) -> list[Path]:
    """Search for images within dir recursively."""
    return [p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS]


def get_train_test_split_indices(
    dataset: SizedDataset,
    split_ratio: float,
    shuffle: bool = False
) -> tuple[list, list]:
    """Generate train/test indices split from dataset, shuffle if needed."""
    total_elements = len(dataset)
    ids = list(range(total_elements))
    if shuffle:
        np.random.shuffle(ids)
    split = int(np.floor(split_ratio * total_elements))

    return ids[:split], ids[split:]
