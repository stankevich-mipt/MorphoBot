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


"""Utilities for correct type hinting of image transformations."""

from typing import Any, Protocol, Type, TypedDict, TypeVar

import numpy as np
from torch.utils.data import Dataset


class ImgIn(TypedDict):  # noqa
    image: np.ndarray


class ImageOnlyTransform(Protocol):
    """Stuctural typing for A.Compose."""
    def __call__(self, data: ImgIn, /) -> dict[str, Any]: ...  # noqa


T = TypeVar('T', bound='ConfigurableDataset')


class ConfigurableDataset(Protocol):
    """Instantiates from config dict."""
    @classmethod
    def from_config(cls: Type[T], cfg_dict: dict[str, Any]) -> T:
        """Instance from config dict (Hydra compatibility)."""
        ...


class SizedDataset(Dataset):
    """Dataset which explicitly follows Sized protocol.

    Notes:
        - despite the presence of __len__ method is required
        for well-behavedness of PyTorch Dataset, it does not comply
        to the Sized ABC in, hence calling len(Dataset)
        may cause type check warnings. Provided code is a quick fix
        for the issue.
    """
    def __len__(self) -> int: ...  # noqa