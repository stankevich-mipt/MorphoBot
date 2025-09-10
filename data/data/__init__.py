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


"""Package data.

This is the package initialization file. Currently, it only contains the
license header and does not expose any public symbols or initialization code.
"""

from pathlib import Path
from typing import Any, Type, Union

from .utkfaces_aligned import UTKFacesDataset


SupportedDataset = Union[
    UTKFacesDataset
]


_DATASET_REGISTRY: dict[str, Type[SupportedDataset]] = {
    "utkfaces": UTKFacesDataset,
}


def _get_dataset_class(name: str) -> Type[SupportedDataset]:
    """Retrieve dataset class by name.

    Raises:
        KeyError if name is not registered.
    """
    key = name.lower()
    if key not in _DATASET_REGISTRY:
        raise KeyError(
            f"Dataset class for '{name}' not found in registry."
        )
    return _DATASET_REGISTRY[key]
