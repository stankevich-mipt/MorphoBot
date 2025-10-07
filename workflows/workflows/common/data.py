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


"""Workflow-agnostic data retrieval/processing utilities."""

from mlflow_registry import configure_mlflow
from mlflow_registry.search import search_runs_by_tags
from mlflow_registry.tags import TagKeys, Type
import torch


def fetch_supported_datasets_mlflow() -> set[str]:
    """Get names of datasets that are registered in MLFow."""
    configure_mlflow()
    dataset_runs = search_runs_by_tags(
        {TagKeys.TAG_TYPE: Type.DATASET},
    )

    dataset_names: set[str] = set()
    for r in dataset_runs:
        if (name := r.tags.get("name", None)) is not None:
            dataset_names.add(name)

    return dataset_names


class TorchShuffler:
    """PyTorch adapter for data.utils.Shuffler protocol."""
    def __init__(self, gen: torch.Generator):
        """Instantiate with Pytorch RNG."""
        self._gen = gen

    def shuffle(self, x: list[int]) -> None:
        """Get shuffle ids from torch.randperm."""
        idx = torch.randperm(len(x), generator=self._gen).tolist()
        x[:] = [x[i] for i in idx]
