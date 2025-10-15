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

"""Dual dataloader wrapper for CycleGAN training.

Combines two Dataloaders into a single iterator that
returns dictionaries with keys 'A' and 'B', gracefully
handling different dataset sizes.
"""

from contextlib import nullcontext
from itertools import cycle
import logging
from typing import Any, Iterator, Literal

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class PairedDataloader:
    """Wrapper that combines two dataloaders for CycleGAN training."""

    def __init__(
        self,
        dataloader_A: DataLoader,
        dataloader_B: DataLoader,
        strategy: Literal[
            "cycle_shorter",
            "stop_at_shorter",
            "resample_shorter"
        ] = "cycle_shorter"
    ):
        """Initialize from pair of dataloaders.

        Args:
            dataloader_A: DataLoader from domain A
            dataloader_B: DataLoader from domain B
            strategy: How to handle different lengths:
                - "cycle_shorter": Cycle the shorter dataset (default)
                - "stop_at_shorter": Stop when shorter loader is exhausted
                - "resample_shorter": resample the shorter loader when exhausted
        """
        self.dataloader_A = dataloader_A
        self.dataloader_B = dataloader_B
        self.strategy = strategy

        self.len_A = len(dataloader_A)
        self.len_B = len(dataloader_B)

        if self.len_A <= self.len_B:
            self.shorter_loader = dataloader_A
            self.longer_loader = dataloader_B
            self.shorter_key = 'A'
            self.longer_key = 'B'
        else:
            self.shorter_loader = dataloader_B
            self.longer_loader = dataloader_A
            self.shorter_key = 'B'
            self.longer_key = 'A'

        logger.info("Initialized paired dataloader:")
        logger.info(f"  Dataloader A: {self.len_A} batches")
        logger.info(f"  Dataloader B: {self.len_B} batches")
        logger.info(f"  Will iterate for {len(self)} batches per epoch")

    def __len__(self) -> int:
        """Return the number of batches per epoch based on strategy."""
        if self.strategy == "stop_at_shorter":
            return min(self.len_A, self.len_B)
        else:
            return max(self.len_A, self.len_B)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Return iterator that yields batch dictionaries."""
        if self.strategy == "cycle_shorter":
            return self._cycle_shorter_iter()
        elif self.strategy == "stop_at_shorter":
            return self._stop_at_shorter_iter()
        elif self.strategy == "resample_shorter":
            return self._resample_shorter_iter()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _cycle_shorter_iter(self) -> Iterator[dict[str, Any]]:
        """Iterator that cycles the shortest dataset."""
        if self.len_A <= self.len_B:
            iter_A = cycle(self.dataloader_A)
            iter_B = iter(self.dataloader_B)
        else:
            iter_A = iter(self.dataloader_A)
            iter_B = cycle(self.dataloader_B)

        for _ in range(len(self)):
            batch_A, batch_B = next(iter_A), next(iter_B)
            yield {
                "A": batch_A,
                "B": batch_B
            }

    def _stop_at_shorter_iter(self) -> Iterator[dict[str, Any]]:
        """Iterator that stops when shorter dataset is exhausted."""
        iter_A = iter(self.dataloader_A)
        iter_B = iter(self.dataloader_B)

        for batch_A, batch_B in zip(iter_A, iter_B):
            yield {
                "A": batch_A,
                "B": batch_B
            }

    def _resample_shorter_iter(self) -> Iterator[dict[str, Any]]:
        """Iterator that resamples shorter datasets."""
        longer_iter = iter(self.longer_loader)
        shorter_iter = iter(self.shorter_loader)

        for _ in range(len(self)):

            batch_longer = next(longer_iter)
            batch_shorter = None

            try:
                batch_shorter = next(shorter_iter)
            except StopIteration:
                shorter_iter = iter(self.shorter_loader)
                batch_shorter = next(shorter_iter)

            if self.longer_key == "A":
                batch_A = batch_longer
                batch_B = batch_shorter
            else:
                batch_A = batch_shorter
                batch_B = batch_longer

            yield {
                "A": batch_A,
                "B": batch_B
            }


class CycleGANStreamDataset(IterableDataset):
    """Streaming dataset of generated images.

    Streams batches from a base loader, runs generator,
    to produce translated images, and yields the result.
    All tensors are yielded on CPU by default.
    """

    def __init__(
        self,
        base_loader: PairedDataloader,
        G_A2B: torch.nn.Module,
        G_B2A: torch.nn.Module,
        total_samples: int = int(1e4),
        device: str = "cpu",
        autocast_dtype: torch.dtype | None = torch.float16,
        return_on_cpu: bool = True
    ):
        """Initialize with pair of models and base dataloader."""
        super().__init__()
        self.base_loader = base_loader
        self.G_A2B = G_A2B
        self.G_B2A = G_B2A
        self.total_samples = total_samples
        self.device = "cuda" if "cuda" in device else "cpu"
        self.autocast_dtype = autocast_dtype
        self.return_on_cpu = return_on_cpu

        self.G_A2B.eval()
        self.G_B2A.eval()

    def _to_device(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Transfer dict of tensors to device."""
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _to_cpu(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Transfer dict of tensors to cpu."""
        return {k: v.to(torch.device('cpu'), non_blocking=True) for k, v in batch.items()}

    @torch.inference_mode()
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield within 'for' cycle over the base loader after processing."""
        ctx_cast = (
            autocast(device_type=self.device, dtype=self.autocast_dtype)
            if self.autocast_dtype else nullcontext()
        )

        samples_fetched = 0

        for batch in self.base_loader:

            if samples_fetched >= self.total_samples:
                break

            batch = self._to_device(batch)

            with ctx_cast:
                fake_batch = {
                    "fake_A": self.G_B2A(batch["B"]),
                    "fake_B": self.G_A2B(batch["A"])
                }

            if self.return_on_cpu:
                cpu_batch = self._to_cpu(batch)
                del batch["A"]
                del batch["B"]
                batch = cpu_batch

                fake_cpu_batch = self._to_cpu(fake_batch)
                del fake_batch["fake_A"]
                del fake_batch["fake_B"]
                fake_batch = fake_cpu_batch

            samples_fetched += batch["A"].shape[0]

            yield {**batch, **fake_batch}
