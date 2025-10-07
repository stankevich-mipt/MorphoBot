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


"""Input/output utilities for router_utkfaces workflow."""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    **extra_data: Any
) -> None:
    """Save model checkpoint with optional optimizer state and extra data."""
    checkpoint = {
        "state_dict": model.state_dict(),
        **extra_data
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None
) -> dict[str, Any]:
    """Load checkpoint and restore model/optimizer state."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
