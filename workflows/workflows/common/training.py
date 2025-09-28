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


"""Reusable training utilities across all model types."""

import math
import os
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np
import torch

def setup_amp_scaler(
    enabled: bool,
    device: torch.device,
) -> torch.cuda.amp.grad_scaler.GradScaler | None:
    """Create AMP scaler if enabled and CUDA is available."""
    if enabled and device.type == "cuda":
        return torch.cuda.amp.grad_scaler.GradScaler()
    return None


def metrics_to_str(
    metrics: dict[str, Any],
    epoch: int | None = None,
    prefix: str = ""
) -> str:
    """Assemble single string out of metrics dict."""
    epoch_str = f"Epoch {epoch}: " if epoch is not None else ""
    metrics_strs = [
        f"{prefix}{k}={v:.4f}" if isinstance(v, float)
        else f"{prefix}{k}={v}"
        for k, v in metrics.items()
    ]
    return f"{epoch_str}{' '.join(metrics_strs)}"


def set_global_seed(seed: int = 42):
    """Assign seed to every possible RNG."""
    # 1) Pure Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) PyTorch CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)          # current GPU
    torch.cuda.manual_seed_all(seed)      # all GPUs

    # 4) cuDNN and algorithm determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # This forces certain ops to use deterministic implementations or error out
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # Fallback for older versions
