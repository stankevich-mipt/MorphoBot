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


"""Augmentations for router-utkfaces pipeline."""

from functools import partial, wraps
from typing import Callable, Optional


from configs.router_utkfaces.augmentations import (
    AugmentationConfig
)
import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F


def rpartial(func, /, *rargs, **rkwargs):
    """Binds arguments to the tail positional slots."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        merged = {**rkwargs, **kwargs}
        return func(*args, *rargs, **merged)
    return wrapper


def build_kornia_stack(
    config: AugmentationConfig | None
) -> K.ImageSequential | nn.Module:
    """Build K.ImageSequential pipeline purely from config."""
    if config is None or config.kornia_config is None:
        return nn.Identity()

    transforms: list[nn.Module] = []

    k_cfg = config.kornia_config

    if (cfg := k_cfg.random_horizontal_flip) is not None:
        transforms.append(K.RandomHorizontalFlip(p=cfg.p))

    if (cfg := k_cfg.random_brightness) is not None:
        transforms.append(K.RandomBrightness(
            brightness=cfg.brightness,
            p=cfg.p
        ))

    if (cfg := k_cfg.random_contrast) is not None:
        transforms.append(K.RandomContrast(
            contrast=cfg.contrast,
            p=cfg.p
        ))

    if (cfg := k_cfg.random_hue) is not None:
        transforms.append(K.RandomHue(
            hue=cfg.hue,
            p=cfg.p
        ))

    if (cfg := k_cfg.random_saturation) is not None:
        transforms.append(K.RandomSaturation(
            saturation=cfg.saturation,
            p=cfg.p
        ))

    if (cfg := k_cfg.random_gaussian_blur) is not None:
        transforms.append(K.RandomGaussianBlur(
            kernel_size=cfg.kernel_size,
            sigma=cfg.sigma,
            p=cfg.p
        ))

    if (cfg := k_cfg.random_gaussian_noise) is not None:
        transforms.append(K.RandomGaussianNoise(
            mean=cfg.mean,
            std=cfg.std,
            p=cfg.p
        ))

    if (cfg := k_cfg.random_gamma) is not None:
        transforms.append(K.RandomGamma(
            gamma=cfg.gamma,
            p=cfg.p
        ))

    return K.ImageSequential(
        *transforms,
        same_on_batch=False,
        random_apply=False
    )

@torch.no_grad()
def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    num_classes: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies mixup augmentation over the (N, C, H, W) batch of images.

    Args:
        x: float tensor (N,C,H,W) in [0,1] or normalized
        y: long tensor (N,) or one-hot (N,K).
        alpha: Beta distribution parameter
        (0 -> no mix, larger -> stronger mixing)

    Returns:
        (x_mix, y_mix) - tuple of mixed images + soft labels

    Raises:
        ValueError if num_classes is not specified for
        labels represented with torch.long tensor
    """
    if alpha <= 0:
        if num_classes is not None and y.dtype == torch.long:
            y_onehot = F.one_hot(y, num_classes=num_classes).float()
            return x, y_onehot
        return x, y

    n = x.size(0)
    if n < 2:
        # nothing to mix with
        if num_classes is not None and y.dtype == torch.long:
            y_onehot = F.one_hot(y, num_classes=num_classes).float()
            return x, y_onehot
        return x, y

    # sample lambda ~ Beta(alpha, alpha), shape (N,1,1,1)
    lam = torch.distributions.Beta(
        alpha, alpha
    ).sample(torch.Size((n,))).to(x.device)
    lam_x = lam.view(n, 1, 1, 1)

    # random permutation for pairing
    perm = torch.randperm(n, device=x.device)

    x_mix = lam_x * x + (1.0 - lam_x) * x[perm]

    # prepare labels as probabilities
    if y.dtype == torch.long:
        if num_classes is None:
            raise ValueError("num_classes must be provided when y is class indices.")
        y1 = F.one_hot(y, num_classes=num_classes).float()
        y2 = F.one_hot(y[perm], num_classes=num_classes).float()
    else:
        # assume already one-hot / soft labels
        y1 = y
        y2 = y[perm]

    lam_y = lam.view(n, 1)
    y_mix = lam_y * y1 + (1.0 - lam_y) * y2
    return x_mix, y_mix


def build_mixup(
    config: AugmentationConfig | None
) -> Callable:
    """Get parameterized mixup callable given the config."""
    if config is None or config.mixup_config is None:
        return lambda x: x

    mixup_cfg = config.mixup_config

    return rpartial(
        mixup_batch, mixup_cfg.alpha, mixup_cfg.num_classes)
