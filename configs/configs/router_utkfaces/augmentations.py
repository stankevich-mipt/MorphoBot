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


"""Augmentation parameters for router+UTKFaces workflow."""

from dataclasses import dataclass, field
from typing import Optional

from serde import deserialize


@dataclass
class RandomHorizontalFlipConfig:  # noqa: D101
    p: float = 0.5


@deserialize
@dataclass
class RandomBrightnessConfig:  # noqa: D101
    brightness: tuple[float, float] = (0.85, 1.15)
    p: float = 0.5


@deserialize
@dataclass
class RandomContrastConfig:  # noqa: D101
    contrast: tuple[float, float] = (0.85, 1.15)
    p: float = 0.5


@deserialize
@dataclass
class RandomHueConfig:  # noqa: D101
    hue: tuple[float, float] = (-0.2, 0.2)
    p: float = 0.5


@deserialize
@dataclass
class RandomSaturationConfig:  # noqa: D101
    saturation: tuple[float, float] = (0.85, 1.15)
    p: float = 0.5


@deserialize
@dataclass
class RandomGaussianBlurConfig:  # noqa: D101
    kernel_size: tuple[int, int] = (3, 3)
    sigma: tuple[float, float] = (0.5, 3.0)
    p: float = 0.2


@deserialize
@dataclass
class RandomGaussianNoiseConfig:  # noqa: D101
    mean: float = 0.0
    std: float = 20. / 255.
    p: float = 0.3


@deserialize
@dataclass
class RandomGammaConfig:  # noqa: D101
    gamma: tuple[float, float] = (0.85, 1.15)
    p: float = 0.3


@deserialize
@dataclass
class KorniaConfig:
    """Define Cornia stack params with dataclasses."""
    random_horizontal_flip: (
        Optional[RandomHorizontalFlipConfig]
    ) = field(default_factory=RandomHorizontalFlipConfig)
    random_brightness: (
        Optional[RandomBrightnessConfig]
    ) = field(default_factory=RandomBrightnessConfig)
    random_contrast: (
        Optional[RandomContrastConfig]
    ) = field(default_factory=RandomContrastConfig)
    random_hue: (
        Optional[RandomHueConfig]
    ) = field(default_factory=RandomHueConfig)
    random_saturation: (
        Optional[RandomSaturationConfig]
    ) = field(default_factory=RandomSaturationConfig)
    random_gaussian_blur: (
        Optional[RandomGaussianBlurConfig]
    ) = field(default_factory=RandomGaussianBlurConfig)
    random_gaussian_noise: (
        Optional[RandomGaussianNoiseConfig]
    ) = field(default_factory=RandomGaussianNoiseConfig)
    random_gamma: (
        Optional[RandomGammaConfig]
    ) = None


@deserialize
@dataclass
class MixupConfig:
    """Mixup augmentation parameters."""
    alpha: float = 0.4
    num_classes: int = 2


@deserialize
@dataclass
class AugmentationConfig:
    """Portfolio class for augmenations configs used in a pipeline."""
    kornia_config: Optional[KorniaConfig] = field(
        default_factory=KorniaConfig
    )
    mixup_config: Optional[MixupConfig] = field(
        default_factory=MixupConfig
    )
