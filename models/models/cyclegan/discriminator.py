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

"""Iteration over PatchGAN Discriminator from CycleGAN paper."""

from configs.cyclegan_utkfaces import PatchDiscriminatorConfig
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm


def _maybe_sn(
    module: nn.Module, use_spectral_norm: bool
) -> nn.Module:
    """Apply spectral normalization over module if specified."""
    return spectral_norm(module) if use_spectral_norm else module


def build_patch_discriminator(
    config: PatchDiscriminatorConfig
) -> nn.Module:
    """Factory function that builds model from config."""
    input_nc = config.input_nc
    n_layers = config.n_layers
    ndf = config.ndf
    norm = config.norm
    padding_mode = config.padding_mode
    use_spectral_norm = config.use_spectral_norm

    def norm_layer(ch: int) -> nn.Module:
        if norm == "instance":
            return nn.InstanceNorm2d(
                ch, affine=False, track_running_stats=False
            )
        elif norm == "batch":
            return nn.BatchNorm2d(ch)
        else:
            raise ValueError(f"Unknown norm: {norm}")

    sequence = []

    kernel_size = 4
    pad_value = 1

    conv1 = nn.Conv2d(
        input_nc, ndf, kernel_size=kernel_size,
        stride=2, padding=pad_value, padding_mode=padding_mode
    )

    sequence += [
        _maybe_sn(conv1, use_spectral_norm),
        nn.LeakyReLU(0.2, inplace=True)
    ]

    mult = 1
    for n in range(1, n_layers):
        mult_prev = mult
        mult = min(2 ** n, 8)
        conv = nn.Conv2d(
            ndf * mult_prev, ndf * mult,
            kernel_size=kernel_size, padding=pad_value,
            stride=2, padding_mode=padding_mode
        )
        sequence += [
            _maybe_sn(conv, use_spectral_norm),
            norm_layer(ndf * mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

    mult_prev = mult
    mult = min(2 ** n_layers, 8)
    conv = nn.Conv2d(
        ndf * mult_prev, ndf * mult,
        kernel_size=kernel_size, padding=1,
        stride=1, padding_mode=padding_mode
    )

    sequence += [
        _maybe_sn(conv, use_spectral_norm),
        norm_layer(ndf * mult),
        nn.LeakyReLU(0.2, inplace=True)
    ]

    conv_out = nn.Conv2d(
        ndf * mult, 1, kernel_size=kernel_size,
        stride=1, padding_mode=padding_mode
    )

    sequence += [_maybe_sn(conv_out, use_spectral_norm)]

    return nn.Sequential(*sequence)


class PatchDiscriminator(nn.Module):
    """PatchGAN with 70x70 receptive field."""
    def __init__(self, cfg: PatchDiscriminatorConfig):
        """Initialize with config dataclass."""
        self.cfg = cfg
        self.model = build_patch_discriminator(cfg)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)  # type: ignore
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass is linear, as model is sequential.

        Args:
            x: Input tensor (N, C, H, W)

        Returns:
            Patch logits (N, 1, H', W'), where each location corresponds
            to a receptive field-level signal
        """
        return self.model(x)
