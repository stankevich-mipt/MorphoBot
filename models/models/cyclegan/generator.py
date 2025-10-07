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

"""ResNet-based Generator from the original CycleGAN paper."""

from typing import Optional

from configs.cyclegan_utkfaces import ResNetGeneratorConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with convolutions."""
    def __init__(
        self,
        dim: int,
        dropout: Optional[float] = None,
        padding_mode: str = "reflect"
    ):
        """Initialize by assembling layers into nn.Sequential."""
        super().__init__()

        self.dropout = dropout

        self.skip_connection = nn.Identity()

        layers = []

        layers += self._get_norm_conv_block(dim, dim, 3, padding_mode)
        if self.dropout:
            layers += [nn.Dropout(self.dropout)]

        layers += self._get_conv_block(
            dim, dim, 3, padding_mode, use_activation=False)

        self.conv_block = nn.Sequential(*layers)

    def _get_norm_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding_mode: str
    ) -> list[nn.Module]:
        return self._get_conv_block(
            in_channels, out_channels,
            kernel_size, padding_mode=padding_mode,
            use_activation=True, use_norm=True
        )

    def _get_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding_mode: str,
        use_norm: bool = True,
        use_activation: bool = True
    ) -> list[nn.Module]:

        layers = []
        pad = kernel_size // 2

        if padding_mode == "reflect":
            layers += [nn.ReflectionPad2d(pad)]
            conv_padding = 0
        elif padding_mode == "replicate":
            layers += [nn.ReplicationPad2d(pad)]
            conv_padding = 0
        elif padding_mode == "zeros":
            conv_padding = pad
        else:
            raise NotImplementedError(
                f"Padding mode {padding_mode} is not supported."
            )

        layers += [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=conv_padding, padding_mode=padding_mode, bias=not use_norm
        )]

        if use_norm:
            layers += [nn.InstanceNorm2d(out_channels)]

        if use_activation:
            layers += [nn.ReLU(inplace=True)]

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.skip_connection(x) + self.conv_block(x)


def build_resnet_generator(
    config: ResNetGeneratorConfig,
) -> nn.Module:
    """Factory function that builds model from config."""
    # initial conv layer

    input_nc = config.input_nc
    ngf = config.ngf

    model = [
        nn.ZeroPad2d(3),
        nn.Conv2d(
            input_nc, ngf, kernel_size=7,
            padding=0, padding_mode=config.padding_mode,
            bias=False
        ),
        nn.InstanceNorm2d(config.ngf),
        nn.ReLU(inplace=True)
    ]

    mult = 1
    for _ in range(2):
        model += [
            nn.Conv2d(
                mult * ngf, mult * ngf * 2,
                kernel_size=3, stride=2, padding=1,
                padding_mode=config.padding_mode,
                bias=False
            ),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        mult *= 2

    for _ in range(config.n_residual_blocks):
        model += [ResidualBlock(
            ngf * mult, dropout=config.dropout,
            padding_mode=config.padding_mode
        )]

    for _ in range(2):
        model += [
            nn.ConvTranspose2d(
                mult * ngf, (mult // 2) * ngf,
                kernel_size=3, stride=2, padding=1,
                output_padding=1, bias=False,
                padding_mode=config.padding_mode
            ),
            nn.InstanceNorm2d((mult // 2) * ngf),
            nn.ReLU(inplace=True)
        ]
        mult //= 2

    model += [
        nn.ZeroPad2d(3),
        nn.Conv2d(ngf, config.output_nc, kernel_size=7, padding=0),
    ]

    return nn.Sequential(*model)


class ResNetGenerator(nn.Module):
    """Resnet-based generator for CycleGAN.

    Architecture:
    - Encoder: 3 conv layers with downsampling
    - Residual stack: n_residual_blocks ResidualBlock's
    - Decoder: 3 deconv layers with upsampling
    """

    def __init__(
        self, cfg: ResNetGeneratorConfig
    ):
        """Initialize by building with factory function."""
        super().__init__()
        self.cfg: ResNetGeneratorConfig = cfg
        self.model = build_resnet_generator(cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights."""
        classname = m.__class__.__name__
        if (hasattr(m, "weight")
            and (
                classname.find("Conv") != -1
                or classname.find("Linear") != -1
        )):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
            or classname.find("InstanceNorm2d") != -1
        ):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass is linear, as model is sequential.

        Args:
            x: Input tensor of shape
            (batch_size, input_nc, height, width)

        Returns:
            Generated tensor of shape
            (batch_size, output_nc, height, width)
        """
        return self.model(x)
