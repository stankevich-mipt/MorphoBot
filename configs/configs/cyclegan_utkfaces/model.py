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


"""Model parameters for CycleGAN+UTKFaces workflow."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ResNetGeneratorConfig:
    """Nessesary fields to initialize the namesake model.

    Attributes:
        input_nc: number of input channels
        output_nc: number of output channels
        ngf: number of generator filters in first conv layer
        n_residual_blocks: number of residual blocks at the
        middle section of the model
        dropout: sets p in nn.Dropout, if provided
        padding_type: how to handle borderline pixels
        while convolving
    """
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    n_residual_blocks: int = 9
    dropout: Optional[float] = None
    padding_type: Literal["reflect", "replicate", "zero"] = "reflect"


@dataclass
class PatchDiscriminatorConfig:
    """Nessesary fields to initialize the namesake model.

    Attributes:
        input_nc: number of input channels
        ngf: base number of filters
        n_layers: number of downsampling layers after the first convolution
        norm: Normalization type
        use_spectral_norm: if set, apply spectral norm to conv layers
        padding_type: how to handle borderline pixels
        while convolving
    """
    input_nc: int = 3
    ndf: int = 64
    n_layers: int = 3
    norm: Literal["instance", "batch"] = "instance"
    padding_mode: Literal["zeros", "reflect", "replicate"] = "zeros"
    use_spectral_norm: bool = True
