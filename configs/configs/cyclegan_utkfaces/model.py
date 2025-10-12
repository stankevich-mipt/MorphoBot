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

from dataclasses import dataclass, field
from typing import Literal, Optional

from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec
from mlflow_registry.tags import Role, Stage, TagKeys, Type
import numpy as np


def _generator_mlflow_signature_factory():
    return ModelSignature(
        inputs=Schema([
            TensorSpec(
                type=np.dtype(np.float32),
                shape=(-1, 3, 64, 64),
                name="source domain images"
            ),
        ]),
        outputs=Schema([
            TensorSpec(
                type=np.dtype(np.float32),
                shape=(-1, 3, 64, 64),
                name="target domain images"
            ),
        ])
    )

def _discriminator_mlflow_signature_factory():
    return ModelSignature(
        inputs=Schema([
            TensorSpec(
                type=np.dtype(np.float32),
                shape=(-1, 3, 64, 64),
                name="real or generated image"
            ),
        ]),
        outputs=Schema([
            TensorSpec(
                type=np.dtype(np.float32),
                shape=(-1, 1),
                name="unnormalized real/fake quantifier"
            ),
        ])
    )


TAGS_RESNET_GENERATOR_DEV = {
    TagKeys.TAG_TYPE: Type.MODEL,
    TagKeys.TAG_ROLE: Role.IMAGE_TO_IMAGE_TRANSLATION_MODEL,
    TagKeys.TAG_STAGE: Stage.DEVELOPMENT,
    "framework": "cyclegan",
    "component": "generator",
    "architecture": "resnet",
}


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
    padding_mode: Literal["reflect", "replicate", "zeros"] = "zeros"

    def get_model_signature(self) -> ModelSignature:  # noqa: D102
        return _generator_mlflow_signature_factory()

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

    def get_model_signature(self) -> ModelSignature:  # noqa: D102
        return _discriminator_mlflow_signature_factory()
