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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec
from mlflow_registry.tags import Role, Stage, TagKeys, Type
import numpy as np
import numpy.typing as npt


TAGS_RESNET_GENERATOR_DEV = {
    TagKeys.TAG_TYPE: Type.MODEL,
    TagKeys.TAG_ROLE: Role.IMAGE_TO_IMAGE_TRANSLATION_MODEL,
    TagKeys.TAG_STAGE: Stage.DEVELOPMENT,
    "framework": "cyclegan",
    "component": "generator",
    "architecture": "resnet",
}


@dataclass
class GeneratorSignature:
    """Required fields to assemble ModelSignature for generator."""
    input_type: npt.DTypeLike = np.dtype(np.float32)
    input_shape: tuple[int, int, int, int] = (-1, 3, 128, 128)
    input_name: str = "source domain images"
    output_type: npt.DTypeLike = np.dtype(np.float32)
    output_shape: tuple[int, int, int, int] = (-1, 3, 128, 128)
    output_name: str = "target domain images"

    def __post_init__(self):
        """Assemble MLflow signature from model fields."""
        self.mlflow_signature_spec = ModelSignature(
            inputs=Schema([
                TensorSpec(
                    type=self.input_type,
                    shape=self.input_shape,
                    name=self.input_name
                ),
            ]),
            outputs=Schema([
                TensorSpec(
                    type=self.output_type,
                    shape=self.output_shape,
                    name=self.output_name
                ),
            ])
        )

    def update_with_mlflow_signature_dict(self, signature_dict) -> GeneratorSignature:
        """Inverts self.mlflow_signature_spec.to_dict()."""
        signature = ModelSignature.from_dict(signature_dict)

        input_name, input_spec = next(
            iter(signature.inputs.input_dict().items()))
        self.input_name = input_name

        if not isinstance(input_spec, TensorSpec):
            raise TypeError(
                f"Expected TensorSpec as first member of input schema: got {type(input_spec)}")

        self.input_type = input_spec.type
        self.input_shape = input_spec.shape

        output_name, output_spec = next(
            iter(signature.outputs.input_dict().items()))
        self.output_name = output_name

        if not isinstance(output_spec, TensorSpec):
            raise TypeError(
                f"Expected TensorSpec as first member of output schema: got {type(output_spec)}")

        self.output_type = output_spec.type
        self.output_shape = output_spec.shape

        return self


@dataclass
class DiscriminatorSignature:
    """Required fields to assemble ModelSignature for discriminator."""
    input_type: npt.DTypeLike = np.dtype(np.float32)
    input_shape: tuple[int, int, int, int] = (-1, 3, 128, 128)
    input_name: str = "real or generated image"
    output_type: npt.DTypeLike = np.dtype(np.float32)
    output_shape: tuple[int] = (-1,)
    output_name: str = "unnormalized real/fake quantifier"

    def __post_init__(self):
        """Assemble MLflow signature from model fields."""
        self.mlflow_signature_spec = ModelSignature(
            inputs=Schema([
                TensorSpec(
                    type=self.input_type,
                    shape=self.input_shape,
                    name=self.input_name
                ),
            ]),
            outputs=Schema([
                TensorSpec(
                    type=self.output_type,
                    shape=self.output_shape,
                    name=self.output_name
                ),
            ])
        )

    def update_with_mlflow_signature_dict(self, signature_dict) -> DiscriminatorSignature:
        """Inverts self.mlflow_signature_spec.to_dict()."""
        signature = ModelSignature.from_dict(signature_dict)

        input_name, input_spec = next(
            iter(signature.inputs.input_dict().items()))
        self.input_name = input_name

        if not isinstance(input_spec, TensorSpec):
            raise TypeError(
                f"Expected TensorSpec as first member of input schema: got {type(input_spec)}")

        self.input_type = input_spec.type
        self.input_shape = input_spec.shape

        output_name, output_spec = next(
            iter(signature.outputs.input_dict().items()))
        self.output_name = output_name

        if not isinstance(output_spec, TensorSpec):
            raise TypeError(
                f"Expected TensorSpec as first member of output schema: got {type(output_spec)}")

        self.output_type = output_spec.type
        self.output_shape = output_spec.shape

        return self


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
    signature: GeneratorSignature = field(default_factory=GeneratorSignature)


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
    signature: DiscriminatorSignature = field(default_factory=DiscriminatorSignature)
