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

"""Utilities for backend-agnostic image translation service."""

import io

import dlib
import numpy as np
from PIL import Image
import torch
from vision.procrustes_aligner_5pt import Template


def rect_to_xyxy(rect: dlib.rectangle) -> tuple[int, int, int, int]:  # type: ignore
    """Rectange object from dlib to tuple of ints."""
    return rect.left(), rect.top(), rect.right(), rect.bottom()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:  # noqa: D103
    tensor = torch.clamp(tensor, 0, 1)

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    numpy_img = tensor.detach().cpu().numpy()
    numpy_img = np.transpose(numpy_img, (1, 2, 0))
    numpy_img = (numpy_img * 255).astype(np.uint8)

    return Image.fromarray(numpy_img)


def pil_to_bytes(  # noqa: D103
    image: Image.Image, format: str = "JPEG", quality: int = 95
) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def create_alignment_template(size: int = 256):
    """Provide the default template for face alignment."""
    points = np.array([
        [0.35 * size, 0.38 * size],
        [0.65 * size, 0.38 * size],
        [0.50 * size, 0.52 * size],
        [0.40 * size, 0.70 * size],
        [0.60 * size, 0.70 * size],
    ], dtype=np.float32)

    return Template(points=points, target_size=size)
