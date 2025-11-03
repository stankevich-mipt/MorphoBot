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

import cv2
import dlib
import numpy as np
from PIL import Image
import torch
from vision.procrustes_aligner_5pt import Template


def rect_to_xyxy(rect: dlib.rectangle) -> tuple[int, int, int, int]:  # type: ignore
    """Rectangle object from dlib to tuple of ints."""
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


def harmonize_and_blend(
    processed: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    feather_width: int = 10
):
    """Adjusts color of the processed image region under mask and feathers the seams.

    Args:
        processed: processed image
        original: original image
        mask: binary mask (1 for pasted/processed region, 0 for background)
        feather_width: feathering radius in pixels
    """
    proc_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)

    matched = proc_lab.copy()

    for i in range(3):
        proc_vals = proc_lab[..., i][mask[..., i]]
        orig_vals = orig_lab[..., i][mask[..., i]]

        p_mean, p_std = proc_vals.mean(), proc_vals.std()
        o_mean, o_std = orig_vals.mean(), orig_vals.std()

        matched[..., i][mask[..., i]] = np.clip(
            ((proc_vals - p_mean) / (p_std + 1e-5)) * (o_std + 1e-5) + o_mean, 0, 255)

    harmonized = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_LAB2BGR)

    feather_size = max(1, feather_width)
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (feather_size*2+1, feather_size*2+1), 0)
    soft_mask = np.clip(soft_mask, 0, 1)

    print(harmonized, original)

    out = harmonized.astype(np.float32) * soft_mask + original.astype(np.float32) * (1 - soft_mask)
    return out.astype(np.uint8)
