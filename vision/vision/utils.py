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

"""Utility functions for data processing aggregated into a single script.

Index:
    - read_image_bgr
    - read_bytes_bgr
    - read_image_rgb
    - bbox_size
    - bbox_aspect_ratio
    - boox_area
    - bbox_is_plausible
    - draw_preview
    - grid preview

"""


from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(root: Path) -> list[Path]:
    """Search for images within dir recursively."""
    return [p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS]


def read_bytes_bgr(data: bytes) -> npt.NDArray[np.uint8]:
    """Convert binary buffer with image data to numpy array."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return np.asarray(img)


def read_image_bgr(path: Path) -> Optional[npt.NDArray[np.uint8]]:
    """Load the image at path as numpy array (H, W, 3) in BGR."""
    img = cv2.imread(str(path))
    return np.asarray(img)


def read_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Load the image at path as numpy array (H, W, 3) in RGB."""
    img = read_image_bgr(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bbox_size(x1: int, y1: int, x2: int, y2: int):
    """Get the absolute values of bbox height and width."""
    return max(0, x2 - x1), max(0, y2 - y1)


def bbox_aspect_ratio(w: int, h: int, eps: float = 1e-6):
    """Get the bbox aspect ratio."""
    a = (max(w, h) + eps) / (min(w, h) + eps)
    return a


def bbox_area(w: int, h: int):
    """Get the area covered by (w, h) bounding box."""
    return w * h


def bbox_is_plausible(
    bbox: list[int],
    W: int,
    H: int,
    min_side: int = 48,
    max_ratio: float = 2.0,
    min_area_frac: float = 0.01,
    max_area_frac: float = 0.85,
) -> tuple[bool, str]:
    """Check whether bbox matches the set of filters.

    Returns:
        (status, reason), with the latter being "ok" if
        the criteria are met, and brief rejection cause
        otherwise.

    """
    x1, y1, x2, y2 = bbox
    w, h = bbox_size(x1, y1, x2, y2)
    if w < min_side or h < min_side:
        return False, "too_small"
    ar = bbox_aspect_ratio(w, h)
    if ar > max_ratio:
        return False, "bad_aspect"
    area = bbox_area(w, h)
    img_area = W * H
    if area < min_area_frac * img_area:
        return False, "area_too_small"
    if area > max_area_frac * img_area:
        return False, "area_too_large"
    return True, "ok"


def draw_preview(
    img: np.ndarray,
    bbox: Optional[tuple[int, int, int, int]],
    pts68: Optional[np.ndarray]
) -> np.ndarray:
    """Draw bbox rectangle and facial keypoints over image copy."""
    vis = img.copy()
    if bbox is not None:

        left, top, right, bottom = bbox

        cv2.rectangle(
            vis,
            (left, top),
            (right, bottom),
            (255, 0, 0),
            2
        )

    if pts68 is not None:
        for (x, y) in pts68.astype(int):
            cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

    return vis


def grid_preview(
    images: list[np.ndarray],
    cols: int = 4,
    cell_size: tuple[int, int] = (256, 256)
) -> Optional[np.ndarray]:
    """Create an image grid from the list of input images."""
    if not images:
        return None

    rows = (len(images) + cols - 1) // cols
    H, W = cell_size[1], cell_size[0]

    grid = np.full((rows * H, cols * W, 3), 255, np.uint8)

    for i, img in enumerate(images[:rows*cols]):
        r, c = divmod(i, cols)
        thumb = cv2.resize(img, (W, H))
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = thumb

    return grid
