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


"""
Provides the implementation of class that does 5 anchor
keypoint alignment with the precomputed templates.
Intended usage is the shared instance pattern: the class
is instantiated once at the start of the app and
is repeatedly addressed throughout its runtime by each request.
Such design pattern reduces the system load caused by repeated JSON I/O.
"""


import cv2
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass


_LEFT_EYE_PTS = tuple(range(36, 42))
_RIGHT_EYE_PTS = tuple(range(42, 48))
_NOSE_TIP_PTS = tuple((30,))
_LEFT_MOUTH_CORNER_PTS = tuple((48,))
_RIGHT_MOUTH_CORNER_PTS = tuple((54, ))


def select_five_from_68(
    pts68: np.ndarray
) -> np.ndarray:
    """
    Build 5 anchors from 68-point landmark array
    following the iBUG/300-W convention.

    Args:
        pts68: float32/64 array of landmark coords.

    Returns:
        np.ndarray: a [5, 2] array representing coordinates
        of left eye, right eye, nose tip, left and right mouth
        corners.
"""

    le = pts68[list(_LEFT_EYE_PTS)].mean(0)
    re = pts68[list(_RIGHT_EYE_PTS)].mean(0)
    nose = pts68[list(_NOSE_TIP_PTS)].mean(0)
    lm = pts68[list(_LEFT_MOUTH_CORNER_PTS)].mean(0)
    rm = pts68[list(_RIGHT_MOUTH_CORNER_PTS)].mean(0)

    return np.stack([le, re, nose, lm, rm], axis=0).astype(np.float32)


def similarity_procrustes(
    src: np.ndarray,
    dst: np.ndarray,
    eps: float = 1e-12
) -> tuple[np.ndarray, float]:
    """Closed-form Procrustes similarity solution mapping src->dst.
    See https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    for additional information.

    Args:
        src: (N, 2) float32/64 source points.
        dst: (N, 2) float32/64 destination poinst.
        eps: stability constant.

    Returns:

        M: (2, 3) float32 affine transformation matrix,
        such that [x_dst, y_dst] ≈ M @ [x_src, y_src, 1]^T
        err: mean L2 distance between points after transform.

    Notes:
        - Requires N >= 2 non-collinear points for a well-posed solution
    """

    if not (src.shape == dst.shape and src.shape[1] == 2):
        raise ValueError("Invalid shape of point cloud array")

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    # 1) Compute centroids
    mu_src = src.mean(axis=0, keepdims=True)
    mu_dst = dst.mean(axis=0, keepdims=True)
    X = src - mu_src
    Y = dst - mu_dst

    # 2) Compute SVD of the covariance matrix
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 3) Uniform scale (trace(S)/ ||X||^2)
    normX2 = np.sum(X * X)
    s = (S.sum()) / (normX2 + eps)

    # 4) Translation
    t = (mu_dst.T - s * R @ mu_src.T).reshape(2)

    # 5) Compose affine 2x3
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = s * R
    M[:, 2] = t

    # 6) Error
    src_h = (s * (R @ src.T)).T + t
    err = float(np.mean(np.linalg.norm(src_h - dst, axis=1)))

    return M.astype(np.float32), err


def invert_similarity(M: np.ndarray) -> np.ndarray:
    """
    Exact inverse of a 2x3 similarity transform (scale*R | t)

    Args:
        M: (2, 3) float32/64

    Returns:
        M_inv: (2, 3) float32, s.t. p_src ≈ M_inv @ [p_dst, 1]^T
    """

    M = np.asarray(M, dtype=np.float64)
    A = M[:, :2]
    b = M[:, 2:3]
    A_inv = np.linalg.inv(A)
    M_inv = np.concatenate([A_inv, -A_inv @ b], axis=1)
    return M_inv.astype(np.float32)


def warp_to_template(
    img_bgr: np.ndarray,
    src5: np.ndarray,
    dst5: np.ndarray,
    dst_shape: tuple[int, int],
    interp: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """
    Compute Procrustes similarity M from
    src5->dst5 and warp the image to the template.

    Returns:
        aligned: (out_size, out_size, 3) or None
        M: (2, 3) forward transform or None
        err: mean alignment error
    """

    M, err = similarity_procrustes(src5, dst5)

    if not np.isfinite(M).all():
        return None, None, float("inf")

    aligned = cv2.warpAffine(
        img_bgr, M, dst_shape, flags=interp,
        borderMode=border_mode, borderValue=border_value
    )

    return aligned, M, err


def warp_back_from_template(
    aligned_bgr: np.ndarray,
    M: np.ndarray,
    dst_shape: tuple[int, int],
    interp: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Apply the inverse to the Procrustes
    optimal transform to map image back to its original orientation.

    Returns:
        back: (dst_shape[0], dst_shape[1], 3) np.array of the original
        image shape.
    """

    H, W = dst_shape
    M_inv = invert_similarity(M)
    back = cv2.warpAffine(
        aligned_bgr, M_inv, (W, H), flags=interp,
        borderMode=border_mode, borderValue=border_value
    )
    return back


@dataclass(frozen=True)
class Template:
    points: np.ndarray
    target_size: int


def load_template(path: str | Path) -> Template:
    obj = json.loads(Path(path).read_text())
    pts = np.array(obj["points"], dtype=np.float32)
    return Template(points=pts, target_size=int(obj["size"]))


@dataclass(frozen=True)
class WarpContext:
    M: np.ndarray
    dst_shape: tuple[int, int]
    interpolation: int = cv2.INTER_LINEAR
    border_mode: int = cv2.BORDER_CONSTANT
    border_value: tuple[int, int, int] = (0, 0, 0)


class FivePointAligner:
    """
    Class that exposes 5 point anchor alignment interface.

    Attributes:
        male_template (Template):
            male alignment template dataclass instance
        female_template (Template):
            female alignment template dataclass instance
        dst_size (int):
            target image size for affine warping.
        default_template (Template): json template for faces that
        cannot be properly classified
    """

    def __init__(
        self,
        male_template: Template,
        female_template: Template,
        default_template: Template | None = None
    ):

        self.male = male_template
        self.female = female_template
        self.default = default_template or male_template

        ts = {
            self.male.target_size,
            self.female.target_size,
            self.default.target_size
        }
        if len(ts) != 1:
            raise ValueError("All templates must share the target size")
        self.target_size = self.male.target_size

    def dst_for_label(self, label: str | None) -> Template:
        """Get destination alignment points for a specific label."""
        lbl = (label or "").strip().lower()
        if lbl in {"male", "m"}:
            return self.male
        elif lbl in {"female", "f"}:
            return self.female
        else:
            return self.default

    def align(
        self,
        img_bgr: np.ndarray,
        pts68: np.ndarray,
        label: str | None = None
    ) -> tuple[np.ndarray | None, WarpContext | None, float]:

        """
        Args
            img_bgr: source image
            pts68: float32/64 array of facial landmarks
            label: image class identifier

        Returns:
            aligned: (target_size, target_size, 3) or None
            ctx: context for future inverse warp

        Note:
            target size is given by self.target_size
            attribute specified at shared instance creation.
        """

        src = select_five_from_68(pts68)
        dst = self.dst_for_label(label)

        aligned, M, err = warp_to_template(
            img_bgr, src, dst.points,
            (dst.target_size, dst.target_size),
        )

        if aligned is None or M is None:
            return None, None, float("inf")

        H, W = img_bgr.shape[:2]
        ctx = WarpContext(
            M=M, dst_shape=(H, W)
        )

        return aligned, ctx, err

    def reverse_warp(
        self,
        img_aligned_bgr: np.ndarray,
        ctx: WarpContext,
    ) -> np.ndarray:

        """
        Args:
            img_aligned_bgr: input image
            ctx: warp context estimated with forward Procrustes pass

        Returns:
            back: (H, W, 3) target of the inverse to the forward map
        """

        back = warp_back_from_template(
            img_aligned_bgr, ctx.M,
            ctx.dst_shape,
        )

        return back
