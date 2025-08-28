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

"""Facial keypoint detector that utilizes dlib.

Provides the nessesary functional to estimate 68-point based
facial landmarks on tight facial crops.
The latter are obtained with innate dlib HOG-based detector.
"""

import os
from typing import Optional

import dlib
import numpy as np


class Dlib68Detector:
    """Class implementing dlib-based keypoint detection."""

    def __init__(
        self,
        predictor_path: str | os.PathLike,
        detector_kind: str = 'hog',
        upsample_times: int = 1,
        margin_ratio: tuple[float, float] = (0.10, 0.25),
    ):
        """Instatiate object with preloaded predictor weights.

        Attributes:
        predictor_path (str | os.PathLike): absolute path to dlib
        facial keypoint predictor model
        detector_kind (str, optional, default='hog'): the type of
        face detector to use while processing the image
        upsample_times (int, optional, default=1): upscale the
        faces to the multiple of their size if they are too small
        margin_ratio (tuple[float, float], optional, default=(0.1, 0.25)):
        margins to expand the bounding box to include chin/hair region
        - dlib generally provides quite tight crops that omit these details
        """
        self.upsample_times = int(upsample_times)
        self.margin_ratio = margin_ratio

        if detector_kind == 'hog':
            self.detector = dlib.get_frontal_face_detector()  # type: ignore
        else:
            raise ValueError("Only 'hog' detector is supported at the moment")

        self.shape_predictor = dlib.shape_predictor(predictor_path)  # type: ignore

    def _rect_to_xyxy(
        self,
        rect: dlib.rectangle  # type: ignore
    ) -> tuple[int, int, int, int]:

        return rect.left(), rect.top(), rect.right(), rect.bottom()

    def _xyxy_to_rect(
            self, x1: int, y1: int, x2: int, y2: int) -> dlib.rectangle:  # type: ignore

        return dlib.rectangle(  # type: ignore
            left=int(x1), top=int(y1), right=int(x2), bottom=int(y2))

    def _clip_box(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img: np.ndarray
    ):
        """Clip the bbox so it fits the image."""
        h, w = img.shape[0], img.shape[1]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)

        return x1, y1, x2, y2

    def expand_with_margin(
        self,
        rect: dlib.rectangle,  # type: ignore
        img: np.ndarray
    ) -> dlib.rectangle:  # type: ignore
        """Stretch the bbox along each dimension with uniform factor.

        Args:
            rect: a rectangle that tightly
            frames the detected face
            img_bgr (np.ndarray, dtype=np.uint8): numpy array
            representing the image being processed

        Returns:
            bbox, expanded with ratio provided by self.margin_ratio
        """
        x1, y1, x2, y2 = self._rect_to_xyxy(rect)
        w, h = x2 - x1, y2 - y1
        dx, dy = int(self.margin_ratio[0] * w), int(self.margin_ratio[1] * h)

        x1m, y1m, x2m, y2m = self._clip_box(
            x1 - dx, y1 - dy, x2 + dx, y2 + dy, img)

        return self._xyxy_to_rect(x1m, y1m, x2m, y2m)

    def detect_one(
        self, img_bgr: np.ndarray
    ) -> Optional[dlib.rectangle]:  # type: ignore
        """Return the facial bbox with maximum area if there is one.

        Args:
            img_bgr (np.ndarray, dtype=np.uint8): numpy array
            representing the image being processed

        Returns:
            Optional[dlib.rectangle]: a rectangle representing the largest
            face bbox.

        """
        detections = self.detector(img_bgr, self.upsample_times)
        if len(detections) == 0:
            return None

        rects = list(detections)

        areas = [
            (r.right() - r.left()) * (r.bottom() - r.top()) for r in rects]
        return rects[int(np.argmax(areas))]

    def landmarks68(
        self,
        img_bgr: np.ndarray,
        rect: dlib.rectangle  # type: ignore
    ) -> np.ndarray:
        """Extract facial landmarks in 68pt format with dlib detector.

        Args:
            img_bgr (np.ndarray, dtype=np.uint8): a face-containing
            image in bgr format
            rect (dlib.rectangle): bbox with tight facial crop

        Returns:
            float32 coordinate array of the detected landmarks
            with shape [68, 2]
        """
        shape = self.shape_predictor(img_bgr, rect)
        pts = np.array(
            [(shape.part(i).x, shape.part(i).y) for i in range(68)],
            dtype=np.float32
        )

        return pts
