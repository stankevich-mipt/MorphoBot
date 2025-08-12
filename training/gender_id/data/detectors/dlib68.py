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
Facial alignment model that utilizes dlib

Provides the nessesary functional to estimate 68-point based
facial landmarks and align the face against the canonical ones.
Alignment is done while solving the orthogonal Procrustes
problem (https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem),
so the resulting affine transformation is invertible.
Face crops are provided by innate dlib HOG-based detector.

"""

import cv2
import dlib
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple


class Dlib68Detector:

    """Class implementing the dlib-based image alignment.

    Attributes:
        predictor_path (str | os.PathLike): absolute path to dlib
        facial keypoint predictor model
        detector_kind (str, optional, default='hog'): the type of
        face detector to use while processing the image
        upsample_times (int, optional, default=1): upscale the
        faces to the multiple of their size if they are too small
        margin_ratio (Tuple[float, float], optional, default=(0.1, 0.25)):
        margins to expand the bounding box to include chin/hair region
        - dlib generally provides quite tight crops that omit these details
    """

    def __init__(
        self,
        predictor_path: str | os.PathLike,
        detector_kind: str = 'hog',
        upsample_times: int = 1,
        margin_ratio: Tuple[float, float] = [0.10, 0.25],
    ):

        self.upsample_times = int(upsample_times)
        self.margin_ratio = margin_ratio

        if detector_kind == 'hog':
            self.detector = dlib.get_frontal_face_detector()
        else:
            raise ValueError("Only 'hog' detector is supported at the moment")

        self.shape_predictor = dlib.shape_predictor(predictor_path)

    def _rect_to_xyxy(self, rect: dlib.rectangle) -> tuple[int, int, int, int]:

        return rect.left(), rect.top(), rect.right(), rect.bottom()

    def _xyxy_to_rect(
            self, x1: int, y1: int, x2: int, y2: int) -> dlib.rectangle:

        return dlib.rectangle(
            left=int(x1), top=int(y1), right=int(x2), bottom=int(y2))

    def _clip_box(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img: np.ndarray
    ):
        """Clib the bbox so it fits the image."""

        h, w = img.shape[0], img.shape[1]

        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))

        if x2 <= x1:
            x2 = min(w-1, x1+1)
        if y2 <= y1:
            y2 = min(h-1, y1+1)

        return x1, y1, x2, y2

    def expand_with_margin(
        self,
        rect: dlib.rectangle,
        img: np.ndarray
    ) -> dlib.rectangle:
        """Stretch the bbox along each dimension with uniform factor.

        Args:
            rect (dlib.rectangle): a rectangle that tightly
            frames the detected face
            img_bgr (np.ndarray, dtype=np.uint8): numpy array
            representing the image being processed

        Returns:

            dlib.rectangle: expanded bbox
        """

        x1, y1, x2, y2 = self._rect_to_xyxy(rect)
        w, h = x2-x1, y2-y1
        dx, dy = int(self.margin_ratio[0] * w), int(self.margin_ratio[1] * h)

        x1m, y1m, x2m, y2m = self._clip_box(x1-dx, y1-dy, x2+dx, y2+dy, img)

        return self._xyxy_to_rect(x1m, y1m, x2m, y2m)

    def detect_one(self, img_bgr: np.ndarray) -> Optional[dlib.rectangle]:
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
        rect: dlib.rectangle
    ) -> np.ndarray:
        """Extract facial landmarks in 68pt format with dlib detector.

        Args:
            img_bgr (np.ndarray): a face-containing image in bgr format
            rect (dlib.rectangle): bbox with tight facial crop

        Returns:
            np.ndarray: coordinate array of the detected landmarks
            with shape [68, 2]
        """

        shape = self.shape_predictor(img_bgr, rect)
        pts = np.array(
            [(shape.part(i).x, shape.part(i).y) for i in range(68)],
            dtype=np.float32
        )

        return pts


if __name__ == "__main__":

    import sys

    POINT_COLOR = (0, 255, 0)
    POINT_RADIUS = 1
    POINT_THICKNESS = 2

    RECT_COLOR = (0, 0, 255)
    RECT_THICKNESS = 2

    PREDICTOR_PATH = os.getenv(
        "DLIB_68_PREDICTOR",
        "./data/detectors/dlib68/shape_predictor_68_face_landmarks.dat"
    )
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./landmarks_preview.jpg")

    # helper functions

    def load_lena_or_backup() -> np.ndarray:

        # try to load Lena from OpenCV samples
        lena = None
        try:
            sample = cv2.samples.findFile("lenna.png", required=False)
            if sample and Path(sample).exists():
                lena = cv2.imread(sample)
        except Exception:
            pass

        if lena is not None:
            return lena

        # Fallback: simple synthetic "face-like image"
        img = np.full((512, 512, 3), 255, dtype=np.uint8)

        # Draw a circle face
        cv2.circle(img, (256, 256), 180, (200, 200, 200), -1)
        # Eyes
        cv2.circle(img, (196, 216), 20, (0, 0, 0), -1)
        cv2.circle(img, (316, 216), 20, (0, 0, 0), -1)
        # Nose
        cv2.circle(img, (256, 266), 12, (50, 50, 50), -1)
        # Mouth
        cv2.ellipse(img, (256, 326), (70, 25), 0, 0, 180, (0, 0, 0), 4)
        return img

    def draw_rect(img, rect, color=RECT_COLOR, thickness=RECT_THICKNESS):

        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def draw_landmarks(
        img,
        landmarks,
        color=POINT_COLOR,
        r=POINT_RADIUS,
        t=POINT_THICKNESS
    ):

        for x, y in landmarks:

            cv2.circle(img, (int(x), int(y)), r, color, t)

    if not Path(PREDICTOR_PATH).exists():
        raise FileNotFoundError(
            f"Smoke test failed: DLIB 68 predictor not found "
            f"at {PREDICTOR_PATH}. Set DLIB_68_PATH to your "
            "shape_predictor_68_face_landmarks.dat"
        )

    detector = Dlib68Detector(
        predictor_path=PREDICTOR_PATH,
    )

    # load test image
    img = load_lena_or_backup()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(OUTPUT_PATH, img)

    # run landmark detection pipeline
    rect = detector._expand_with_margin(detector.detect_one(img), img)
    if not rect:
        print("Smoke test failed: no images detected")
        sys.exit(1)

    landmarks = detector.landmarks68(img, rect)

    draw_rect(img, rect)
    draw_landmarks(img, landmarks)

    cv2.imwrite(OUTPUT_PATH, img)
    print(f"Smoke test passed: saved preview with landmarks to {OUTPUT_PATH}")
