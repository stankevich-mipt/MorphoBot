"""
Build a manifest with extracted facial keypoints (68 landmarks)
using the off-the-shelf dlib keypoint detector model.
The whole process traverses through the given image folder,
logging the process as MLFlow experiment.

Action sequence
    1. Recursively scan input folder for image files
    2. Select the largest face with dlib HOG detector
    3. Extract 68-point landmarks
    4. Save the JSONL manifest with fields:
        {
            "src": "absolute image path",
            "status": "ok" | "no_face" | "landmarks_failed" |
            "read_error" | "invalid_image",
            "bbox": [x1, y1, x2, y2] or null,
            "landmarks68": [[x, y], ...] or null,
            "width": W,
            "height" H,
            "label": "optional label derived from folder name",
        }
    5. Log MLFLow run with parameters, counters, and artifacts

Usage:
    poetry run python training/gender_id/scripts/build_manifest.py \
        --input-dir ./data/gender_datasets/utk_face/raw \
        --output-path ./data/gender_datasets/utk_face/manifest.jsonl \
        --predictor-path
            ./training/gender_id/data/detectors/dlib68/
            shape_predictor_68_face_landmarks.dat \
        --experiment-name gender-data-prep \
        --label-from-folder
"""


import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tqdm.auto import tqdm

import cv2
import dlib
import numpy as np
import mlflow

from data.detectors.dlib68 import Dlib68Detector


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def read_image_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    return img


def get_label_from_name(img_path: Path) -> Optional[str]:

    label = str(img_path.name).split('_')[1]
    if label == "":
        return None
    return "female" if int(label) else "male"


def rect_to_xyxy(rect: dlib.rectangle) -> Tuple[int, int, int, int]:
    return rect.left(), rect.top(), rect.right(), rect.bottom()


def bbox_size(x1: int, y1: int, x2: int, y2: int):
    return max(0, x2 - x1), max(0, y2 - y1)


def bbox_aspect_ratio(w: int, h: int, eps: float = 1e-6):
    a = (max(w, h) + eps) / (min(w, h) + eps)
    return a  # >= 1


def bbox_area(w: int, h: int):
    return w * h


def bbox_is_plausible(
    bbox: dlib.rectangle,
    W: int,
    H: int,
    min_side: int = 48,
    max_ratio: float = 2.0,
    min_area_frac: float = 0.01,
    max_area_frac: float = 0.85,
):

    x1, y1, x2, y2 = rect_to_xyxy(bbox)
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
    rect: Optional[dlib.rectangle],
    pts68: Optional[np.ndarray]
) -> np.ndarray:

    vis = img.copy()
    if rect is not None:

        cv2.rectangle(
            vis,
            (rect.left(), rect.top()),
            (rect.right(), rect.bottom()),
            (255, 0, 0),
            2
        )

    if pts68 is not None:
        for (x, y) in pts68.astype(int):
            cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

    return vis


def grid_preview(
    images: List[np.ndarray],
    cols: int = 4,
    cell_size: Tuple[int, int] = (256, 256)
) -> Optional[np.ndarray]:

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


def main():

    os.environ['PYTHONPATH'] = '/workspace'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", required=True,
        help="Root directory with images (scanned_recursively)"
    )
    parser.add_argument(
        "--output-path", required=True,
        help="Output manifest JSONL path"
    )
    parser.add_argument(
        "--predictor-path", required=True,
        help="Path to dlib 68 landmarks predictor .dat file"
    )
    parser.add_argument(
        "--upsample-times", type=int, default=1,
        help="Detector upsample factor"
    )
    parser.add_argument(
        "--experiment-name", default="gender-data-prep",
        help="MLFlow experiment name prefix"
    )
    parser.add_argument(
        "--preview-limit", type=int, default=16,
        help="Number of preview images to log as artifact"
    )
    parser.add_argument(
        "--preview-artifact", default="preview_grid.jpg",
        help="Filename for preview grid artifact"
    )

    args = parser.parse_args()

    # resolve input, output, and detector paths
    input_root = Path(args.input_dir).resolve()
    output_path = Path(args.output_path).resolve()

    if not Path(args.predictor_path).exists():
        raise FileNotFoundError(
            f"Predictor file not found: {args.predictor_path}")

    # setup dlib detector
    dlib_detector = Dlib68Detector(
        predictor_path=args.predictor_path,
        detector_kind='hog',
        upsample_times=int(args.upsample_times)
    )

    # get all images from input dir and its subfolders
    images = find_images(input_root)
    total = len(images)

    print(f"Found {total} images under {input_root}")

    # setup mlflow experiment
    artifact_root = os.getenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT",
        "mlflow-artifacts-morphobot-dev"
    )

    artifact_location = (
        's3://' + os.path.join(
            artifact_root, str(input_root.name) + '_manifest'
        )
    )

    try:
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(
            args.experiment_name, artifact_location=artifact_location)

    with mlflow.start_run(experiment_id=experiment_id):

        mlflow.log_params({
            "input_dir": str(input_root),
            "upsample_times": args.upsample_times,
            "predictor_path": str(Path(args.predictor_path).name),
        })

        counters: Dict[str, int] = {
            "total": total,
            "ok": 0,
            "no_face": 0,
            "no_label": 0,
            "landmarks_failed": 0,
            "read_error": 0,
            "invalid_image": 0,
        }

        preview_images: List[np.ndarray] = []
        manifest_f = output_path.open("w", encoding="utf-8")

        iterator = tqdm(enumerate(sorted(images), 1))

        for idx, img_path in iterator:

            tqdm.set_description(
                iterator, f"Processing element {idx} out of {len(images)}"
            )

            rec = {
                "src": str(img_path),
                "status": None,
                "bbox": None,
                "width": None,
                "height": None,
                "label": None,
                "reason": None
            }

            img = read_image_bgr(img_path)

            if img is None:
                rec["status"] = "read_error"
                rec["reason"] = "cv2_imread_failed"
                counters["read_error"] += 1
                manifest_f.write(json.dumps(rec) + "\n")
                continue

            label = get_label_from_name(img_path)
            if not label:
                rec["status"] = "no_label"
                rec["reason"] = "img_name_doesn't_follow_convention"
                counters["no_label"] += 1
                manifest_f.write(json.dumps(rec) + "\n")
                continue

            rec["label"] = label

            H, W = img.shape[:2]
            rec["width"], rec["height"] = W, H

            # bounding box
            try:
                bbox = dlib_detector.detect_one(img)

            except Exception as e:
                rec["status"] = "invalid_image"
                rec["reason"] = f"detector_exception:{type(e).__name__}"
                counters["invalid_image"] += 1
                manifest_f.write(json.dumps(rec) + "\n")
                continue

            if bbox is None:
                rec["status"] = "no_face"
                rec["reason"] = "dlib_couldn't_detect_any_faces"
                counters["no_face"] += 1
                manifest_f.write(json.dumps(rec) + "\n")
                continue

            ok_bbox, reason_bbox = bbox_is_plausible(bbox, W, H)

            if not ok_bbox:
                rec["status"] = "no_face"
                rec["reason"] = f"bbox_{reason_bbox}"
                counters["no_face"] += 1
                manifest_f.write(json.dumps(rec) + "\n")
                continue

            # landmarks
            try:
                landmarks = dlib_detector.landmarks68(img, bbox)
            except Exception as e:
                rec["status"] = "landmarks_failed"
                rec["reason"] = f"predictor_exception:{type(e).__name__}"
                counters["landmarks_failed"] += 1
                manifest_f.write(json.dumps(rec) + "\n")
                continue

            bbox = dlib_detector.expand_with_margin(bbox, img)
            x1, y1, x2, y2 = rect_to_xyxy(bbox)
            rec["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
            rec["landmarks68"] = landmarks.tolist()
            rec["status"] = "ok"
            counters["ok"] += 1
            manifest_f.write(json.dumps(rec) + "\n")

            # Collect preview sample
            if len(preview_images) < args.preview_limit:
                preview_images.append(draw_preview(img, bbox, landmarks))

        manifest_f.close()

        for k, v in counters.items():
            mlflow.log_metric(k, v)

        # log manifest file
        mlflow.log_artifact(str(output_path), artifact_path='data-prep')

        # log preview grid
        preview = grid_preview(preview_images, cols=4)
        if preview is not None:

            tmp_preview = output_path.parent / args.preview_artifact
            cv2.imwrite(str(tmp_preview), preview)
            mlflow.log_artifact(str(tmp_preview), artifact_path='data-prep')

        print("Data prep complete")
        print("Counters:", counters)


if __name__ == "__main__":
    main()
