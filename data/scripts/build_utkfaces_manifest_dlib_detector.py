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


"""Label UTKFaces dataset with dlib keypoint detector.

Build a manifest for UTKFaces dataset with extracted facial
keypoints (68 landmarks) using the off-the-shelf dlib keypoint
detector model. The whole process traverses through the directory,
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
from typing import Optional

import cv2
import dlib
import mlflow
from mlflow_registry import (
    build_artifact_s3_uri,
    configure_mlflow,
    ensure_experiment,
    find_and_fetch_artifacts_by_tags,
)
from mlflow_registry.tag_profiles import TAG_PROFILES
import numpy as np
from tqdm.auto import tqdm
from vision.dlib68_detector import Dlib68Detector
from vision.utils import (
    bbox_is_plausible,
    draw_preview,
    find_images,
    grid_preview,
    read_image_bgr,
)


DEFAULT_PREDICTOR_WEIGHTS_FILENAME = "shape_predictor_68_face_landmarks.dat"


def rect_to_xyxy(rect: dlib.rectangle) -> tuple[int, int, int, int]:  # type: ignore
    """Rectange object from dlib to tuple of ints."""
    return rect.left(), rect.top(), rect.right(), rect.bottom()


def get_label_from_name(img_path: Path) -> Optional[str]:
    """Helper function to pick up the label from the image path.

    For more details on the metadata provided with UTKfaces
    image filenames see https://susanqq.github.io/UTKFace/.
    """
    label = str(img_path.name).split('_')[1]
    if label == "":
        return None
    return "female" if int(label) else "male"


def process_single_image(
    img_path: Path,
    dlib_detector: Dlib68Detector,
    preview_limit: int,
    counters: dict,
    preview_images: list[np.ndarray]
):
    """Create an image record with metadata."""
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
        return rec

    label = get_label_from_name(img_path)
    if not label:
        rec["status"] = "no_label"
        rec["reason"] = "img_name_doesn't_follow_convention"
        counters["no_label"] += 1
        return rec

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
        return rec

    if bbox is None:
        rec["status"] = "no_face"
        rec["reason"] = "dlib_couldn't_detect_any_faces"
        counters["no_face"] += 1
        return rec

    ok_bbox, reason_bbox = bbox_is_plausible(list(rect_to_xyxy(bbox)), W, H)

    if not ok_bbox:
        rec["status"] = "no_face"
        rec["reason"] = f"bbox_{reason_bbox}"
        counters["no_face"] += 1
        return rec

    # landmarks
    try:
        landmarks = dlib_detector.landmarks68(img, bbox)
    except Exception as e:
        rec["status"] = "landmarks_failed"
        rec["reason"] = f"predictor_exception:{type(e).__name__}"
        counters["landmarks_failed"] += 1
        return rec

    bbox = dlib_detector.expand_with_margin(bbox, img)
    bbox = rect_to_xyxy(bbox)
    rec["bbox"] = list(bbox)
    rec["landmarks68"] = landmarks.tolist()
    rec["status"] = "ok"
    counters["ok"] += 1

    # Collect preview sample
    if len(preview_images) < preview_limit:
        preview_images.append(draw_preview(img, bbox, landmarks))

    return rec


def write_manifest(manifest_path: Path, records: list[dict]):
    """Dump image records into jsonl."""
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def log_mlflow_metrics_and_artifacts(
    experiment_id: str, counters: dict,
    manifest_path: Path, preview_images: list[np.ndarray],
    preview_artifact_name: str
):
    """Push artifacts and metadata into the registry."""
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metrics(counters)
        mlflow.set_tags(TAG_PROFILES["alignment_manifest"])
        mlflow.set_tags({
            "version": "v1",
            "description":
                "manifest for landmark dataset "
                "created from UTKFaces and labeled "
                "with open-source dlib detector"
        })
        mlflow.log_artifact(str(manifest_path), artifact_path="data-prep")
        if preview_images:
            preview = grid_preview(preview_images, cols=4)
            if preview is not None:
                tmp_preview = manifest_path.parent / preview_artifact_name
                cv2.imwrite(str(tmp_preview), preview)
                mlflow.log_artifact(str(tmp_preview), artifact_path="data-prep")
                os.remove(tmp_preview)


def parse_args():
    """Process CLI arguments."""
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

    return parser.parse_args()


def main(): # noqa

    args = parse_args()

    # resolve input, output, and detector paths
    input_root = Path(args.input_dir).resolve()
    output_path = Path(args.output_path).resolve()

    # fetch dlib detector weights from registry
    configure_mlflow()
    predictor_path = find_and_fetch_artifacts_by_tags(
        dst_dir=args.predictor_path,
        tags=TAG_PROFILES["vision_landmarks_detector"],
        unique=True,
    ) / DEFAULT_PREDICTOR_WEIGHTS_FILENAME

    # setup dlib detector
    dlib_detector = Dlib68Detector(
        predictor_path=str(predictor_path),
        detector_kind='hog',
        upsample_times=int(args.upsample_times)
    )

    # get all images from input dir and its subfolders
    images = find_images(input_root)
    total = len(images)

    print(f"Found {total} images under {input_root}")

    # setup manifest experiment in registry
    artifact_location = build_artifact_s3_uri(str(input_root.name) + "_manifest")
    experiment_id = ensure_experiment(args.experiment_name, artifact_location)

    counters: dict[str, int] = {
        "total": total,
        "ok": 0,
        "no_face": 0,
        "no_label": 0,
        "landmarks_failed": 0,
        "read_error": 0,
        "invalid_image": 0,
    }
    preview_images: list[np.ndarray] = []
    records: list[dict] = []

    iterator = tqdm(enumerate(sorted(images), 1))

    for idx, img_path in iterator:

        tqdm.set_description(
            iterator, f"Processing element {idx} out of {len(images)}"
        )
        rec = process_single_image(
            img_path, dlib_detector, args.preview_limit,
            counters, preview_images
        )
        records.append(rec)

    write_manifest(output_path, records)

    log_mlflow_metrics_and_artifacts(
        experiment_id, counters,
        output_path, preview_images, args.preview_artifact
    )

    print("Data prep complete")
    print("Counters:", counters)


if __name__ == "__main__":
    main()
