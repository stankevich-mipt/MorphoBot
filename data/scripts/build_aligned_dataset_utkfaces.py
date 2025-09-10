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


"""CLI script for building two datasets of aligned faces.

Consumes a manifest.jsonl file created by the
build_utkfaces_manifest_dlib_detector.py script to
create two monoclass (male/female) folders with
cropped and aligned images.

Input:
    - JSONL manifest with fields:
        - status: "ok" for usable samples
        - bbox: image region containing a face
        - landmarks68: [[x, y] * 68]
        - label: "male" or "female", case insensitive
        - src: path to image
    - output directory

Output:
    - two subfolders in the output directory with assembled datasets
    - visualization PNGs with grids of male and female faces
    - MLFlow run that logs parameters, counts, and artifacts

Usage:

    poetry run python training/gender_id/scripts/build_alignment_datasets.py \
    --manifest /data/gender_datasets/utkfaces/manifest.jsonl \
    --out-root /data/gender_datasets/utkfaces \
    --template-male template_male.json --template-female template_female.json
    --size 256 --limit 10

Notes:
    By default, scripts tries to pull the latest
    alignment templates from the bitbucket attached to the MLFlow server.
    If this fails, the stub created at runtime in load_template function
    acts as a fallback.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Optional
from urllib.parse import urlparse

import cv2
import mlflow
from mlflow_registry import (
    build_artifact_s3_uri,
    configure_mlflow,
    ensure_experiment,
    find_and_fetch_artifacts_by_tags,
    get_latest_run_by_tags,
)
from mlflow_registry.tag_profiles import TAG_PROFILES
import numpy as np
from numpy import typing as npt
from tqdm.auto import tqdm
from vision.procrustes_aligner_5pt import FivePointAligner, Template
from vision.utils import bbox_is_plausible, read_image_bgr


def load_template_json(
    path: Optional[Path],
    fallback_size: int
) -> Template:
    """Try to load .json template, if there's none, create a fallback."""
    try:
        obj = json.loads(Path(path).read_text())  # type: ignore
        pts = np.array(obj["points"], dtype=np.float32)
        size = int(obj.get("target_size", fallback_size))
        if size != fallback_size:
            # Scale template to requested output size
            scale = float(fallback_size) / float(size)
            pts = pts * scale
        return Template(pts, size)

    finally:
        t = float(fallback_size)
        pts = np.array([
            [0.35 * t, 0.38 * t],
            [0.65 * t, 0.38 * t],
            [0.50 * t, 0.52 * t],
            [0.40 * t, 0.70 * t],
            [0.60 * t, 0.70 * t],
        ], dtype=np.float32)
        return Template(pts, fallback_size)


def process_record(
    rec: dict,
    aligner_5pt: FivePointAligner,
) -> tuple[Optional[npt.NDArray[np.uint8]], Optional[dict]]:
    """Process manifest records, providing meaningful summaries."""
    status = rec.get("status")
    if status != "ok":
        return None, None

    src_path = rec.get("src")
    if not src_path:
        return None, None

    img = read_image_bgr(src_path)
    if img is None:
        return None, None

    bbox = rec.get("bbox")

    summary: dict[str, str | None] = {
        "src": src_path,
        "label": str(rec.get("label", "male")),
        "saved": "no",
        "reason": None,
        "err": None,
    }

    H, W = img.shape[:2]

    if bbox and isinstance(bbox, list) and len(bbox) == 4:
        bbox = list(map(int, bbox))
        ok_bbox, reason = bbox_is_plausible(bbox, W, H)
        if not ok_bbox:
            summary["reason"] = f"bbox_{reason}"
            return None, summary
    else:
        summary["reason"] = "no_valid_bbox"
        return None, summary

    lm68 = rec.get("landmarks68")
    if lm68 is None:
        summary["reason"] = "no_landmark"
        return None, summary

    pts68 = np.array(lm68, dtype=np.float32)
    if pts68.shape != (68, 2):
        summary["reason"] = "bad_landmark_shape"
        return None, summary

    lbl = (rec.get("label") or "").strip().lower()

    aligned, _, err = aligner_5pt.align(img, pts68, lbl)

    if not np.isfinite(err):
        summary["reason"] = "affine_nan"
        return None, summary

    summary["saved"] = "yes"
    summary["reason"] = "ok"
    summary["err"] = f"{err:.4f}"

    return aligned, summary


def save_aligned_image_if_ok(
    aligned_img: Optional[npt.NDArray[np.uint8]],
    summary: dict[str, str],
    jpeg_quality: int
) -> int:
    """Save properly processed image with original stem."""
    stem = Path(summary["src"]).stem
    out_path = Path(summary["dst"]) / f"{stem}.jpg"

    if aligned_img is not None:
        cv2.imwrite(
            str(out_path), aligned_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        return 1

    return 0


def build_two_folders_from_manifest(
    manifest_path: Path,
    out_root: Path,
    size: int,
    male_dir_name: str,
    female_dir_name: str,
    template_male: Optional[Path],
    template_female: Optional[Path],
    limit: int = 0,
    jpeg_quality: int = 95
) -> tuple[int, int, list[dict[str, str]]]:
    """Create subfolders of aligned male/female faces from .jsonl anno.

    Args:
        manifest_path: annotation file absolute path
        out_root: root directory for subfolders
        size: image size post alignment
        male_dir_name: relative path to male folder within root
        female_dir_name: relative path to female folder within root
        template_male: absolute path to .json with male face alignment template
        template_female: absolute path to .json with female face alignment template
        limit: if not 0, caps the amount of processed manifest lines.
        jpeg_quality: level of lossy compression
    """
    # 1) Setup paths
    out_male = out_root / male_dir_name
    out_female = out_root / female_dir_name
    out_male.mkdir(parents=True, exist_ok=True)
    out_female.mkdir(parents=True, exist_ok=True)

    # 2) Initialize aligner class
    aligner = FivePointAligner(
        load_template_json(template_male, size),
        load_template_json(template_female, size)
    )

    summary_rows: list[dict[str, str]] = []
    processed = 0
    ok_count = 0

    with open(manifest_path, "r", encoding="utf-8") as f:

        lines = tqdm([line for line in f])

        for j, line in enumerate(lines):

            lines.set_description(f'Processed {j} out of {len(lines)} files')

            if limit and processed >= limit:
                break
            processed += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            aligned_img, summary = process_record(rec, aligner)
            if summary is not None:
                summary_rows.append(summary)

                summary["dst"] = (
                    out_male if summary["label"] == "male"
                    else out_female
                )

                ok_count += save_aligned_image_if_ok(
                    aligned_img, summary, jpeg_quality
                )

    return processed, ok_count, summary_rows


def parse_arguments():
    """Process CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", required=True,
        help="Path to JSONL manifest with landmarks and labels"
    )
    parser.add_argument(
        "--out-root", required=True, help="Root output directory")
    parser.add_argument(
        "--experiment-name", type=str,
        default="build-aligned-subsets-utkfaces",
        help="MLFlow experiment name"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="Output square size (pixels)")
    parser.add_argument(
        "--template-male", type=str, required=True,
        help="Path to male template JSON"
    )
    parser.add_argument(
        "--template-female", type=str, required=True,
        help="Path to female template JSON"
    )
    parser.add_argument(
        "--male-out", default="male", help="Subfolder for male faces")
    parser.add_argument(
        "--female-out", default="female", help="Subfolder for female faces")

    parser.add_argument(
        "--limit", type=int, default=0,
        help="Optional limit of processed records (0 = all)"
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPEG quality for output images"
    )
    return parser.parse_args()


def main():  # noqa

    args = parse_arguments()

    configure_mlflow()
    artifact_location = build_artifact_s3_uri(args.experiment_name)
    experiment_id = ensure_experiment(
        args.experiment_name, artifact_location
    )

    manifest_path = Path(args.manifest)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:

        root = Path(tmp)
        artifact_root = find_and_fetch_artifacts_by_tags(
            dst_dir=str(root),
            tags=TAG_PROFILES["alignment_templates"],
            unique=False,
        )

        run_data = get_latest_run_by_tags(
            tags=TAG_PROFILES["alignment_templates"]
        )

        template_male = artifact_root / args.template_male
        template_female = artifact_root / args.template_female

        if not template_male.exists():
            template_male = None
        if not template_female.exists():
            template_female = None

        processed, ok_count, summary_rows = build_two_folders_from_manifest(
            manifest_path=manifest_path,
            out_root=out_root,
            size=int(args.size),
            male_dir_name=args.male_out,
            female_dir_name=args.female_out,
            template_male=template_male,
            template_female=template_female,
            limit=int(args.limit),
            jpeg_quality=int(args.jpeg_quality),
        )

        out_csv = artifact_root / "export_summary.csv"

        with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(
                csvf,
                fieldnames=["src", "label", "saved", "reason", "err", "dst"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        with mlflow.start_run(experiment_id=experiment_id):

            mlflow.log_params({
                "manifest": str(Path(args.manifest).name),
                "target_size": args.size,
                "registry_template_run": str(run_data.run_id)
            })

            mlflow.log_metrics({
                "processed": processed,
                "ok_count": ok_count
            })

            mlflow.set_tags(TAG_PROFILES["dataset"])

            mlflow.set_tags({
                "version": "v1",
                "description": (
                    "Building monoclass subsets "
                    "of aligned male/female faces from UTK"
                ),
                "name": "utkfaces",
            })

            mlflow.log_artifact(
                str(out_csv), artifact_path="export_summary"
            )

        print(f"Processed: {processed}, Saved: {ok_count}, CSV: {out_csv}")


if __name__ == "__main__":
    main()
