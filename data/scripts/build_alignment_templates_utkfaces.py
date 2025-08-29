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


"""CLI UTKFaces alignment template building tool.

Consumes a manifest.jsonl file created by the
build_utkfaces_manifest_dlib_detector.py script to
Build per-class (male/female) facial alignment
templates from a subset of UTKFaces.

Input:
    - JSONL manifest with fields:
        - status: "ok" for usable samples
        - landmarks68: [[x, y] * 68]
        - label: "male" or "female", case insensitive
    - target_size: output canvas size (e.g., 256)
    - class name: by default male/female


Output:
    - JSON templates with 5-point canonical coordinates for each class
    {
        "class": "male",
        "target_size": 256,
        "points": [[x, y] * 5]
    }
    - visualization PNGs showing the canonical coordinates over a blank canvas
    - MLFlow run that logs parameters, counts, and artifacts

Usage:

    poetry run python training/gender_id/scripts/build_alignment_templates.py \
    --manifest /data/gender_datasets/utkfaces/manifest.jsonl \
    --out-dir /data/gender_datasets/utkfaces/templates \
    --target-size 256
    --experiment-name gender-template-build
"""


import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import mlflow
from mlflow_registry import (
    build_artifact_s3_uri,
    configure_mlflow,
    ensure_experiment
)
from mlflow_registry.tag_profiles import TAG_PROFILES
import numpy as np
from vision.procrustes_aligner_5pt import (
    select_five_from_68,
    similarity_procrustes
)


def normalize_points(
    pts: np.ndarray
) -> tuple[str, dict[str, np.ndarray]]:
    """Normalize the set of 5 points by removing translation and scale.

    New points have mean = 0 and scale s.t RMS distance to origin = 1

    Returns:
        normed, {"mu": translation, "s": scale} - transformed points
        and stats dict to reapply later
    """
    mu = pts.mean(axis=0, keepdims=True)
    centered = pts - mu
    rms = np.sqrt((centered ** 2).sum() / pts.size)
    s = 1.0 / (rms + 1e-12)
    normed = centered * s
    return normed.astype(np.float32), {
        "mu": mu.squeeze().astype(np.float32),
        "s": s.astype(np.float32)
    }


def denormalize_points(
    normed: np.ndarray, mu: np.ndarray, s: float
) -> np.ndarray:
    """Revert normalization by adding scale and transation."""
    return (normed / s) + mu


def compute_class_template(
    points_list: list[np.ndarray],
    target_size: int
) -> dict:
    """Build a canonical template from the list of 5-point arrays.

    Operation sequence:
    - Normalize each to remove scale and translation,
      align by Procrustes to their mean
    - Average normalized points
    - Re-scale to the target canvas by mapping the mean eye
      distance to a canonical distance
    """
    if len(points_list) == 0:
        raise ValueError("No points provided for class computation")

    # normalize all sets
    normed_sets = []
    for P in points_list:
        N, stats = normalize_points(P)
        normed_sets.append(N)

    # iteratively refine means with Procrustes alignment
    mean_shape = np.mean(np.stack(normed_sets, axis=0), axis=0)

    for _ in range(5):

        aligned = []
        for N in normed_sets:
            M, _ = similarity_procrustes(N, mean_shape)
            Nh = (M[:2, :2] @ N.T).T + M[:2, 2]
            aligned.append(Nh)

        mean_shape = np.mean(np.stack(aligned, axis=0), axis=0)

    left_eye, right_eye = mean_shape[0], mean_shape[1]

    # pick empirical relative values for eye placement
    t = float(target_size)
    target_left = np.array([0.35 * t, 0.38 * t], dtype=np.float32)
    target_right = np.array([0.65 * t, 0.38 * t], dtype=np.float32)

    # place on target canvas - scale
    # by inter-ocular distance of reference points

    src = np.stack([left_eye, right_eye], axis=0)
    dst = np.stack([target_left, target_right], axis=0)
    M_lr, _ = similarity_procrustes(src, dst)

    placed = (M_lr[:2, :2] @ mean_shape.T).T + M_lr[:2, 2]

    template = {
        "target_size": int(target_size),
        "points": placed.tolist()
    }
    return template


def preview_template(
    template: dict,
    size: int,
    out_path: Path,
    label: str
):
    """Render the template points over the blank canvas."""
    canvas = np.full((size, size, 3), 245, np.uint8)
    pts = np.array(template["points"], dtype=np.float32).astype(int)
    colors = [(0, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 0)]

    for i, (x, y) in enumerate(pts):
        cv2.circle(
            canvas, (int(x), int(y)), 3, colors[i], -1, lineType=cv2.LINE_AA)

    # draw eye and mouth lines as a sanity check
    cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (0, 0, 200), 1, cv2.LINE_AA)
    cv2.line(canvas, tuple(pts[3]), tuple(pts[4]), (200, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, f"{label} template", (10, size-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA
    )
    cv2.imwrite(str(out_path), canvas)


def read_5p_templates_from_manifest(
    manifest_file: str,
    male_label: str = "male",
    female_label: str = "female",
    limit_per_class: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Get 5pt landmarks from manifest lines with status ok."""
    male_pts: list[np.ndarray] = []
    female_pts: list[np.ndarray] = []

    # load manifest, collect anchors per class
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") != "ok":
                continue
            lbl = rec.get("label", "").strip().lower()

            if rec.get("landmarks68", None) is None:
                continue

            pts68 = np.array(rec["landmarks68"], dtype=np.float32)
            if pts68.shape != (68, 2):
                continue
            pts5 = select_five_from_68(pts68)

            if lbl == male_label:
                male_pts.append(pts5)
            elif lbl == female_label:
                female_pts.append(pts5)

    if limit_per_class > 0:
        male_pts = male_pts[:limit_per_class]
        female_pts = female_pts[:limit_per_class]

    counts = {
        "male": len(male_pts),
        "female": len(female_pts)
    }

    print("Collected anchors:", counts)

    return male_pts, female_pts


def build_and_save_template(
    plist: list[np.ndarray],
    target_size: int,
    label: str,
    out_dir: Path
) -> Optional[dict]:
    """Build averaging template, save as .json."""
    if len(plist) == 0:
        print(f"No samples for {label}, skipping.")
        return None
    template = compute_class_template(plist, target_size)
    # Save JSON
    json_path = out_dir / f"template_{label}.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"class": label, **template}, jf, indent=2)
    # Save preview
    png_path = out_dir / f"template_{label}.png"
    preview_template(template, target_size, png_path, label)
    print(f"Saved {label} template:", json_path, png_path)
    return template


def parse_arguments():
    """Process CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to JSONL manifest with landmarks and labels"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for preview and templates"
    )
    parser.add_argument(
        "--experiment-name", default="build-align-templates-UTKFaces",
        help="MLFlow experiment name prefix"
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--male-label", type=str, default="male")
    parser.add_argument("--female-label", type=str, default="female")
    parser.add_argument(
        "--limit-per-class", type=int, default=0,
        help="Optional cap per class instances (0=no cap)"
    )

    return parser.parse_args()


def main():  # noqa 

    args = parse_arguments()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_mlflow()
    artifact_location = build_artifact_s3_uri(args.experiment_name)
    experiment_id = ensure_experiment(
        args.experiment_name, artifact_location
    )

    male_pts, female_pts = read_5p_templates_from_manifest(
        args.manifest, args.male_label,
        args.female_label, args.limit_per_class
    )

    build_and_save_template(
        male_pts, args.target_size, args.male_label, out_dir
    )
    build_and_save_template(
        female_pts, args.target_size, args.female_label, out_dir
    )

    with mlflow.start_run(experiment_id=experiment_id):

        mlflow.log_params({
            "manifest": str(Path(args.manifest).name),
            "target_size": args.target_size,
            "limit_per_class": args.limit_per_class,
            "male_labels": args.male_label,
            "female_labels": args.female_label,
        })

        mlflow.set_tags(TAG_PROFILES["alignment_templates"])
        mlflow.set_tags({
            "version": "v1",
            "description":
                "facial alignment templates "
                "created from subset of UTKFaces"
        })

        for lbl in {args.male_label, args.female_label}:
            if (p := out_dir / f"template_{lbl}.json").exists():
                mlflow.log_artifact(str(p), artifact_path=f"template_{lbl}")
            if (j := out_dir / f"template_{lbl}.png").exists():
                mlflow.log_artifact(str(j), artifact_path=f"template_{lbl}")


if __name__ == "__main__":
    main()
