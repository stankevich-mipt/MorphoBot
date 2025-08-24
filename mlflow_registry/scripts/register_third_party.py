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

"""Register dlib-68 landmark shape predictor in MLFlow as 3p-dependency.

Usage (from shell or entrypoint):
    python mlflow/register_third_party.py --predictor-path
    /models/third_party/shape_predictor_68_face_landmarks.dat

Options:
    -- predictor-path: Path to predictor file
    -- source (optional, default=DLIB_MODEL_URL): valid url to dlib
    predictor checkpoint file
    -- force (optional, default=True): always upload, even if run exists



"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import urllib.request

import mlflow
import mlflow.tracking as mlflow_tracking

REGISTRY_EXPERIMENT_NAME = "third_party_assets"
REGISTRY_RUN_NAME = "dlib_shape_predictor_68_init"
ARTIFACT_SUBPATH = "dlib_shape_predictor_68"
REGISTRY_INFO_FILE = "third_party_registry_location.txt"
DLIB_MODEL_URL = (
    "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
)


def download_predictor_if_needed(
    target_path: Path, url: str = DLIB_MODEL_URL
) -> None:
    """Pull dlib predictor weights from external url into target_path."""
    if target_path.exists():
        print(f"Using existing file: {target_path}")
        return
    print(f"Downloading predictor from {url}")
    if url.endswith(".bz2"):

        tmpfile = str(target_path) + ".bz2"
        urllib.request.urlretrieve(url, tmpfile)
        import bz2
        with bz2.open(tmpfile, "rb") as fin, open(target_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.remove(tmpfile)
    else:
        urllib.request.urlretrieve(url, target_path)

    print(f"Downloaded predictor checkpoint and extracted to {target_path}")


def registed_dlib_predictor(
    predictor_path: str,
    source_url: str,
    artifact_location: str,
    force: bool = True
) -> str:
    """Check for predictor artifact existence; if there's none, create one.

    Args:
        predictor_path: absolute path to predictor artifact
        source_url: url to download predictor weights
        artifact_location: where exactly to store artifacts
        force: whether to overwrite artifact if there is one
    """
    client = mlflow_tracking.MlflowClient()
    print("Client tracking URI:", client._tracking_client.tracking_uri)
    exp = client.get_experiment_by_name(REGISTRY_EXPERIMENT_NAME)
    if exp is None:
        exp_id = client.create_experiment(
            REGISTRY_EXPERIMENT_NAME,
            artifact_location=artifact_location
        )
    else:
        exp_id = exp.experiment_id

    run_id = None

    if runs := client.search_runs(
        [exp_id], f"tags.run_name = '{REGISTRY_RUN_NAME}'",
        max_results=1
    ):
        run_id = runs[0].info.run_id
        if not force:
            print("Third-party artifact already registered in run_id={run_id}")
            return run_id

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=REGISTRY_RUN_NAME,
    ) as run:
        mlflow.log_artifact(predictor_path)
        mlflow.set_tag("source_url", str(source_url))
        mlflow.set_tag("type", "third-party")
        mlflow.set_tag("role", "landmark-detector")
        mlflow.set_tag("version", "v1")
        mlflow.set_tag(
            "description",
            "dlib 68pt shape predictor uploaded at registry bootstrap"
        )
        run_id = run.info.run_id
        print(f"Registered dlib shape predictor artifact under run_id={run_id}")

    return run_id


def write_registry_reference(
    run_id: str,
    info_file: str = REGISTRY_INFO_FILE
) -> None:
    """Dump the artifact uri into text buffer."""
    uri = (
        f"runs:/{run_id}/{ARTIFACT_SUBPATH}/"
        "shape_predictor_68_face_landmarks.det"
    )
    with open(info_file, 'w') as f:
        f.write(uri + '\n')
    print(f"Wrote MLFlow artifact URI for predictor to {info_file}:\n{uri}")


def main():  # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor-path", type=str, required=True,
        help="Path to dlib shape predictor .dat path"
    )
    parser.add_argument(
        "--source-url", type=str,
        default=DLIB_MODEL_URL, help="URL to download predictor if missing"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-upload even if the artifact run exists"
    )
    args = parser.parse_args()

    # get correct tracking uri
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
    )
    # pickup the artifact root from env, fallback to local file if not set
    artifact_root = os.getenv(
        "MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT", "file:///mlflow/artifacts"
    )
    artifact_location = f"s3://{artifact_root}/{ARTIFACT_SUBPATH}"
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", None)
    aws_secret_access_key = os.getenv(
        "AWS_SECRET_ACCESS_KEY", None
    )
    print(tracking_uri, artifact_location, aws_access_key_id, aws_secret_access_key)
    mlflow.set_tracking_uri(tracking_uri)

    predictor_path = Path(args.predictor_path)
    predictor_path.parent.mkdir(parents=True, exist_ok=True)
    download_predictor_if_needed(predictor_path, args.source_url)

    run_id = registed_dlib_predictor(
        str(predictor_path), args.source_url,
        artifact_location, force=args.force
    )
    write_registry_reference(run_id)


if __name__ == "__main__":
    main()
