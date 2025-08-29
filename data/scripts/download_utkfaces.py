# !/usr/bin/env python

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

"""Download the UTKFace dataset archive and verify its integrity via MD5."""

import argparse
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Optional

import requests


# Archive.org mirror hosts both UTKFaces files and hashes

DATASET_URLS = [
    "https://archive.org/download/UTKFace/part1.tar.gz",  # part 1
    "https://archive.org/download/UTKFace/part2.tar.gz",  # part 2
    "https://archive.org/download/UTKFace/part3.tar.gz",  # part 3
]

EXPECTED_MD5S = [
    '4c987669d98b4385d5279056cecdd88b',  # part 1
    'ff9a734ffcab5ae235dc7c5e665900b8',  # part 2
    '9038f25ba7173fae23ea805c5f3ba1e4',  # part 3
]

FILENAMES = [
    'utkface_part1.tar.gz',
    'utkface_part2.tar.gz',
    'utkface_part3.tar.gz',
]


class DownloadError(Exception):
    """Custom exception within the script scope."""
    pass


def md5_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 checksum of a file in chunks."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()


def human_size(n: Optional[int]) -> str:
    """Convert size in bytes into human-readable format."""
    if n is None:
        return "unknown"

    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f}{units[i]}"


def get_remote_size_and_resume_support(url: str, timeout: int = 20):
    """Check the content length and range request support in mirror."""
    try:
        # First, try the HEAD request
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        resp.raise_for_status()
        size = resp.headers.get("Content-Length")
        accepts_ranges = (
            resp.headers.get("Accept-Ranges", "").lower() == "bytes")
        return int(size) if size is not None else None, accepts_ranges

    except Exception:
        # Some servers do not support HEAD; in such case fallback to GET
        try:
            with closing(requests.get(url, stream=True, timeout=timeout)) as r:
                r.raise_for_status()
                size = r.headers.get("Content-Length")
                accepts_ranges = (
                    r.headers.get("Accept-Ranges", "").lower() == "bytes")
                return int(size) if size is not None else None, accepts_ranges
        except Exception as e:
            raise DownloadError(f"Unable to query remote size: {e}") from e


def download_with_resume(
    url: str,
    dest_path: str,
    expected_size: Optional[int],
    resume_supported: bool,
    chunk_size: int = 1024 * 1024
) -> None:
    """Download file with resume capability when possible."""
    temp_path = dest_path + ".part"
    existing = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    headers = {}
    if resume_supported and existing > 0:
        headers["Range"] = f"bytes={existing}-"

    with closing(
        requests.get(url, stream=True, headers=headers, timeout=30)
    ) as r:

        if r.status_code not in (200, 206):
            raise DownloadError(f"HTTP error: {r.status_code}")

        mode = "ab" if ("Range" in headers and r.status_code == 206) else "wb"
        if mode == "wb":
            existing = 0

        downloaded = existing
        total_reported = expected_size if expected_size is not None else None
        last_report = time.time()

        with open(temp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_report >= 0:
                    if total_reported is not None:
                        pct = (
                            (float(downloaded) / float(total_reported)) * 100.
                            if total_reported > 0 else 0
                        )
                        sys.stderr.write(
                            f"\rDowloading: {human_size(downloaded)}/"
                            f"{human_size(total_reported)} ({pct:.1f}%)"
                        )
                    else:
                        sys.stderr.write(
                            f"\rDownloading: {human_size(downloaded)}")

                        sys.stderr.flush()
                    last_report = now

    # Verify size if known
    if expected_size is not None:
        actual = os.path.getsize(temp_path)
        if actual != expected_size:
            raise DownloadError(
                f"Size mismatch after download: got {actual}"
                f"bytes, expected {expected_size} bytes"
            )

    # Move to final
    os.replace(temp_path, dest_path)
    sys.stderr.write("\nDownload complete.\n")


def main(dest_dir: Path):  # noqa

    for url, filename, expected_md5 in zip(
        DATASET_URLS, FILENAMES, EXPECTED_MD5S
    ):

        dest_path = str(dest_dir / filename)

        try:
            remote_size, resume_supported = (
                get_remote_size_and_resume_support(url))
            sys.stderr.write(
                f"Remote size: {human_size(remote_size)}, "
                f"resume: {resume_supported}\n"
            )

            if os.path.exists(dest_path):
                local_size = os.path.getsize(dest_path)
                if remote_size is not None and local_size == remote_size:
                    sys.stderr.write(
                        "File with matching size already exists. "
                        "Skipping the download.\n"
                    )
                    continue
                else:
                    sys.stderr.write(
                        "Existing file differs; re-downloading.\n"
                    )
                    download_with_resume(
                        url, dest_path, remote_size, resume_supported)
            else:
                download_with_resume(
                    url, dest_path, remote_size, resume_supported)

            sys.stderr.write("Computing MD5...\n")
            md5 = md5_file(dest_path)

            if md5.lower() == expected_md5.lower():
                print("MD5 verification: OK")
                continue
            else:
                print("MD5 verification: Failed")
                print(f"Expected: {expected_md5.lower()}")
                print(f"Actual: {md5.lower()}")
                sys.exit(2)

        except DownloadError as e:
            sys.stderr.write(f"\nDownload error: {e}\n")
            sys.exit(1)
        except requests.RequestException as e:
            sys.stderr.write(f"\nNetwork error: {e}\n")
            sys.exit(1)
        except KeyboardInterrupt:
            sys.stderr.write("\nInterrupted by user.\n")
            sys.exit(130)
        except Exception as e:
            sys.stderr.write(f"\nUnexpected error: {e}\n")
            sys.exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest-dir", type=str, required=True,
        help="output directory for dataset files",
    )

    args = parser.parse_args()
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    main(dest_dir)
