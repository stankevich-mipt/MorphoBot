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

"""Interface for registry querying."""


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import mlflow.artifacts as mlflow_artifacts
import mlflow.tracking as mlflow_tracking


@dataclass(frozen=True)
class RunRecord:
    """Lightweight dataclass representing run metadata."""
    run_id: str
    experiment_id: str
    tags: dict[str, str]
    params: dict[str, str]


class RegistrySearchError(RuntimeError):
    """Custom class for registry-related exceptions."""
    pass


def _build_tag_filter(tags: dict[str, str]) -> str:
    """Join multiple tags with AND clause for a single condition."""
    clauses = []
    for k, v in tags.items():
        val = v.replace("'", "\\'")
        clauses.append(f"tags.{k} = '{val}'")
    return " AND ".join(clauses) if clauses else ""


def search_run_by_tags(
    tags: dict[str, str],
    experiment_names: Optional[Sequence[str]] = None,
    max_results: int = 2000,
    order_by: list[str] = ["attribute.start_time DESC"],
) -> list[RunRecord]:
    """Search MLFlow runs by tag equality filters across one or more experiments."""
    client = mlflow_tracking.MlflowClient()

    exp_ids: list[str] = []
    if experiment_names:
        for name in experiment_names:
            exp = client.get_experiment_by_name(name)
            if exp is not None:
                exp_ids.append(exp.experiment_id)

    else:
        for exp in client.search_experiments():
            if exp.lifecycle_stage == "active":
                exp_ids.append(exp.experiment_id)

    if not exp_ids:
        return []

    filter_string = _build_tag_filter(tags)
    results: list[RunRecord] = []

    for exp_id in exp_ids:

        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by
        )

        for run in runs:
            results.append(
                RunRecord(
                    run_id=run.info.run_id,
                    experiment_id=run.info.experiment_id,
                    tags={k: v for k, v in run.data.tags.values()},
                    params={k: v for k, v in run.data.params.items()}
                )
            )

    return results


def get_unique_run_by_tags(
    tags: dict[str, str],
    experiment_names: Optional[Sequence[str]] = None,
) -> RunRecord:
    """Return exactly one run matching tags; raise if zero or multiple."""
    runs = search_run_by_tags(tags, experiment_names=experiment_names, max_results=50)
    if not runs:
        raise RegistrySearchError(f"No MLFlow runs found for tags={tags}")

    unique = {r.run_id: r for r in runs}
    if len(unique) != 1:
        raise RegistrySearchError(
            f"Non-unique runs for tags={tags}: "
            f"found {len(unique)}"
        )

    return next(iter(unique.values()))


def get_latest_runs_by_tags(
    tags: dict[str, str],
    experiment_names: Optional[Sequence[str]] = None,
) -> RunRecord:
    """Return the most recent run matching tags; raise if zero."""
    runs = search_run_by_tags(
        tags, experiment_names=experiment_names,
        max_results=50, order_by=["attribute.start_time DESC"]
    )
    if not runs:
        raise RegistrySearchError(f"No MLFlow runs found for tags={tags}")

    return runs[0]


def resolve_artifact_to_local(
    run: RunRecord,
    artifact_subpath: str,
    dst_dir: Optional[str] = None,
) -> Path:
    """Load an artifact produced in run into the local dir."""
    artifact_uri = f"runs:/{run.run_id}/{artifact_subpath}"
    local = mlflow_artifacts.download_artifacts(
        artifact_uri=artifact_uri, dst_path=dst_dir
    )
    return Path(local)


def find_and_fetch_artifacts_by_tags(
    tags: dict[str, str],
    artifact_subpath: str,
    experiment_names: Optional[Sequence[str]] = None,
    unique: bool = True,
    dst_dir: Optional[str] = None
) -> Path:
    """Find a run by tags and fetch the specified artifact to local.

    This method provides facade for consumer/registry interaction.

    Notes:
        - unique=True enforces exactly one run; otherwise, pick latest.
    """
    if unique:
        run = get_unique_run_by_tags(tags=tags, experiment_names=experiment_names)
    else:
        run = get_latest_runs_by_tags(tags=tags, experiment_names=experiment_names)

    return resolve_artifact_to_local(run, artifact_subpath, dst_dir=dst_dir)
