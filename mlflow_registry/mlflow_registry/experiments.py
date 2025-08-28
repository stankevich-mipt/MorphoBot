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

"""Utilities for MLFlow experiment management."""

import mlflow

def ensure_experiment(
    exp_name: str,
    artifact_location: str
) -> str:
    """Get id of existing experiment; create one if there's none.

    Args:
        exp_name: titular experiment
        artifact_location: URI pointing to the location within
        storage attached to MLFlow registry
    """
    try:
        experiment = mlflow.get_experiment_by_name(exp_name)
        experiment_id = experiment.experiment_id  # type: ignore
    except AttributeError:
        experiment_id = mlflow.create_experiment(
            exp_name, artifact_location=artifact_location)

    return experiment_id
