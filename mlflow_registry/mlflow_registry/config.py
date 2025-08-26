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

"""Implementation of MLFlow registry configuration."""

import os

import mlflow

class RegistryConfig:
    """An object representing MLFlow registry configuration."""

    def __init__(self):
        """Infer class attributes from env variables."""
        self.tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", None)
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", None)
        self.aws_secret_access_key = os.getenv(
            "AWS_SECRET_ACCESS_KEY", None
        )

    def configure_mlflow(self):
        """Override other possible .env changes if needed."""
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.s3_endpoint_url:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.s3_endpoint_url
        if self.aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
