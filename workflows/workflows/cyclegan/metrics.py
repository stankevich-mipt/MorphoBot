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

"""Evaluation metrics module for CycleGAN training workflow.

Provides FID and Inception Score
"""

from abc import ABC, abstractmethod
import logging
from typing import (
    Generic, Optional, TypeVar,
)

from configs.cyclegan_utkfaces.evaluation import (
    EvaluatorConfig,
    FrechetInceptionDistanceConfig,
    InceptionScoreConfig,
    StatefulMetricConfig
)
import mlflow
from mlflow_registry import configure_mlflow, find_and_fetch_artifacts_by_tags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from workflows.common import ExperimentLogger

from .data import CycleGANStreamDataset, PairedDataloader

logger = logging.getLogger(__name__)


BaseConfig = TypeVar("BaseConfig", bound=StatefulMetricConfig)

class BaseMetric(ABC, Generic[BaseConfig]):
    """Base class for evaluation metrics."""

    def __init__(
        self, config: BaseConfig,
        experiment_logger: Optional[ExperimentLogger]
    ) -> None:
        """Instantiate with configuration dataclass."""
        self.config = config
        self.device = torch.device(config.device)
        self.experiment_logger = experiment_logger
        self._model = None
        self._is_initalized = False

    @abstractmethod
    def initialize(self) -> None:
        """Load models and setup inference-ready state."""
        pass

    @abstractmethod
    def compute(self, *args, **kwargs) -> dict[str, float]:
        """Metric-specific estimation logic."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the state of the stateful metric."""
        pass

    def _load_model_from_registry(
        self, model_name: str
    ) -> Optional[torch.nn.Module]:
        """Search for models within the registry."""
        configure_mlflow()

        try:
            artifacts_path = find_and_fetch_artifacts_by_tags(
                dst_dir="./temp_models", tags=self.config.mlflow_tags, unique=True
            )

            model_files = (
                list(artifacts_path.glob("*.pth"))
                + list(artifacts_path.glob("*.pt"))
            )

            if model_files:
                model_path = model_files[0]
                model = torch.load(model_path, map_location=self.device)
                logger.info(f"Loaded {model_name} from registry: {model_path}")
                return model
            else:
                logger.warning(f"No models were found in {artifacts_path}")
                return None

        except Exception as e:
            logger.warning(f"Failed to load {model_name} from registry: {e}")
            return None


class FIDMetric(BaseMetric):
    """Frechet Inception Distance evaluator.

    Measures quality of generated images by comparing feature
    distributions of real and generated images using Inception-v3
    features.
    """

    def __init__(
        self,
        config: FrechetInceptionDistanceConfig,
        experiment_logger: ExperimentLogger | None
    ) -> None:
        """State is kept in two arrays with real and fake features."""
        super().__init__(config, experiment_logger)
        self.real_features = []
        self.fake_features = []

    def initialize(self) -> None:
        """Initialize Inception-v3 feature extractor."""
        if self._is_initalized:
            return

        logger.info("Initializing FID metric...")

        self.fid_metric = FrechetInceptionDistance(
            feature=self.config.feature_dim,
            normalize=True
        ).to(self.device)
        logger.info("Using Torchmetrics FID implementation")
        self._is_initalized = True
        logger.info("FID metric initialized")

    def update(self, images: torch.Tensor, is_real: bool) -> None:
        """Update metric state with batch of images.

        Args:
            images: Batch of images (N, 3, H, W) in range [0, 1]
            is_real: whether images are from real image distribution
        """
        if not self._is_initalized:
            self.initialize()

        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        self.fid_metric.update(images, real=is_real)

    def compute(self) -> dict[str, float]:
        """Call compute for torchmetrics.FrechetInceptionDistance."""
        fid = self.fid_metric.compute().item()

        if self.experiment_logger:
            with self.experiment_logger:
                mlflow.log_metric("fid", fid)

        return {"fid": fid}

    def reset(self) -> None:
        """Free feature pools."""
        self.fid_metric.reset()

class InceptionScoreMetric(BaseMetric):
    """Inception Score metric."""
    def __init__(
        self,
        config: InceptionScoreConfig,
        experiment_logger: Optional[ExperimentLogger] = None
    ):
        """Use images array to keep the state."""
        super().__init__(config, experiment_logger)
        self.splits = config.splits
        self.images = []

    def initialize(self):
        """Load setup-dependent Inception-V3 classificator version."""
        if self._is_initalized:
            return

        logger.info("Initializing Inception Score metric...")
        self.is_metric = InceptionScore(
            splits=self.splits,
            normalize=True
        ).to(self.device)
        logger.info("Using TorchMetrics Inception Score implementation.")
        self._is_initalized = True
        logger.info("Inception Score metric initialized")

    def update(self, images: torch.Tensor):
        """Update metric with batch of generated images.

        Args:
            images: Generated images (N, 3, H, W) in range [0, 1]
        """
        if not self._is_initalized:
            self.initialize()

        self.is_metric.update(images)

    def compute(self) -> dict[str, float]:
        """Branch computation logic depending on setup."""
        is_mean, is_std = self.is_metric.compute()
        result = {"is_mean": is_mean.item(), "is_std": is_std.item()}

        if self.experiment_logger:
            with self.experiment_logger:
                mlflow.log_metric("inception_score_mean", result["is_mean"])
                mlflow.log_metric("inception_score_std", result["is_std"])

        return result

    def reset(self) -> None:  # noqa: D102
        self.is_metric.reset()


class CombinedEvaluator:
    """Combined evaluator for all metrics."""

    def __init__(
        self,
        config: EvaluatorConfig,
        experiment_logger: Optional[ExperimentLogger] = None
    ):
        """Initialize from config."""
        self.config = config
        self.experiment_logger = experiment_logger

        self.fid = FIDMetric(config.fid_config, experiment_logger)
        self.inception_score = InceptionScoreMetric(config.is_config, experiment_logger)

        self.results_history = []

    def initialize_all(self) -> None:
        """Run necessary state setup operations for each metric."""
        self.fid.initialize()
        self.inception_score.initialize()

    def evaluate_datasets(
        self,
        G_A2B: nn.Module,
        G_B2A: nn.Module,
        dataloader_A: DataLoader,
        dataloader_B: DataLoader,
        eval_device: str = "cpu",
    ) -> dict[str, float]:
        """Evaluate quality of CycleGAN-generated samples against real data.

        Args:
            G_A2B: CycleGAN generator A->B
            G_B2A: CycleGAN generator B->A
            dataloader_A: DataLoader for images from domain A
            dataloader_B: DataLoader for images from domain B
            eval_device: use this device to calculate metrics

        Returns:
            Dictionary of metric scores

        """
        self.initialize_all()

        logger.info("Starting comprehensive evaluation...")

        if self.experiment_logger:
            self.experiment_logger.log_dataclass_configs(evaluation_config=self.config)

        self.fid.reset()
        self.inception_score.reset()

        # resample base dataloaders with metric-specific batch size

        dataloader_A_fid = DataLoader(
            dataset=dataloader_A.dataset,
            generator=dataloader_A.generator,
            sampler=dataloader_A.sampler,
            batch_size=self.config.fid_config.batch_size,
            num_workers=dataloader_A.num_workers,
            pin_memory=True
        )

        dataloader_B_fid = DataLoader(
            dataset=dataloader_B.dataset,
            generator=dataloader_B.generator,
            sampler=dataloader_B.sampler,
            batch_size=self.config.fid_config.batch_size,
            num_workers=dataloader_B.num_workers,
            pin_memory=True
        )

        stream_dataset_fid = CycleGANStreamDataset(
            PairedDataloader(
                dataloader_A=dataloader_A_fid,
                dataloader_B=dataloader_B_fid
            ),
            G_A2B=G_A2B,
            G_B2A=G_B2A,
            total_samples=self.config.fid_config.total_samples,
            device=eval_device,
            return_on_cpu=False
        )

        # resample original dataset for inception score

        dataloader_A_is = DataLoader(
            dataset=dataloader_A.dataset,
            generator=dataloader_A.generator,
            sampler=dataloader_A.sampler,
            batch_size=self.config.is_config.batch_size,
            num_workers=dataloader_A.num_workers,
            pin_memory=True
        )

        dataloader_B_is = DataLoader(
            dataset=dataloader_B.dataset,
            generator=dataloader_B.generator,
            sampler=dataloader_B.sampler,
            batch_size=self.config.is_config.batch_size,
            num_workers=dataloader_B.num_workers,
            pin_memory=True
        )

        stream_dataset_is = CycleGANStreamDataset(
            PairedDataloader(
                dataloader_A=dataloader_A_is,
                dataloader_B=dataloader_B_is
            ),
            G_A2B=G_A2B,
            G_B2A=G_B2A,
            total_samples=self.config.is_config.total_samples,
            device=eval_device,
            return_on_cpu=False
        )

        results = {}

        for batch in stream_dataset_fid:

            self.fid.update(batch["A"], is_real=True)
            self.fid.update(batch["B"], is_real=True)
            self.fid.update(batch["fake_A"], is_real=False)
            self.fid.update(batch["fake_B"], is_real=False)

        fid_results = self.fid.compute()
        results.update(fid_results)

        for batch in stream_dataset_is:

            self.inception_score.update(batch["fake_A"])
            self.inception_score.update(batch["fake_B"])

        is_results = self.inception_score.compute()
        results.update(is_results)

        if self.experiment_logger:
            with self.experiment_logger:
                for metric_name, value in results.items():
                    mlflow.log_metric(f"eval_{metric_name}", value)

        self.results_history.append(results)

        return results

    def get_results_history(self) -> list[dict[str, float]]:
        """Get history of evaluation results."""
        return self.results_history.copy()
