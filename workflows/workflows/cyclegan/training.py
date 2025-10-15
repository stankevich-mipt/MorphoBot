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

"""CLI script for CycleGAN training on UTKFaces.

Artifacts produced during script runtime are pushed
to the MLFlow registry under a new run in scope
of the experiment "experiment-name"; if registry is
out of reach, the fallback is local filesystem.

Inputs:
    - experiment-name: prefix for experiment-related artifacts
    in MLFlow registry
    - seed: global seed for RNGs used in training process
    - dataset-config: absolute path to YAML file providing at least
    two fields:
        1) root_male - image folder with male faces
        2) root_female - image folder with female faces

Usage:
    poetry run python -m workflows.router_utkfaces.training
    --experiment-name cyclegan_utkfaces_baseline
    --dataset-config /path/to/dataset/config.yaml
    --seed 42
"""


import argparse
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import time
from typing import Optional
import warnings

import albumentations as A
from configs.cyclegan_utkfaces.dataset import DatasetConfig
from configs.cyclegan_utkfaces.evaluation import EvaluatorConfig
from configs.cyclegan_utkfaces.training import CycleGANConfig, OptimizationConfig
from data import ImageFolder
from data.utils import get_train_test_split_indices
import mlflow
from mlflow.types import TensorSpec
import numpy as np
import torch
from torch.amp.autocast_mode import autocast
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import make_grid, save_image
from workflows.common.config import get_config_instance
from workflows.common.data import TorchShuffler
from workflows.common.mlflow_experiment_logger import ExperimentLogger
from workflows.common.training import (
    get_numpy_rng,
    get_torch_rng,
    set_global_seed,
    setup_amp_scaler,
)

from models.cyclegan import (
    PatchDiscriminator, ResNetGenerator
)
from .data import PairedDataloader
from .io import (
    auto_resume_training,
    CycleGANCheckpointManager,
    CycleGANModels,
    TrainingState
)
from .metrics import CombinedEvaluator


def setup_logging(level=logging.INFO):  # noqa: D103
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(ch)


setup_logging()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fmt3(x):
    """Round to 3 fp is x is float, otherwise return N/A."""
    return (
        f"{x:.3f}"
        if isinstance(x, (int, float)) and math.isfinite(x)
        else "N/A"
    )


class ImagePool:
    """Image pool to store previously generated images.

    Stabilization tool that intents to prevent rapid parameter
    changes by showing a history of generated images instead of
    the most recent ones.
    """

    def __init__(self, pool_size: int = 50):
        """Initialize with empty list and size spec."""
        self.pool_size = pool_size
        if pool_size > 0:
            self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor):
        """Return images from pool, potentially replace some."""
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)

            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)

            else:
                if np.random.uniform(0, 1) > 0.5:

                    idx = np.random.randint(0, self.pool_size)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)

                else:
                    return_images.append(image)

        return torch.cat(return_images, 0)

class CycleGANLoss(nn.Module):
    """Combined loss function.

    Includes:
    - Adversartial loss (LSGAN)
    - Cycle consistency loss (L1)
    - Identity loss (L1)
    """

    def __init__(self, lambda_cycle: float = 10.0, lambda_identity: float = 0.5):  # noqa: D107
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def adversarial_loss(  # noqa: D102
        self, pred: torch.Tensor, is_real: bool
    ) -> torch.Tensor:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.mse_loss(pred, target)

    def cycle_consistency_loss(  # noqa: D102
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        return self.l1_loss(reconstructed, original)

    def identity_loss(  # noqa: D102
        self,
        identity: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        return self.l1_loss(identity, original)


class LinearLRScheduler:
    """Linear learning rate scheduler.

    Keeps LR constant for first n epochs, then linearly decays to 0.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        decay_start_epoch: int
    ):
        """Get initital LRs from optimizer param groups."""
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.decay_start_epoch = decay_start_epoch
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        """Update LRs based on current epoch."""
        if epoch < self.decay_start_epoch:
            multipliter = 1.0
        else:
            decay_epochs = self.total_epochs - self.decay_start_epoch
            elapsed_decay = epoch - self.decay_start_epoch
            multipliter = max(0.0, 1 - elapsed_decay / decay_epochs)

        for param_group, initial_lr in zip(
            self.optimizer.param_groups, self.initial_lrs
        ):
            param_group['lr'] = initial_lr * multipliter


class CycleGANTrainer:
    """Main CycleGAN training class with full experiment tracking."""

    def __init__(
        self,
        config: CycleGANConfig,
        np_rng: np.random.Generator,
        torch_rng: torch.Generator
    ):
        """Setup infrastructure components using global config."""
        self.config = config
        self.np_rng = np_rng
        self.torch_rng = torch_rng
        self.device = torch.device(config.device)

        # initialize experiment logger
        self.experiment_logger = ExperimentLogger(config.experiment_name)

        # initialize checkpoint manager
        self.checkpoint_manager = CycleGANCheckpointManager(
            checkpoint_dir=config.logs_and_checkpoints.checkpoint_dir,
            save_top_k=3,
            save_last=True
        )

        # initialize evaluator
        eval_config = EvaluatorConfig(device=config.device)
        self.evaluator = CombinedEvaluator(eval_config, self.experiment_logger)

        Path(config.logs_and_checkpoints.sample_dir).mkdir(parents=True, exist_ok=True)

        self._setup_models()
        self._setup_optimizers()
        self._setup_schedulers()
        self._setup_loss()
        self._setup_image_pools()

        self.scaler = setup_amp_scaler(
            config.optimization.mixed_precision, self.device)

        self.global_step = 0
        self.current_epoch = 0

        logger.info(f"CycleGAN trainer initialized on {self.device}")
        logger.info(f"Mixed precision: {config.optimization.mixed_precision}")

    @staticmethod
    def count_parameters(model: nn.Module):  # noqa: D102
        return sum(p.numel() for p in model.parameters())

    def _setup_models(self) -> None:
        """Initialize generator and discriminator models."""
        # Generators
        self.G_A2B = ResNetGenerator(self.config.generator).to(self.device)
        self.G_B2A = ResNetGenerator(self.config.generator).to(self.device)

        # Discriminators
        self.D_A = PatchDiscriminator(self.config.discriminator).to(self.device)
        self.D_B = PatchDiscriminator(self.config.discriminator).to(self.device)

        logger.info("Models initialized:")
        logger.info(f"  G_A2B parameters: {self.count_parameters(self.G_A2B)}")
        logger.info(f"  G_B2A parameters: {self.count_parameters(self.G_B2A)}")
        logger.info(f"  D_A parameters: {self.count_parameters(self.D_A)}")
        logger.info(f"  D_B parameters: {self.count_parameters(self.D_B)}")

    def _setup_optimizers(self) -> None:
        """Initialize optimizers for generators and discriminators."""
        self.optimizer_G = torch.optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=self.config.optimization.lr_G
        )
        self.optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(),
            lr=(
                self.config.optimization.lr_TTUR_multiplier
                * self.config.optimization.lr_D
            ),
            betas=(
                self.config.optimization.beta1,
                self.config.optimization.beta2
            )
        )
        self.optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(),
            lr=(
                self.config.optimization.lr_TTUR_multiplier
                * self.config.optimization.lr_D
            ),
            betas=(
                self.config.optimization.beta1,
                self.config.optimization.beta2
            )
        )

    def _setup_schedulers(self) -> None:
        """Initialize learning rate schedulers."""
        self.scheduler_G = LinearLRScheduler(
            self.optimizer_G,
            self.config.optimization.epochs,
            self.config.optimization.lr_decay_start,
        )
        self.scheduler_D_A = LinearLRScheduler(
            self.optimizer_D_A,
            self.config.optimization.epochs,
            self.config.optimization.lr_decay_start
        )
        self.scheduler_D_B = LinearLRScheduler(
            self.optimizer_D_B,
            self.config.optimization.epochs,
            self.config.optimization.lr_decay_start
        )

    def _setup_loss(self) -> None:
        """Initialize loss functions."""
        self.loss_fn = CycleGANLoss(
            lambda_cycle=self.config.optimization.lambda_cycle,
            lambda_identity=self.config.optimization.lambda_identity
        )

    def _setup_image_pools(self) -> None:
        self.fake_A_pool = ImagePool(self.config.optimization.pool_size)
        self.fake_B_pool = ImagePool(self.config.optimization.pool_size)

    def train_generators(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> dict[str, float]:
        """Forward-backward pass for generators with batches from domains A and B.

        Returns:
            Dictionary of generator losses
        """
        self.optimizer_G.zero_grad()

        with autocast(
            enabled=self.config.optimization.mixed_precision,
            device_type="cuda" if "cuda" in str(self.device) else "cpu"
        ):
            # A -> B -> A
            fake_B = self.G_A2B(real_A)
            recovered_A = self.G_B2A(fake_B)

            # B -> A -> B
            fake_A = self.G_B2A(real_B)
            recovered_B = self.G_A2B(fake_A)

            # identity mapping
            if self.config.optimization.lambda_identity > 0:
                identity_A = self.G_B2A(real_A)
                identity_B = self.G_A2B(real_B)
            else:
                identity_A, identity_B = None, None

            # Adversarial losses
            pred_fake_B = self.D_B(fake_B)
            pred_fake_A = self.D_A(fake_A)

            loss_G_A2B = self.loss_fn.adversarial_loss(pred_fake_B, is_real=True)
            loss_G_B2A = self.loss_fn.adversarial_loss(pred_fake_A, is_real=True)

            # Cycle consistency loss
            loss_cycle_A = self.loss_fn.cycle_consistency_loss(recovered_A, real_A)
            loss_cycle_B = self.loss_fn.cycle_consistency_loss(recovered_B, real_B)

            if identity_A is not None and identity_B is not None:
                loss_identity_A = self.loss_fn.identity_loss(identity_A, real_A)
                loss_identity_B = self.loss_fn.identity_loss(identity_B, real_B)
            else:
                loss_identity_A = loss_identity_B = torch.tensor(0.0, device=self.device)

            loss_G = (
                loss_G_A2B + loss_G_B2A
                + (
                    self.config.optimization.lambda_cycle
                    * (loss_cycle_A + loss_cycle_B)
                )
                + (
                    self.config.optimization.lambda_identity
                    * (loss_identity_A + loss_identity_B)
                )
            )

        # backward pass
        if self.scaler:
            self.scaler.scale(loss_G).backward()  # type: ignore
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            loss_G.backward()
            self.optimizer_G.step()

        return {
            'loss_G_A2B': loss_G_A2B.item(),
            'loss_G_B2A': loss_G_B2A.item(),
            'loss_cycle_A': loss_cycle_A.item(),
            'loss_cycle_B': loss_cycle_B.item(),
            'loss_identity_A': loss_identity_A.item(),
            'loss_identity_B': loss_identity_B.item(),
            'loss_G_total': loss_G.item()
        }

    def train_discriminator_A(
        self,
        real_A: torch.Tensor,
        fake_A: torch.Tensor
    ) -> float:
        """Forward-backward pass for G_A with batch of real and fake images."""
        self.optimizer_D_A.zero_grad()

        # use image pool
        fake_A = self.fake_A_pool.query(fake_A.detach())

        with autocast(
            enabled=self.config.optimization.mixed_precision,
            device_type="cuda" if "cuda" in str(self.device) else "cpu"
        ):
            pred_real = self.D_A(real_A)
            loss_D_real = self.loss_fn.adversarial_loss(pred_real, is_real=True)

            pred_fake = self.D_A(fake_A)
            loss_D_fake = self.loss_fn.adversarial_loss(pred_fake, is_real=False)

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

        if self.scaler:
            self.scaler.scale(loss_D_A).backward()  # type: ignore
            self.scaler.step(self.optimizer_D_A)
            self.scaler.update()
        else:
            loss_D_A.backward()
            self.optimizer_D_A.step()

        return loss_D_A.item()

    def train_discriminator_B(
        self,
        real_B: torch.Tensor,
        fake_B: torch.Tensor
    ) -> float:
        """Forward-backward pass for G_B with batch of real and fake images."""
        self.optimizer_D_B.zero_grad()

        # use image pool
        fake_B = self.fake_B_pool.query(fake_B.detach())

        with autocast(
            enabled=self.config.optimization.mixed_precision,
            device_type="cuda" if "cuda" in str(self.device) else "cpu"
        ):
            pred_real = self.D_B(real_B)
            loss_D_real = self.loss_fn.adversarial_loss(pred_real, is_real=True)

            pred_fake = self.D_B(fake_B)
            loss_D_fake = self.loss_fn.adversarial_loss(pred_fake, is_real=False)

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

        if self.scaler:
            self.scaler.scale(loss_D_B).backward()  # type: ignore
            self.scaler.step(self.optimizer_D_B)
            self.scaler.update()
        else:
            loss_D_B.backward()
            self.optimizer_D_B.step()

        return loss_D_B.item()

    def train_epoch(
        self, dataloader: PairedDataloader, epoch: int
    ) -> dict[str, float]:
        """Whole dataloader pass, alterntating generator/discriminator updates."""
        self.G_A2B.train()
        self.G_B2A.train()
        self.D_A.train()
        self.D_B.train()

        epoch_losses = {
            "loss_G_A2B": [],
            "loss_G_B2A": [],
            "loss_D_A": [],
            "loss_D_B": [],
            "loss_cycle_A": [],
            "loss_cycle_B": [],
            "loss_identity_A": [],
            "loss_identity_B": [],
            "loss_G_total": [],
            "loss_D_total": []
        }

        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            real_A = batch['A'].to(self.device)
            real_B = batch['B'].to(self.device)

            with torch.inference_mode():
                fake_A = self.G_B2A(real_B)
                fake_B = self.G_A2B(real_A)

            # train generators
            gen_losses = self.train_generators(real_A, real_B)

            # train discriminators
            loss_D_A = self.train_discriminator_A(real_A, fake_A)
            loss_D_B = self.train_discriminator_B(real_B, fake_B)

            for k, v in gen_losses.items():
                epoch_losses[k].append(v)

            epoch_losses["loss_D_A"].append(loss_D_A)
            epoch_losses["loss_D_B"].append(loss_D_B)
            epoch_losses["loss_D_total"].append(loss_D_A + loss_D_B)

            if batch_idx % self.config.logs_and_checkpoints.log_freq == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed

                logger.info(
                    f"Epoch [{epoch}/{self.config.optimization.epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"({batches_per_sec:.1f} batch/s) - "
                    f"G_total: {gen_losses['loss_G_total']:.4f}, "
                    f"D_total: {loss_D_A + loss_D_B:.4f}, "
                    f"Cycle: {gen_losses['loss_cycle_A'] + gen_losses['loss_cycle_B']:.4f}"
                )

                mlflow.log_metrics({
                    "batch_loss_G_total": gen_losses['loss_G_total'],
                    "batch_loss_D_total": loss_D_A + loss_D_B,
                    "batch_loss_cycle_total": (
                        gen_losses['loss_cycle_A'] + gen_losses['loss_cycle_B']
                    ),
                    "batches_per_sec": batches_per_sec
                })

            self.global_step += 1

        return {k: float(np.mean(v)) for k, v in epoch_losses.items()}

    def save_sample_images(
        self,
        dataloader: PairedDataloader,
        epoch: int,
        total: int = 4
    ):
        """Save comparison grid of original and translated imgs as MLFlow artifact."""
        self.G_A2B.eval()
        self.G_B2A.eval()

        with torch.inference_mode():
            batch = next(iter(dataloader))
            real_A = batch['A'][:total].to(self.device)
            real_B = batch['B'][:total].to(self.device)

            fake_B = self.G_A2B(real_A)
            fake_A = self.G_B2A(real_B)

            recovered_A = self.G_B2A(fake_B)
            recovered_B = self.G_A2B(fake_A)

            comparison = torch.cat([
                torch.cat([real_A, fake_B, recovered_A], dim=0),
                torch.cat([real_B, fake_A, recovered_B], dim=0)
            ], dim=0)
            comparison = 0.5 * comparison + 0.5

            print(comparison.shape)

            grid = make_grid(comparison, nrow=total, padding=2, normalize=False)

            save_path = Path(self.config.logs_and_checkpoints.sample_dir) / f"epoch_{epoch:04d}.png"
            save_image(grid, save_path)

            mlflow.log_artifact(str(save_path), "sample_images")
            save_path.unlink()

    def evaluate_model(
        self, dataloader_A: DataLoader, dataloader_B: DataLoader, epoch: int
    ) -> dict[str, float]:
        """Evaluate model using metrics.CombinedEvaluator."""
        logger.info(f"Starting evaluation at epoch: {epoch}")

        results = self.evaluator.evaluate_datasets(
            dataloader_A=dataloader_A,
            dataloader_B=dataloader_B,
            G_A2B=self.G_A2B,
            G_B2A=self.G_B2A,
            eval_device=str(self.device)
        )

        logger.info(
            "Evaluation results - "
            f"FID: {fmt3(results.get('fid'))}, "
            f"LPIPS: {fmt3(results.get('lpips'))}, "
            f"IS: {fmt3(results.get('is_mean'))}±{fmt3(results.get('is_std'))}"
        )

        return results

    def _validate_tensor_against_spec(
        self,
        tensor: torch.Tensor,
        spec: TensorSpec,
        name: str
    ):
        """A helper function to check if tensor fits the spec."""
        torch_dtype = tensor.dtype
        spec_dtype = np.dtype(spec.type)

        if torch_dtype != torch.from_numpy(np.array(0, dtype=spec_dtype)).dtype:
            raise ValueError(
                f"{name}: dtype mismatch ({torch_dtype} != {spec_dtype}) "
            )

        actual_shape = list(tensor.shape)
        expected_shape = list(spec.shape)

        if actual_shape[1:] != expected_shape[1:]:
            raise ValueError(
                f"{name}: shape mismatch({actual_shape} != {expected_shape})"
            )

    def validate_signatures(self, dataloader):
        """Perform a dry-run i/o validation against config-declared signatures."""
        gen_batch = next(iter(dataloader))

        G_A2B_signature = self.G_A2B.cfg.get_model_signature()
        G_B2A_signature = self.G_B2A.cfg.get_model_signature()

        D_A_signature = self.D_A.cfg.get_model_signature()
        D_B_signature = self.D_B.cfg.get_model_signature()

        # validate input tensors
        for name, spec in G_A2B_signature.inputs.input_dict().items():
            self._validate_tensor_against_spec(gen_batch["A"], spec, name)  # type: ignore

        for name, spec in D_A_signature.inputs.input_dict().items():
            self._validate_tensor_against_spec(gen_batch["A"], spec, name)  # type: ignore

        for name, spec in G_B2A_signature.inputs.input_dict().items():
            self._validate_tensor_against_spec(gen_batch["B"], spec, name)  # type: ignore

        for name, spec in D_B_signature.inputs.input_dict().items():
            self._validate_tensor_against_spec(gen_batch["B"], spec, name)  # type: ignore

        with torch.inference_mode():
            ouput_gen_a = self.G_A2B(gen_batch["A"].to(self.device))
            ouput_gen_b = self.G_B2A(gen_batch["B"].to(self.device))
            output_disc_a = self.D_A(gen_batch["A"].to(self.device))
            output_disc_b = self.D_B(gen_batch["B"].to(self.device))

        # validate output tensors
        for name, spec in G_A2B_signature.outputs.input_dict().items():
            self._validate_tensor_against_spec(ouput_gen_a, spec, name)  # type: ignore

        for name, spec in D_A_signature.outputs.input_dict().items():
            self._validate_tensor_against_spec(output_disc_a, spec, name)  # type: ignore

        for name, spec in G_B2A_signature.outputs.input_dict().items():
            self._validate_tensor_against_spec(ouput_gen_b, spec, name)  # type: ignore

        for name, spec in D_B_signature.outputs.input_dict().items():
            self._validate_tensor_against_spec(output_disc_b, spec, name)  # type: ignore

    def _summarize_epoch(
        self,
        epoch: int,
        models: CycleGANModels,
        state: TrainingState,
        epoch_losses: dict[str, float],
        train_dataloader: PairedDataloader,
        eval_dataloader_A: DataLoader | None,
        eval_dataloader_B: DataLoader | None,
    ):
        """Log/evaluate/save model depending on the epoch."""
        mlflow.log_metrics({
            f"epoch_{k}": v for k, v in epoch_losses.items()
        }, step=epoch)

        best_fid = state.best_fid_score

        # update training state
        state.epoch = epoch
        state.global_step = self.global_step
        for k, v in epoch_losses.items():
            state.loss_history[k].append(v)

        if epoch % self.config.logs_and_checkpoints.sample_freq == 0:
            self.save_sample_images(train_dataloader, epoch)

        eval_results = {}
        if (
            epoch % self.config.logs_and_checkpoints.eval_freq == 0
            and eval_dataloader_A is not None
            and eval_dataloader_B is not None
        ):
            eval_results = self.evaluate_model(
                eval_dataloader_A, eval_dataloader_B, epoch
            )

            if 'fid' in eval_results and eval_results['fid'] < best_fid:
                best_fid = eval_results['fid']
                state.best_fid_score = best_fid

        if (
            epoch % self.config.logs_and_checkpoints.save_freq == 0
            or epoch == self.config.optimization.epochs - 1
        ):
            is_best = (
                'fid' in eval_results
                and eval_results['fid'] <= state.best_fid_score
            )
            self.checkpoint_manager.save_checkpoint(
                models, state,
                metrics=eval_results,
                is_best=is_best
            )

        logger.info(
            f"Epoch {epoch} completed - "
            f"G_loss: {epoch_losses['loss_G_total']:.4f}, "
            f"D_loss: {epoch_losses['loss_D_total']:.4f}, "
            f"Best FID: {state.best_fid_score:.3f}"
        )

    def train(
        self,
        train_dataloader: PairedDataloader,
        eval_dataloader_A: Optional[DataLoader] = None,
        eval_dataloader_B: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """Main training loop logic."""
        self.validate_signatures(train_dataloader)

        with self.experiment_logger:

            self.experiment_logger.log_dataclass_configs(
                cyclegan_config=self.config,
                evaluation_config=self.evaluator.config
            )

            self.experiment_logger.log_experiment_tags(
                self.config.tags
            )

            models = CycleGANModels(
                G_A2B=self.G_A2B,
                G_B2A=self.G_B2A,
                D_A=self.D_A,
                D_B=self.D_B,
                optimizer_G=self.optimizer_G,
                optimizer_D_A=self.optimizer_D_A,
                optimizer_D_B=self.optimizer_D_B,
            )

            if resume_from:
                state, resumed = auto_resume_training(
                    self.checkpoint_manager, models, resume_from
                )
                if resumed:
                    self.current_epoch = state.epoch
                    self.global_step = state.global_step
                    logger.info(f"Resumed training from epoch {self.current_epoch}")
                else:
                    state = TrainingState()

            else:
                state = TrainingState()

            if (
                eval_dataloader_A is not None
                and eval_dataloader_B is not None
            ):
                self.evaluate_model(eval_dataloader_A, eval_dataloader_B, -1)

            for g in models.optimizer_D_A.param_groups:
                g["lr"] = self.config.optimization.lr_D
            for g in models.optimizer_D_B.param_groups:
                g["lr"] = self.config.optimization.lr_D
            for g in models.optimizer_G.param_groups:
                g["lr"] = self.config.optimization.lr_G

            for epoch in range(
                self.current_epoch, self.config.optimization.epochs
            ):

                self.current_epoch = epoch

                self.scheduler_G.step(epoch)
                self.scheduler_D_A.step(epoch)
                self.scheduler_D_B.step(epoch)

                mlflow.log_metrics({
                    'lr_G': self.optimizer_G.param_groups[0]['lr'],
                    'lr_D_A': self.optimizer_D_A.param_groups[0]['lr'],
                    'lr_D_B': self.optimizer_D_B.param_groups[0]['lr']
                }, step=epoch)

                epoch_losses = self.train_epoch(train_dataloader, epoch)

                self._summarize_epoch(
                    epoch,
                    models,
                    state,
                    epoch_losses,
                    train_dataloader,
                    eval_dataloader_A,
                    eval_dataloader_B,
                )

            logger.info("Training completed!")

            self.checkpoint_manager.save_inference_models(
                models,
                Path(self.config.logs_and_checkpoints.checkpoint_dir) / "final_inference_models.pth",
                metadata={
                    "training_epochs": self.config.optimization.epochs,
                    "best_fid_score": state.best_fid_score,
                    "dataset": "UTKFaces",
                    "task": "gender swap"
                }
            )


def get_dataloaders(
    dataset_male_cfg: DatasetConfig,
    dataset_female_cfg: DatasetConfig,
    RNG: torch.Generator,
) -> tuple[PairedDataloader, DataLoader, DataLoader]:
    """Instantiate dataloaders for training and validation subsets.

    Args:
        dataset_male_cfg: configuration dataclass instance for male image folder
        dataset_female_cfg: configuration dataclass instance for female image folder
        RNG: random number generator instance for the experiment
    """
    dataset_male = ImageFolder.from_config({
        'root': dataset_male_cfg.root,
        'transforms': A.Compose([
            A.Resize(dataset_male_cfg.image_size, dataset_male_cfg.image_size),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.)
        ])
    })
    dataset_female = ImageFolder.from_config({
        'root': dataset_female_cfg.root,
        'transforms': A.Compose([
            A.Resize(dataset_male_cfg.image_size, dataset_male_cfg.image_size),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.)
        ])
    })

    train_ids_male, val_ids_male = get_train_test_split_indices(
        dataset_male, split_ratio=dataset_male_cfg.train_split,
        shuffle=True, shuffler=TorchShuffler(RNG)
    )

    train_ids_female, val_ids_female = get_train_test_split_indices(
        dataset_female, split_ratio=dataset_male_cfg.train_split,
        shuffle=True, shuffler=TorchShuffler(RNG)
    )

    train_loader_male = DataLoader(
        dataset_male,
        generator=RNG,
        sampler=SubsetRandomSampler(train_ids_male, RNG),
        batch_size=dataset_male_cfg.batch_size,
        num_workers=dataset_male_cfg.num_workers,
        pin_memory=True
    )

    val_loader_male = DataLoader(
        dataset_male,
        generator=RNG,
        sampler=SubsetRandomSampler(val_ids_male, RNG),
        batch_size=dataset_male_cfg.batch_size,
        num_workers=dataset_male_cfg.num_workers,
        pin_memory=True
    )

    train_loader_female = DataLoader(
        dataset_female,
        generator=RNG,
        sampler=SubsetRandomSampler(train_ids_female, RNG),
        batch_size=dataset_female_cfg.batch_size,
        num_workers=dataset_female_cfg.num_workers,
        pin_memory=True
    )

    val_loader_female = DataLoader(
        dataset_female,
        generator=RNG,
        sampler=SubsetRandomSampler(val_ids_female, RNG),
        batch_size=dataset_female_cfg.batch_size,
        num_workers=dataset_female_cfg.num_workers,
        pin_memory=True
    )

    train_loader = PairedDataloader(
        train_loader_male, train_loader_female
    )

    return train_loader, val_loader_male, val_loader_female


def create_dummy_dataloader(
    batch_size: int = 1, num_batches: int = 100
) -> DataLoader:
    """Dummy dataloader for test run."""
    data = torch.randn(num_batches * batch_size, 3, 256, 256) * 2. - 1.
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Train CycleGAN for UTKFaces gender swap"
    )
    parser.add_argument(
        "--dataset-config-male", type=str, required=True,
        help="absolute path to YAML file with male dataset config overrides"
    )
    parser.add_argument(
        "--dataset-config-female", type=str, required=True,
        help="absolute path to YAML file with female dataset config overrides"
    )
    parser.add_argument(
        "--optimization-config", type=str, default=None,
        help="absolute path to YAML file with optimization options overrides"
    )
    parser.add_argument("-seed", type=int, default=42, help="Seed for random number generators")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="Specific checkpoint to resume from")
    parser.add_argument("--test-run", action="store_true", help="Run with dummy data for testing")

    args = parser.parse_args()

    set_global_seed(seed=int(args.seed))
    torch_rng = get_torch_rng(int(args.seed))
    numpy_rng = get_numpy_rng(int(args.seed))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

    training_config = CycleGANConfig(device=str(device))

    training_config.optimization = get_config_instance(
        OptimizationConfig, args.optimization_config
    )

    dataset_config_male = get_config_instance(
        DatasetConfig, args.dataset_config_male,
    )
    dataset_config_female = get_config_instance(
        DatasetConfig, args.dataset_config_female
    )

    (
        train_loader,
        eval_loader_A,
        eval_loader_B
    ) = get_dataloaders(
        dataset_config_male, dataset_config_female,
        RNG=torch_rng
    )

    trainer = CycleGANTrainer(
        training_config, numpy_rng, torch_rng
    )

    resume_path = args.checkpoint_path if args.resume else None
    trainer.train(
        train_dataloader=train_loader,
        eval_dataloader_A=eval_loader_A,
        eval_dataloader_B=eval_loader_B,
        resume_from=resume_path
    )


if __name__ == "__main__":
    main()
