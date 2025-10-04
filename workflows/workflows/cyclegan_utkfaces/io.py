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


"""Checkpoint I/O module for CycleGAN training workflow.

Handles saving and loading of complete training state including:
- 2 generators
- 2 discriminator
- 4 optimizers (gen_A2B, gen_B2A, disc_A, disc_B)
- 4 AMP scalers (gen_A2B, gen_B2A, disc_A, disc_B)
- 2 schedulers (gen_schedulers, disc_schedulers)
- Training metadata (epoch, step, losses, metrics)

Supports both full checkpoint and inference-only model exports
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)

CHECKPOINT_SIGNATURE = set([
    'G_A2B_state_dict', 'G_B2A_state_dict',
    'D_A_state_dict', 'D_B_state_dict',
    'optimizer_G_state_dict',
    'optimizer_D_A_state_dict',
    'optimizer_D_B_state_dict'
])

CHECKPOINT_VERSION = "1.0"


@dataclass
class CycleGANModels:
    """Container for all CycleGAN model components."""
    # Generators
    G_A2B: nn.Module
    G_B2A: nn.Module

    # Discriminators
    D_A: nn.Module
    D_B: nn.Module

    # Optimizers
    optimizer_G: Optimizer
    optimizer_D_A: Optimizer
    optimizer_D_B: Optimizer

    # Schedulers
    scheduler_G: Optional[_LRScheduler] = None
    scheduler_D: Optional[_LRScheduler] = None


@dataclass
class TrainingState:
    """Training state metadata."""
    epoch: int = 0
    global_step: int = 0
    best_fid_score: float = float('inf')
    best_lpips_score: float = float('inf')

    loss_history: dict[str, list] = field(
        default_factory=lambda: {
            'loss_G_A2B': [],
            'loss_G_B2A': [],
            'loss_D_A': [],
            'loss_D_B': [],
            'loss_cycle_A': [],
            'loss_cycle_B': [],
            'loss_identity_A': [],
            'loss_identity_B': [],
            'loss_G_total': [],
            'loss_D_total': []
        }
    )

    # Additional metadata
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    pytorch_version: str = field(
        default_factory=lambda: torch.__version__
    )
    device_info: str = field(default_factory=str)


class CycleGANCheckpointManager:
    """Manages saving and loading of CycleGAN checkpoints."""
    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_top_k: int = 3,
        save_last: bool = True
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_top_k: Keep top K best checkpoints based on FID score
            save_last: Always save the latest checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_top_k = save_top_k
        self.save_last = save_last

        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        models: CycleGANModels,
        state: TrainingState,
        metrics: Optional[dict[str, float]] = None,
        is_best: bool = False,
        checkpoint_name: Optional[str] = None
    ) -> Path:
        """Save complete training checkpoints.

        Args:
            models: all model components
            state: training state metadata
            metrics: current metrics (FID, LPIPS, etc.)
            is_best: whether this is the best checkpoint so far
            checkpoint_name: optional custom checkpoint name

        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = (
                f"checkpoint_epoch_{state.epoch:04d}_step_"
                f"{state.global_step:04d}.pth"
            )

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # prepare checkpoint data
        checkpoint_data = {
            # Model state dicts
            "G_A2B_state_dict": models.G_A2B.state_dict(),
            "G_B2A_state_dict": models.G_B2A.state_dict(),
            "D_A_state_dict": models.D_A.state_dict(),
            "D_B_state_dict": models.D_B.state_dict(),

            # Optimizer state dicts
            "optimizer_G_state_dict": models.optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": models.optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": models.optimizer_D_B.state_dict(),

            # Scheduler state dicts
            "scheduler_G_state_dict": (
                models.scheduler_G.state_dict() if models.scheduler_G else None
            ),
            "scheduler_D_state_dict": (
                models.scheduler_D.state_dict() if models.scheduler_D else None
            ),

            # training state
            "training_state": state.__dict__,

            # current metrics
            "metrics": metrics or {},

            # additional metadata
            "checkpoint_version": CHECKPOINT_VERSION,
            "save_timestamp": datetime.now().isoformat(),
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

            if is_best:
                best_path = self.checkpoint_dir / "best_checkpoint.pth"
                self._create_symlink(checkpoint_path, best_path)
                logger.info(f"Best checkpoint updated: {best_path}")

            if self.save_last:
                last_path = self.checkpoint_dir / "last_checkpoint.pth"
                self._create_symlink(checkpoint_path, last_path)

            # Cleanup old checkpoints if needed:
            if self.save_top_k > 0:
                self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(
        self,
        models: CycleGANModels,
        checkpoint_path: str | Path,
        map_location: Optional[str] = None,
        strict: bool = True
    ) -> TrainingState:
        """Load complete training checkpoint.

        Args:
            models: Model components to load state into
            checkpoint_path: Path to checkpoint file
            map_location: Device mapping for loading
            strict: whether to strictly enforce state dict keys match
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        try:
            checkpoint_data = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=False
            )

            self._verify_checkpoint_compatibility(checkpoint_data)

            models.G_A2B.load_state_dict(checkpoint_data["G_A2B_state_dict"], strict=strict)
            models.G_B2A.load_state_dict(checkpoint_data["G_B2A_state_dict"], strict=strict)
            models.D_A.load_state_dict(checkpoint_data["D_A_state_dict"], strict=strict)
            models.D_B.load_state_dict(checkpoint_data["D_B_state_dict"], strict=strict)

            models.optimizer_G.load_state_dict(checkpoint_data["optimizer_G_state_dict"])
            models.optimizer_D_A.load_state_dict(checkpoint_data["optimizer_D_A_state_dict"])
            models.optimizer_D_B.load_state_dict(checkpoint_data["optimizer_D_B_state_dict"])

            if models.scheduler_G and checkpoint_data.get("scheduler_G_state_dict"):
                models.scheduler_G.load_state_dict(checkpoint_data["scheduler_G_state_dict"])
            if models.scheduler_D and checkpoint_data.get("scheduler_D_state_dict"):
                models.scheduler_D.load_state_dict(checkpoint_data["scheduler_D_state_dict"])

            state_dict = checkpoint_data.get("training_state", {})
            state = TrainingState(**{
                k: v for k, v in state_dict.items()
                if k in TrainingState.__dataclass_fields__
            })

            logger.info(
                f"Checkpoint loaded successfully. "
                f"Epoch: {state.epoch}, Step: {state.global_step}"
            )
            return state

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def save_inference_models(
        self,
        models: CycleGANModels,
        save_path: str | Path,
        metadata: Optional[dict[str, Any]] = None
    ) -> Path:
        """Save generators for inference only.

        Args:
            models: Model components
            save_path: Path to save inference models
            metadata: additional metadata to save

        Returns:
            Path to saved inference models
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        inference_data = {
            "G_A2B_state_dict": models.G_A2B.state_dict(),
            "G_B2A_state_dict": models.G_B2A.state_dict(),
            "model_config": {
                "G_A2B_class": models.G_A2B.__class__.__name__,
                "G_B2A_class": models.G_B2A.__class__.__name__
            },
            "inference_metadata": metadata or {},
            "export_timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__
        }

        try:
            torch.save(inference_data, save_path)
            logger.info(f"Inference models saved: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save inference models: {e}")
            raise

    def load_inference_models(
        self,
        G_A2B: nn.Module,
        G_B2A: nn.Module,
        checkpoint_path: str | Path,
        map_location: Optional[str] = None
    ) -> dict[str, Any]:
        """Load generators for inference only.

        Args:
            G_A2B: Generator A2B model instance
            G_B2A: Generator B2A model instance
            checkpoint_path: Path to inference checkpoint
            map_location: Device mapping

        Returns:
            loaded metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Inference checkpoint not found: {checkpoint_path}"
            )

        logger.info(f"Loading inference models: {checkpoint_path}")

        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

            G_A2B.load_state_dict(checkpoint_data["G_A2B_state_dict"])
            G_B2A.load_state_dict(checkpoint_data["G_B2A_state_dict"])

            G_A2B.eval()
            G_B2A.eval()

            metadata = checkpoint_data.get("inference_metadata", {})
            logger.info("Inference models loaded successfully")

            return metadata

        except Exception as e:
            logger.error(f"Failed to loaded inference models: {e}")
            raise

    def list_checkpoints(self) -> dict[str, Path]:
        """List all available checkpoints.

        Returns:
            Dictionary mapping checkpoint names to paths
        """
        checkpoints = {}

        if self.checkpoint_dir.exists():
            for checkpoint_file in self.checkpoint_dir.glob("*.pth"):
                if checkpoint_file.is_file():
                    checkpoints[checkpoint_file.stem] = checkpoint_file

        return checkpoints

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the most recent checkpoint."""
        last_checkpoint = self.checkpoint_dir / "last_checkpoint.pth"
        if last_checkpoint.exists():
            return last_checkpoint.resolve()

        # Fallback: find latest by modification time
        return Path(
            max(self.list_checkpoints().values(), key=lambda p: p.stat().st_mtime)
        )

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        best_checkpoint = self.checkpoint_dir / "best_checkpoint.pth"

        if best_checkpoint.exists():
            return best_checkpoint.resolve()

        return None

    def _create_symlink(self, target: Path, link: Path) -> None:
        """Create a symlink, removing existing one if needed."""
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target.name)
        except OSError as e:
            logger.warning(f"Failed to create symlink {link}: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the top K."""
        if self.save_top_k <= 0:
            return

        checkpoints = [
            p for p in self.list_checkpoints().values()
            if p.is_file() and not p.is_symlink()
        ]

        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for checkpoint in checkpoints[self.save_top_k:]:
            try:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
            except OSError as e:
                logger.warning(f"Failed to remove old checkpoint {checkpoint}: {e}")

    def _verify_checkpoint_compatibility(
        self,
        checkpoint_data: dict[str, Any]
    ):
        """Validate checkpoint candidate against minimal set of required keys."""
        version = checkpoint_data.get("checkpoint_verions", CHECKPOINT_VERSION)

        if version != "1.0":
            logger.warning(
                f"Loading checkpoint with version {version}"
                f", expected {CHECKPOINT_VERSION}"
            )

        missing_keys = [k for k in CHECKPOINT_SIGNATURE if k not in checkpoint_data]
        if missing_keys:
            raise KeyError(f"Checkpoint missing required keys: {missing_keys}")


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    # create model and optimizer mocks
    G_A2B = nn.Linear(10, 10)
    G_B2A = nn.Linear(10, 10)
    D_A = nn.Linear(10, 1)
    D_B = nn.Linear(10, 1)

    optimizer_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002)
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002)

    # create model container
    models = CycleGANModels(
        G_A2B=G_A2B,
        G_B2A=G_B2A,
        D_A=D_A, D_B=D_B,
        optimizer_G=optimizer_G,
        optimizer_D_A=optimizer_D_A,
        optimizer_D_B=optimizer_D_B
    )

    manager = CycleGANCheckpointManager("./checkpoints")

    state = TrainingState(epoch=10, global_step=1000)
    checkpoint_path = manager.save_checkpoint(models, state, is_best=True)

    checkpoints = manager.list_checkpoints()
    print(f"Available checkpoints: {list(checkpoints.keys())}")
