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


"""CLI script for gender classificator training.

Artifacts produced during script runtime are pushed
to the MLFlow registry under a new run in scope
of the experiment experiment-name; if registry is
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
    --experiment-name router-utkfaces-baseline
    --dataset-config /path/to/dataset/config.yaml
    --seed 42
"""

import argparse
import contextlib
from dataclasses import asdict
import logging
from pathlib import Path
import tempfile
from typing import Any


from configs.router_utkfaces import (
    AugmentationConfig,
    DatasetConfig,
    RouterConfig,
    TrainingConfig
)
from data.utils import get_train_test_split_indices
from data.utkfaces_aligned import UTKFacesDataset
import mlflow
from mlflow_registry.tag_profiles import TAG_PROFILES
import numpy.typing as npt
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from workflows.common.config import get_config_instance
from workflows.common.data import TorchShuffler
from workflows.common.mlflow_experiment_logger import ExperimentLogger
from workflows.common.training import (
    get_torch_rng,
    metrics_to_str,
    set_global_seed,
    setup_amp_scaler,
)


from models.routing import RouterClassifier
from .augmentations import build_kornia_stack, build_mixup
from .io import save_checkpoint


def train_routing_for_single_epoch(
    epoch: int,
    model: RouterClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    aug_dict: dict[str, Any],
    device: torch.device,
    scaler: torch.cuda.amp.grad_scaler.GradScaler | None = None,
) -> tuple[float, float]:
    """Train routing model for one epoch, return metrics."""
    model.train()

    logger = logging.getLogger()

    running_loss = 0.0
    predictions = []
    targets = []

    for batch_idx, (images, labels) in enumerate(loader):

        images, labels = (
            images.to(device, dtype=torch.float32),
            labels.to(device, dtype=torch.long)
        )

        images = aug_dict["kornia_stack"](images)
        images, labels = aug_dict["mixup"](images, labels)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=str(device)):
            logits = model(images)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(labels * log_probs).sum(dim=1).mean()

        if scaler is not None:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Accumulate metrics
        running_loss += loss.item()
        predictions.extend(
            logits.argmax(dim=1).cpu().numpy())
        targets.extend(
            labels.argmax(dim=1).cpu().numpy()
        )
        if batch_idx % 100 == 0:
            logger.info((
                f"Batch {batch_idx}/{len(loader)} "
                f"Loss: {loss.item():.4f}"
            ))

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = accuracy_score(
        targets, predictions
    )

    return epoch_loss, float(epoch_accuracy)


def validate_routing(
    epoch: int,
    model: RouterClassifier,
    loader: DataLoader,
    device: torch.device
) -> tuple[float, float, list[npt.NDArray], list[npt.NDArray]]:
    """Evaluate routing model, return metrics."""
    model.eval()

    predictions = []
    targets = []
    running_loss = 0.

    with torch.inference_mode():

        for images, labels in loader:
            images, labels = (
                images.to(device, dtype=torch.float32),
                labels.to(device, dtype=torch.long)
            )
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            # Accumulate metrics
            running_loss += loss.item()
            predictions.extend(
                logits.argmax(dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(targets, predictions)

    return epoch_loss, float(epoch_acc), predictions, targets


def get_dataloaders(
    dataset_cfg: DatasetConfig,
    RNG: torch.Generator,
):
    """Instantiate dataloaders for training and validation subsets.

    Args:
        dataset_cfg: configuration dataclass
        RNG: random number generator instance for the experiment
    """
    dataset = UTKFacesDataset.from_config({
        'root_male': dataset_cfg.root_male,
        'root_female': dataset_cfg.root_female,
    })

    train_ids, val_ids = get_train_test_split_indices(
        dataset, split_ratio=dataset_cfg.train_split,
        shuffle=True, shuffler=TorchShuffler(RNG)
    )

    train_loader = DataLoader(
        dataset,
        generator=RNG,
        sampler=SubsetRandomSampler(train_ids, RNG),
        batch_size=dataset_cfg.batch_size,
        num_workers=dataset_cfg.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset,
        generator=RNG,
        sampler=SubsetRandomSampler(val_ids, RNG),
        batch_size=dataset_cfg.batch_size,
        num_workers=dataset_cfg.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def parse_args():
    """Process CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name", type=str, required=True,
        help="MLFlow experiment name"
    )
    parser.add_argument(
        "--dataset-config", type=str, required=True,
        help="absolute path to YAML file with dataset config overrides"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="optional path for local storage of produced artifacts"
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="absolute path to YAML file with model config overrides"
    )
    parser.add_argument(
        "--augmentation-config", type=str, default=None,
        help="absolute path to YAML file with augmentation config overrides"
    )
    parser.add_argument(
        "--training_config", type=str, default=None,
        help="absolute path to YAML file with training config overrides"
    )

    return parser.parse_args()


def setup_logging(level=logging.INFO):  # noqa: D103
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(ch)


def main():  # noqa

    setup_logging()
    logger = logging.getLogger()
    args = parse_args()

    # seed everything, get random number generator
    set_global_seed(seed=int(args.seed))
    RNG = get_torch_rng(int(args.seed))

    # setup path if needed
    existing_output_dir = None
    if args.output_dir is not None:
        existing_output_dir = Path(args.output_dir)
        existing_output_dir.mkdir(parents=True, exist_ok=True)

    # assemble model
    model_cfg = get_config_instance(RouterConfig, args.model_config)
    model = RouterClassifier(model_cfg)

    # create dataloaders
    dataset_cfg = get_config_instance(DatasetConfig, args.dataset_config)
    train_loader, val_loader = get_dataloaders(dataset_cfg, RNG=RNG)

    # load augmentations
    aug_train_cfg = get_config_instance(
        AugmentationConfig, args.augmentation_config)
    aug_train = {
        "kornia_stack": build_kornia_stack(aug_train_cfg),
        "mixup": build_mixup(aug_train_cfg)
    }

    # setup parameter wrappers
    train_cfg = get_config_instance(TrainingConfig, args.training_config)

    if (torch.cuda.is_available() and train_cfg.device == "cuda"):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    param_groups = model.param_groups(
        base_lr=train_cfg.optim_lr,
        weight_decay=train_cfg.optim_weight_decay,
    )
    optimizer = AdamW(param_groups)
    scaler = setup_amp_scaler(enabled=train_cfg.use_amp, device=device)

    with (
        tempfile.TemporaryDirectory()
        if existing_output_dir is None
        else contextlib.nullcontext(existing_output_dir)
    ) as output_dir, ExperimentLogger(args.experiment_name) as mlflow_logger:

        mlflow_logger.log_experiment_tags(TAG_PROFILES["routing_model"])
        mlflow_logger.log_dataclass_configs(
            model=model_cfg,
            dataset=dataset_cfg,
            augmentation=aug_train_cfg,
            train=train_cfg
        )

        output_dir = Path(output_dir)
        best_val_acc = 0.0
        best_model_path = None

        logger.info(f"Starting training for {train_cfg.epochs} epochs")

        for epoch in range(1, train_cfg.epochs + 1):
            train_loss, train_acc = train_routing_for_single_epoch(
                epoch, model, train_loader, optimizer, aug_train, device, scaler
            )
            val_loss, val_acc, val_preds, val_targets = (
                validate_routing(epoch, model, val_loader, device)
            )

            all_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }

            logger.info(metrics_to_str(all_metrics, epoch=epoch))

            mlflow.log_metrics(all_metrics, step=epoch)

            if val_acc >= best_val_acc:

                best_val_acc = val_acc
                best_model_path = output_dir / "best_model.pt"
                save_checkpoint(
                    model=model,
                    path=best_model_path,
                    config=asdict(model_cfg),
                    epoch=epoch,
                    val_acc=best_val_acc
                )
                mlflow.log_artifact(str(best_model_path), "best_model")

                report = classification_report(
                    val_targets, val_preds,
                    target_names=["male", "female"],
                    output_dict=True
                )

                precision_male: float = report["male"]["precision"]  # type: ignore
                precision_female: float = report["female"]["precision"]  # type: ignore
                recall_male: float = report["male"]["recall"]  # type: ignore
                recall_female: float = report["female"]["recall"]  # type: ignore

                mlflow.log_metrics({
                    "best_val_acc": best_val_acc,
                    "precision_male": precision_male,
                    "precision_female": precision_female,
                    "recall_male": recall_male,
                    "recal_female": recall_female
                })

                logger.info(f"New best validation accuracy: {best_val_acc:.4f}")

        if best_model_path is not None:
            # reassemble best model and save it with mlflow
            model.load_state_dict(
                torch.load(best_model_path, map_location=device)["state_dict"]
            )
            input_example = next(iter(val_loader))[0].to(device)
            mlflow_logger.log_pytorch_model(
                model, model_cfg.model_name, input_example=input_example
            )

        logger.info(
            "Training completed. "
            f"Best validation accuracy: {best_val_acc:.4f}"
        )


if __name__ == "__main__":  # noqa 
    main()
