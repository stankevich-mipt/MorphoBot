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


"""Resnet backbone + classification head from torchvision."""

from __future__ import annotations

import logging
from typing import Iterable

from configs.router_utkfaces.model import RouterConfig
import torch
import torch.nn as nn
from torchvision import models


logger = logging.getLogger(__name__)


def build_resnet_classifier(
    config: RouterConfig
) -> nn.Module:
    """Create torchvision.resnet and replace final FC head."""
    model_name = str(config.backbone.name)

    try:
        weights_enum = models.get_model(model_name)
        logger.info(f"Using '{model_name}' as a backbone")

    except Exception as e:
        weights_enum = None
        logger.info(f"Could not find weight enum for model '{model_name}': {e}")

    raw_weights = config.weights
    if raw_weights is True:
        tv_weights = "DEFAULT"
    elif raw_weights is False or raw_weights is None:
        tv_weights = None
    elif isinstance(raw_weights, str):
        tv_weights = raw_weights.upper() if raw_weights in (
            "imagenet1k_v1", "imagenet1k_v2", "default"
        ) else raw_weights
    else:
        tv_weights = raw_weights

    tv_weights_resolved = None
    if weights_enum is not None and isinstance(tv_weights, str):
        tv_weights_resolved = getattr(weights_enum, tv_weights, None)
        if tv_weights_resolved is None and hasattr(weights_enum, "DEFAULT"):
            tv_weights_resolved = weights_enum.DEFAULT
    if tv_weights_resolved is None:
        tv_weights_resolved = tv_weights

    # that will likely fail with ValueError if
    # tv_weights_resolved is not from the supported enum,
    # which is fine for dev code, but should be iterated upon later
    net = getattr(models, model_name)(weights=tv_weights_resolved)
    in_features = net.fc.in_features
    head: list[nn.Module] = (
        [nn.Linear(in_features, config.num_classes)]
    )
    if hasattr(config, 'dropout') and config.dropout and config.dropout > 0:
        head = [nn.Dropout(p=float(config.dropout))] + head

    net.fc = nn.Sequential(*head)
    return net


class RouterClassifier(nn.Module):
    """Resnet-based classifier with helpers for param groups."""

    def __init__(self, cfg: RouterConfig):
        """Construct from RouterConfig."""
        super().__init__()
        self.cfg: RouterConfig = cfg
        self.model: nn.Module = build_resnet_classifier(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple through the whole model."""
        return self.model(x)

    @property
    def backbone(self) -> nn.Module:
        """Everything except the fine-tuning head."""
        return nn.Sequential(*list(self.model.children())[:-1])

    @property
    def head(self) -> nn.Module:
        """Classification head module (shallow FC stack)."""
        return self.model.fc

    def set_backbone_frozen(self, frozen: bool = True) -> None:
        """Enable/disable grad calulation for backbone weights."""
        for name, param in self.model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = not frozen

        for p in self.head.parameters():
            p.requires_grad = True

    def param_groups(
        self, base_lr: float, weight_decay: float = 1e-4
    ):
        """Group parameters by backbone/head, set different LRs."""
        bb_parameters: Iterable[nn.Parameter] = (
            p for n, p in self.model.named_parameters()
            if not n.startswith("fc.")
        )
        head_parameters: Iterable[nn.Parameter] = (
            p for n, p in self.model.named_parameters()
            if n.startswith("fc.")
        )

        groups: list[dict] = []

        if self.cfg.freeze_backbone:

            self.set_backbone_frozen(True)
            groups.append({
                "params": list(
                    param for param in head_parameters
                    if param.requires_grad
                )
            })

        else:

            trainable_bb = [
                p for p in bb_parameters if p.requires_grad
            ]
            trainable_head = [
                p for p in head_parameters if p.requires_grad
            ]

            bb_lr = max(0.0, float(self.cfg.backbone_lr_mult) * base_lr)

            if trainable_bb:
                groups.append({
                    "params": trainable_bb,
                    "lr": bb_lr,
                    "weight_decay": weight_decay
                })
            if trainable_head:
                groups.append({
                    "params": trainable_head,
                    "lr": base_lr,
                    "weight_decay": weight_decay
                })

        return groups
