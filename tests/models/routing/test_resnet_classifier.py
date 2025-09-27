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

"""Unit tests for models.routing.resnet_classifier.py."""


from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from torchvision import models

from configs.router_utkfaces.model import BackboneName, RouterConfig

from models.routing.resnet_classifier import (
    build_resnet_classifier,
    RouterClassifier
)

@pytest.fixture
def basic_config():  # noqa: D101
    return RouterConfig(
        backbone=BackboneName.resnet18,
        weights=None,
        num_classes=10,
        dropout=None,
        freeze_backbone=False,
        backbone_lr_mult=0.1
    )


@pytest.fixture
def config_with_dropout():  # noqa: D101
    return RouterConfig(
        backbone=BackboneName.resnet18,
        weights=True,
        num_classes=5,
        dropout=0.5,
        freeze_backbone=False,
        backbone_lr_mult=0.1
    )


@pytest.fixture
def config_frozen_backbone():  # noqa: D101
    return RouterConfig(
        backbone=BackboneName.resnet18,
        weights=True,
        num_classes=5,
        dropout=None,
        freeze_backbone=True,
        backbone_lr_mult=0.1
    )


@pytest.fixture
def mock_resnet():
    """Expose Resnet-like interface for inspection logic."""
    class DummyBackbone(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.fc = nn.Linear(in_features, 1000)

        def forward(self, x):
            return x
    return DummyBackbone(in_features=2048)


@pytest.fixture()
def patch_model_and_constructor(mocker):
    """Setup mocks for model and its constructor."""
    def _patch(model_instance):
        mock_tv_models = mocker.patch(
            "models.routing.resnet_classifier.models")

        class WeightsEnumStub:  # noqa: D101
            DEFAULT = "DEFAULT"
            IMAGENET1K_V1 = "IMAGENET1K_V1"
            IMAGENET1K_V2 = "IMAGENET1K_V2"

        weights_enum = WeightsEnumStub()
        mock_tv_models.get_model = mocker.MagicMock(
            return_value=weights_enum
        )

        # patch constructor lookup with getattr
        constructor = mocker.MagicMock(
            side_effect=lambda **kwargs: model_instance)

        def selective_getattr(obj, name, default=None):
            if obj is mock_tv_models:
                return constructor
            return getattr(obj, name, default)

        mocker.patch(
            "models.routing.resnet_classifier.getattr",
            side_effect=selective_getattr
        )

        return mock_tv_models, weights_enum, constructor

    return _patch


class TestBuildResnetClassifier:
    """Test suite for build_resnet_classifier function."""

    @pytest.mark.parametrize("backbone", list(BackboneName))
    @pytest.mark.parametrize("weights_input, expected_attr", [
        (True, "DEFAULT"),
        (False, None),
        (None, None),
        ("imagenet1k_v1", "IMAGENET1K_V1"),
        ("imagenet1k_v2", "IMAGENET1K_V2"),
        ("default", "DEFAULT")
    ])
    def test_constructor_receives_resolved_weights(
        self, backbone, weights_input, expected_attr,
        patch_model_and_constructor
    ):
        """Model class __init__ consumes resolved model weights."""
        mock_model = MagicMock()
        mock_model.fc.in_features = 512

        tv_models, weights_enum, constructor = patch_model_and_constructor(
            model_instance=mock_model
        )

        cfg = RouterConfig(
            backbone=backbone,
            weights=weights_input
        )

        result = build_resnet_classifier(cfg)

        tv_models.get_model.assert_called_once_with(backbone.name)
        constructor.assert_called_once()
        _, kwargs = constructor.call_args
        if expected_attr is None:
            assert kwargs.get("weights", None) is None
        else:
            assert kwargs["weights"] == getattr(weights_enum, expected_attr)
        assert result is mock_model

    def test_model_not_found_logs_error(
        self, basic_config,
        patch_model_and_constructor, mocker
    ):
        """Model enumeration error logs do not fail silenlty."""
        mock_model = MagicMock()
        mock_model.fc.in_features = 512

        tv_models, _, constructor = patch_model_and_constructor(
            model_instance=mock_model
        )
        tv_models.get_model.side_effect = Exception("Model not found")
        logger_mock = mocker.patch("models.routing.resnet_classifier.logger")

        result = build_resnet_classifier(basic_config)

        # error info passes to logger
        logger_mock.info.assert_any_call(
            "Could not find weight enum for model "
            f"'{str(basic_config.backbone.name)}': Model not found"
        )
        # model is still constructed with None
        # as default in basic config is None
        tv_models.get_model.assert_called_once_with(basic_config.backbone.name)
        constructor.assert_called_once()
        _, kwargs = constructor.call_args
        assert kwargs.get("weights", None) is None
        assert result is mock_model

    def test_dropout_layer_added_when_configured(
        self, config_with_dropout,
        patch_model_and_constructor, mock_resnet,
    ):
        """Dropout layer is added when respective field > 0."""
        _, _, constructor = patch_model_and_constructor(
            model_instance=mock_resnet
        )

        net = build_resnet_classifier(config_with_dropout)

        constructor.assert_called_once()
        assert isinstance(net.fc, nn.Sequential)
        layers = list(net.fc.children())
        assert len(layers) == 2
        assert isinstance(layers[0], nn.Dropout)
        assert abs(layers[0].p - 0.5) < 1e-8
        assert isinstance(layers[1], nn.Linear)
        assert layers[1].in_features == 2048
        assert layers[1].out_features == 5

    def test_no_dropout_when_disabled(
        self, basic_config,
        patch_model_and_constructor, mock_resnet,
    ):
        """No dropout if the option is unspecified."""
        _, _, constructor = patch_model_and_constructor(
            model_instance=mock_resnet
        )

        net = build_resnet_classifier(basic_config)

        constructor.assert_called_once()
        assert isinstance(net.fc, nn.Sequential)
        layers = list(net.fc.children())
        assert len(layers) == 1
        assert layers[0].in_features == 2048
        assert layers[0].out_features == 10


class TestRouterClassifier:
    """Test suite for RouterClassifier class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Save required mocks as class attributes."""
        self.mock_build_resnet = mocker.patch(
            'models.routing.resnet_classifier.build_resnet_classifier'
        )

        # create a realistic mock model, mimicking backbone structure
        self.mock_model = MagicMock()
        self.mock_model.fc = MagicMock()

        self.mock_conv_layer = MagicMock()
        self.mock_bn_layer = MagicMock()
        self.mock_relu_layer = MagicMock()
        self.mock_model.children.return_value = [
            self.mock_conv_layer, self.mock_bn_layer,
            self.mock_relu_layer, self.mock_model.fc
        ]

        # mock named_parameters for parameter group testing
        self.backbone_param1 = MagicMock()
        self.backbone_param1.requires_grad = True
        self.backbone_param2 = MagicMock()
        self.backbone_param2.requires_grad = True
        self.head_param1 = MagicMock()
        self.head_param1.requires_grad = True
        self.head_param2 = MagicMock()
        self.head_param2.requires_grad = True

        self.mock_model.named_parameters.return_value = [
            ("conv1.weight", self.backbone_param1),
            ("layer1.0.conv1.weight", self.backbone_param2),
            ("fc.weight", self.head_param1),
            ("fc.bias", self.head_param2)
        ]

        self.mock_build_resnet.return_value = self.mock_model

    def test_instantiation(self, basic_config):
        """Instance creation follows the expected logic."""
        classifier = RouterClassifier(basic_config)

        self.mock_build_resnet.assert_called_once_with(basic_config)
        assert classifier.cfg == basic_config
        assert classifier.model == self.mock_model

    def test_forward_pass(self, basic_config):
        """Forward pass delegates to model attribute."""
        classifier = RouterClassifier(basic_config)
        input_tensor = torch.randn(2, 3, 224, 224)
        expected_output = torch.randn(2, 10)
        self.mock_model.return_value = expected_output

        result = classifier.forward(input_tensor)

        self.mock_model.assert_called_once_with(input_tensor)
        assert torch.all(result == expected_output)

    def test_backbone_property(self, basic_config, mocker):
        """Backbone includes all layers sans classification head."""
        classifier = RouterClassifier(basic_config)
        sequential_mock = mocker.patch(
            "models.routing.resnet_classifier.nn.Sequential"
        )

        backbone = classifier.backbone

        expected_children = [
            self.mock_conv_layer,
            self.mock_bn_layer,
            self.mock_relu_layer,
        ]
        sequential_mock.assert_called_once_with(*expected_children)
        assert backbone == sequential_mock.return_value

    def test_head_property(self, basic_config):
        """Head is stored as fc attribute."""
        classifier = RouterClassifier(basic_config)

        head = classifier.head

        assert head == self.mock_model.fc

    def test_set_backbone_frozen_true(self, basic_config):
        """Only head parameters require grad."""
        classifier = RouterClassifier(basic_config)

        classifier.set_backbone_frozen(True)

        # backbone is frozen
        assert not self.backbone_param1.requires_grad
        assert not self.backbone_param2.requires_grad

        # head is trainable
        assert self.head_param1.requires_grad
        assert self.head_param2.requires_grad

    def test_set_backbone_frozen_false(self, basic_config):
        """Both head and backbone are trainable."""
        classifier = RouterClassifier(basic_config)

        classifier.set_backbone_frozen(True)
        classifier.set_backbone_frozen(False)

        assert self.backbone_param1.requires_grad
        assert self.backbone_param2.requires_grad
        assert self.head_param1.requires_grad
        assert self.head_param2.requires_grad

    def test_param_groups_frozen_backbone(
        self, config_frozen_backbone
    ):
        """Frozen backbone results in single parameter group."""
        classifier = RouterClassifier(config_frozen_backbone)
        self.mock_model.fc.parameters.return_value = [
            self.head_param1, self.head_param2
        ]

        groups = classifier.param_groups(base_lr=0.001, weight_decay=1e-4)

        assert len(groups) == 1
        group = groups[0]
        assert "params" in group
        assert "lr" not in group
        assert "weight_decay" not in group

    def test_param_groups_trainable_backbone(self, basic_config):
        """Trainable backbone yields two parameter groups with distinct lrs."""
        classifier = RouterClassifier(basic_config)

        groups = classifier.param_groups(base_lr=0.001, weight_decay=1e-4)

        assert len(groups) == 2

        # check backbone group
        backbone_group = groups[0]
        assert "params" in backbone_group
        assert abs(backbone_group["lr"] - 0.001 * basic_config.backbone_lr_mult) < 1e-10

        # check head group
        head_group = groups[1]
        assert "params" in head_group
        assert abs(head_group["lr"] - 0.001) < 1e-10
        assert abs(head_group["weight_decay"] - 1e-4) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
