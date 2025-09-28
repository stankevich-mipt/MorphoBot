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


"""Unit tests for workflows/common/config module.

Coverage:
    - Full / partial instantiation of linear / nested configs
    - Handling extra YAML fields
    - Handling of file opening / YAML reading errors
"""

from dataclasses import dataclass, field
import os
import tempfile
from typing import Optional

import pytest
from workflows.common.config import get_config_instance
import yaml


@dataclass
class LinearConfig:
    """Simple linear config mock."""
    dataset_path: str = "data/default"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class RotationConfig:  # noqa 
    rotation_degrees: tuple[int, int] = (0, 15)
    p: float = 0.5


@dataclass
class HorizontalFlipConfig:  # noqa
    p: float = 0.5


@dataclass
class BrightSatHueConfig:  # noqa
    brightness: float = 0.1
    saturation: float = 0.1
    hue: float = 0.1
    p: float = 0.5


@dataclass
class NestedConfig:
    """Nested config mock."""
    rotation: RotationConfig = field(
        default_factory=RotationConfig
    )
    color_shakeup: BrightSatHueConfig = field(
        default_factory=BrightSatHueConfig
    )
    flip: HorizontalFlipConfig = field(
        default_factory=HorizontalFlipConfig
    )
    mean: Optional[tuple[float, float, float]] = None
    std: Optional[tuple[float, float, float]] = None


@dataclass
class ConfigWithNoDefaults:
    """Config mock with no defaults."""
    experiment_name: str
    seed: int
    device: str
    log_dir: str


class TestGetConfigInstance:
    """Test suite for get_config_instance function."""

    def test_linear_config_defaults(self):
        """Linear config instantiation with defaults works as expected."""
        config = get_config_instance(LinearConfig)

        assert isinstance(config, LinearConfig)
        assert config.dataset_path == "data/default"
        assert config.train_split == 0.7
        assert config.val_split == 0.15
        assert config.test_split == 0.15
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.shuffle is True

    def test_linear_config_yaml_override(self):
        """Provided YAML overrides default fields in linear config."""
        yaml_content = {
            "dataset_path": "/mnt/ssd/datasets/imagenet",
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "batch_size": 128,
            "num_workers": 8,
            "shuffle": False
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = get_config_instance(LinearConfig, temp_path)
            assert config.dataset_path == "/mnt/ssd/datasets/imagenet"
            assert config.train_split == 0.8
            assert config.val_split == 0.1
            assert config.test_split == 0.1
            assert config.batch_size == 128
            assert config.num_workers == 8
            assert config.shuffle is False
        finally:
            os.unlink(temp_path)

    def test_linear_config_partial_yaml(self):
        """Partial overrides replace intended fields."""
        yaml_content = {
            'dataset_path': '/data/custom',
            'batch_size': 64
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = get_config_instance(LinearConfig, temp_path)
            assert config.dataset_path == '/data/custom'
            assert config.batch_size == 64
            # Defauls preserved
            assert config.train_split == 0.7
            assert config.val_split == 0.15
            assert config.test_split == 0.15
            assert config.num_workers == 4
            assert config.shuffle is True
        finally:
            os.unlink(temp_path)

    def test_nested_config_defaults(self):
        """Nested config instantiation with defaults works as expected."""
        config = get_config_instance(NestedConfig)

        assert isinstance(config, NestedConfig)
        assert isinstance(config.rotation, RotationConfig)
        assert isinstance(config.color_shakeup, BrightSatHueConfig)
        assert isinstance(config.flip, HorizontalFlipConfig)

        # Check nested defaults
        assert config.rotation.rotation_degrees == (0, 15)
        assert config.rotation.p == 0.5
        assert config.color_shakeup.brightness == 0.1
        assert config.color_shakeup.saturation == 0.1
        assert config.color_shakeup.hue == 0.1
        assert config.color_shakeup.p == 0.5
        assert config.flip.p == 0.5

        # Optional fields
        assert config.mean is None
        assert config.std is None

    def test_nested_config_yaml_override(self):
        """YAML overrides take priority for nested config init."""
        yaml_content = {
            'rotation': {
                'rotation_degrees': [0, 45],
                'p': 0.8
            },
            'color_shakeup': {
                'brightness': 0.3,
                'saturation': 0.2,
                'hue': 0.0,
                'p': 0.9
            },
            'flip': {
                'p': 0.3
            },
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = get_config_instance(NestedConfig, temp_path)

            assert config.rotation.rotation_degrees == (0, 45)
            assert config.rotation.p == 0.8
            assert config.color_shakeup.brightness == 0.3
            assert config.color_shakeup.saturation == 0.2
            assert config.color_shakeup.hue == 0.0
            assert config.color_shakeup.p == 0.9
            assert config.flip.p == 0.3
            assert config.mean == (0.485, 0.456, 0.406)
            assert config.std == (0.229, 0.224, 0.225)

        finally:
            os.unlink(temp_path)

    def test_nested_config_partial_yaml(self):
        """Parital overrides work with nested configs."""
        yaml_content = {
            'rotation': {
                'p': 0.9
            },
            'mean': [0.5, 0.5, 0.5]
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = get_config_instance(NestedConfig, temp_path)

            # partial override
            assert config.rotation.rotation_degrees == (0, 15)
            assert config.rotation.p == 0.9

            # other defaults preserved
            assert config.color_shakeup.brightness == 0.1
            assert config.flip.p == 0.5

            # explicit override
            assert config.mean == (0.5, 0.5, 0.5)
            assert config.std is None
        finally:
            os.unlink(temp_path)

    def test_config_with_no_defauls_yaml(self):
        """Config with no defaults requires full YAML fields."""
        yaml_content = {
            'experiment_name': 'test_experiment',
            'seed': 12345,
            'device': 'cuda:1',
            'log_dir': '/tmp/logs'
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = get_config_instance(ConfigWithNoDefaults, temp_path)
            assert config.experiment_name == 'test_experiment'
            assert config.seed == 12345
            assert config.device == 'cuda:1'
            assert config.log_dir == '/tmp/logs'
        finally:
            os.unlink(temp_path)

    def test_file_not_found_error(self):
        """Raises if YAML doesn't exist."""
        with pytest.raises(FileNotFoundError):
            get_config_instance(LinearConfig, '/path/that/does/not/exist.yaml')

    def test_invalid_yaml_content(self):
        """Raises if YAML is malformed."""
        invalid_yaml = "dataset_path: /data\ntrain_split: 0.8\n  invalid_indent: true"

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write(invalid_yaml)
            temp_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                get_config_instance(LinearConfig, temp_path)
        finally:
            os.unlink(temp_path)

    def test_empty_yaml_file(self):
        """Empty YAML results in defaults."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write("")
            temp_path = f.name

        try:

            config = get_config_instance(LinearConfig, temp_path)
            assert config.dataset_path == "data/default"
            assert config.train_split == 0.7
            assert config.batch_size == 32
        finally:
            os.unlink(temp_path)

    def test_yaml_with_extra_fields(self):
        """Config does not inherit out-of-domain fields."""
        yaml_content = {
            'dataset_path': '/data/test',
            'batch_size': 16,
            'extra_field': 'should_be_ignored',
            'unknown_param': 42
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = get_config_instance(LinearConfig, temp_path)
            assert config.dataset_path == '/data/test'
            assert config.batch_size == 16
            # extra fields do not exist
            assert not hasattr(config, 'extra_field')
            assert not hasattr(config, 'unknown_param')
        finally:
            os.unlink(temp_path)

    def test_mock_file_open_error(self, mocker):
        """Inaccessible YAML is properly handled."""
        mocker.patch(
            'builtins.open',
            side_effect=PermissionError('Permission denied')
        )
        with pytest.raises(PermissionError):
            get_config_instance(LinearConfig, 'resricted/filed.yaml')

    def test_mock_yaml_load_error(self, mocker):
        """YAML loading error is propery handled."""
        mock_file = mocker.mock_open(read_data='invalid: yaml: content')
        mocker.patch("builtins.open", mock_file)
        mocker.patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML"))

        with pytest.raises(yaml.YAMLError):
            get_config_instance(LinearConfig, '/fake/config.yaml')
