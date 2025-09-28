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


"""Configuration utilities applicable across different workflows."""

from collections.abc import Mapping
from dataclasses import (
    fields, is_dataclass, MISSING
)
from typing import Optional, Type, TypeVar

from serde import from_dict
import yaml

T = TypeVar('T')


def dataclass_defaults(cls):
    """Build dataclass defaults witout instantiation."""
    out = {}
    for f in fields(cls):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:
            candidate = f.default_factory()
            if is_dataclass(candidate):
                out[f.name] = dataclass_defaults(type(candidate))
            else:
                out[f.name] = candidate

    return out


def deep_merge(base, override):
    """Recursive merge for nested configs."""
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        return override
    merged = dict(base)
    for k, v in override.items():
        merged[k] = deep_merge(merged[k], v) if k in merged else v
    return merged


def get_config_instance(
    config_cls: Type[T],
    yaml_path: Optional[str] = None
) -> T:
    """Get dataclass instance that adheres to config_cls schema.

    If optional yaml_path parameter is provided,
    dataclass fields are overriden with those in YAML.

    Args:
        config_cls: schema-defining class
        yaml_path: absolute path to YAML config modifier.

    """
    raw_cfg = {}

    if yaml_path:
        with open(yaml_path) as f:
            raw_cfg = yaml.safe_load(f)

    base_defaults = dataclass_defaults(config_cls)
    merged = deep_merge(base_defaults, {} if raw_cfg is None else raw_cfg)

    return from_dict(config_cls, merged)
