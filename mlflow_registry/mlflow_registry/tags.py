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

"""Script that establishes tag constants used in project.

Tag lists are provided with the set of enums.
"""

from enum import Enum
from typing import Union


class strEnum(str, Enum):
    """Class that modifies default enum member conversion to string."""
    def __str__(self):  # noqa
        return self.value


class TagKeys(strEnum):
    """Meta enum specifying all tag lookup keys."""
    TAG_TYPE = "type"
    TAG_ROLE = "role"
    TAG_STAGE = "stage"
    TAG_VERSION = "version"
    TAG_MODEL_FAMILY = "model_family"
    TAG_ARTIFACT_ROLE = "artifact_role"
    TAG_TEMPLATE_KIND = "template_kind"

class Type(strEnum):
    """Types of artifact groups stored in registry."""
    THIRD_PARTY = "third_party"
    MODEL = "model"
    TEMPLATE = "template"
    METADATA = "metadata"
    DATASET = "dataset"
    MISC = "misc"

class Role(strEnum):
    """Possible roles attributed to the artifact set."""
    LANDMARK_DETECTOR = "landmark_detector"
    ALIGNMENT_TEMPLATE = "alignment_template"
    LANDMARK_MANIFEST = "landmark_manifest"
    ALIGNED_DATASET = "aligned_dataset"
    PREVIEW = "functionality_preview"
    CLASSIFICATION_MODEL = "classification_model"

class Stage(strEnum):
    """Project lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ModelFamily(strEnum):
    """Network family."""
    RESNET_CLASSIFIERS = "resnet_classifiers"


TagValues = Union[Type, Role, Stage, ModelFamily]
