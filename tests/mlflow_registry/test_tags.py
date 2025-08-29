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


"""Unit tests for mlflow_registry.tags module.

Tests cover:
    - Presence
    - Uniqueness
    - Behaviour consistent with str-enum pattern
"""

from mlflow_registry import tags
import pytest


def test_tagkeys_enum_members_and_values():
    """Enum TagKeys contains expected members with correct string values."""
    assert tags.TagKeys.TAG_TYPE == "type"
    assert tags.TagKeys.TAG_ROLE == "role"
    assert tags.TagKeys.TAG_STAGE == "stage"
    assert isinstance(tags.TagKeys.TAG_TYPE, tags.TagKeys)
    assert str(tags.TagKeys.TAG_MODEL_FAMILY) == "model_family"


def test_type_enum_members():
    """Type enum has expected artifact group types."""
    assert (
        set(member.value for member in tags.Type) ==
        {"third_party", "model", "template", "data", "misc"}
    )

def test_role_enum_members():
    """Role enum members have correct artifact roles."""
    assert tags.Role.LANDMARK_DETECTOR.value == "landmark_detector"
    assert tags.Role.ALIGNMENT_TEMPLATE.value == "alignment_template"
    assert tags.Role.LANDMARK_MANIFEST.value == "landmark_manifest"
    assert tags.Role.PREVIEW.value == "functionality_preview"


def test_stage_enum_members():
    """Stage enum has expected artifact group types."""
    assert (
        set(member.value for member in tags.Stage) ==
        {"development", "staging", "production"}
    )

def test_enum_values_uniqueness():
    """All values within each enum are unique."""
    for enum_cls in [tags.TagKeys, tags.Type, tags.Role, tags.Stage]:
        values = [member.value for member in enum_cls]
        assert len(values) == len(set(values)), (
            f"Duplicates found in {enum_cls.__name__}"
        )


def test_enum_str_behaviour():
    """Enum members properly convert to str."""
    for enum_cls in [tags.TagKeys, tags.Type, tags.Role, tags.Stage]:
        for member in enum_cls:
            assert str(member) == member.value
