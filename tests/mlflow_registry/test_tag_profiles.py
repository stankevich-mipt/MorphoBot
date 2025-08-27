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


"""Unit tests for mlflow_registry.tag_profiles module.

Tests cover:
    - Validity of tag keys and enum-typed values
    - Uniqueness of profiles
"""

from mlflow_registry import tag_profiles
from mlflow_registry import tags
import pytest


def test_tag_profiles_keys_are_valid_enum_members():
    """All keys in TAG_PROFILES must be valid TagKeys enum member."""
    for profile_name, profile_dict in tag_profiles.TAG_PROFILES.items():
        for key in profile_dict.keys():
            assert key in tags.TagKeys, (
                f"Invalid tag key in profile {profile_name}: {key}"
            )


def test_tag_profiles_values_are_correct_enum_types():
    """All values in TAG_PROFILES must be members of key-matching enums."""
    key_to_enum = {
        tags.TagKeys.TAG_TYPE: tags.Type,
        tags.TagKeys.TAG_ROLE: tags.Role,
        tags.TagKeys.TAG_STAGE: tags.Stage
    }

    for profile_name, profile_dict in tag_profiles.TAG_PROFILES.items():
        for key, value in profile_dict.items():
            expected_enum = key_to_enum.get(key)
            if expected_enum is not None:
                assert isinstance(value, expected_enum), (
                    f"Profile {profile_name} key {key} value "
                    f"must be {expected_enum.__name__} instance"
                )
            else:
                pytest.fail(
                    f"Unexpected key {key} in profile "
                    f"{profile_name}"
                )

def test_tag_profiles_not_empty():
    """TAG_PROFILES must not be empty."""
    assert tag_profiles.TAG_PROFILES, "TAG_PROFILES dictionary is empty"


def test_each_profile_contains_type_keys():
    """Each profile must contain the required TAG_TYPE key."""
    for profile_name, profile_dict in tag_profiles.TAG_PROFILES.items():
        assert tags.TagKeys.TAG_TYPE in profile_dict, (
            f"Profile {profile_name} missing TAG_TYPE key"
        )
