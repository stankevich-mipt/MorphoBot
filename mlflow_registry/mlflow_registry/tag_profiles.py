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

"""Script that signifies service-specific tag lookup profiles.

The purpose of the script is to facilitate registry querying
and experiment addition. For that purpose the code
serves as a single source of truth for tag profiles
that uniquely identify artifacts belonging to different services.
"""

from .tags import (
    Role,
    Stage,
    TagKeys,
    TagValues,
    Type,
)

TAG_PROFILES: dict[str, dict[str | TagKeys, str | TagValues]] = {
    "vision_landmarks_detector": {
        TagKeys.TAG_TYPE: Type.THIRD_PARTY,
        TagKeys.TAG_ROLE: Role.LANDMARK_DETECTOR,
        TagKeys.TAG_STAGE: Stage.DEVELOPMENT
    },
    "alignment_manifest": {
        TagKeys.TAG_TYPE: Type.DATA,
        TagKeys.TAG_ROLE: Role.LANDMARK_MANIFEST,
        TagKeys.TAG_STAGE: Stage.DEVELOPMENT
    },
}
