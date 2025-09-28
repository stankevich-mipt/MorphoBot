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


"""Dataset configuration for router+UTKFaces workflow."""

from dataclasses import dataclass

@dataclass
class DatasetConfig:  # noqa: D101
    root_male: str
    root_female: str
    train_split: float = 0.8
    val_split: float = 0.2
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
