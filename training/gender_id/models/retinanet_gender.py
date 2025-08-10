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


"""
RetinaNet-based gender classification model.
Uses pretrained RetinaNet backbone with custom gender classification head.
RetinaNet v2 backbone is a stack of ResNet50 + FPN trained on ImageNet1k.
"""


import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2


_RETINANET_FEATURE_SIZE = 256
_HIGH_LEVEL_FEATURE_KEY = '0'


class RetinaNetGenderClassifier(nn.Module):
    """Gender classifier built on RetinaNet backbone.

    Attributes:
        num_classes (int, optional, default=2):
        Number of gender classes (male/female in our case)
        weights (bool, optional, default=True): Whether to use
        pretrained backbone weights or not
        freeze_backbone (bool, optional, default=True): Whether to
        freeze backbone during training

    """

    def __init__(
        self,
        num_classes: int = 2,
        weights: bool = True,
        freeze_backbone: bool = True
    ):

        super().__init__()

        # load pretrained RetinaNet and extract backbone
        self.backbone_model = retinanet_resnet50_fpn_v2(weights=weights)
        self.backbone = self.backbone_model.backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # custom gender classification head
        # high dropout value to regularize powerful backbone
        self.gender_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(_RETINANET_FEATURE_SIZE, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for gender classification.

        Attributes:
            x (torch.Tensor): Input tensor [batch_size, 3, height, width]

        Returns:
            Gender logits [batch_size, num_classes]
        """

        # extract features with backbone
        features = self.backbone(x)

        # gender classification with highest resolution features
        gender_logits = self.gender_head(features[_HIGH_LEVEL_FEATURE_KEY])

        return gender_logits


if __name__ == "__main__":

    net = RetinaNetGenderClassifier(
        num_classes=2, weights=False, freeze_backbone=True)
    net.eval()
    test_input = torch.randn((1, 3, 256, 256))
    with torch.inference_mode():
        output = net(test_input)
    assert output.shape == torch.Size([1, 2])
    print("Smoke check passed.")
