from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m
