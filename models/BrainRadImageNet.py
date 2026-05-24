import os
import torch
import torch.nn as nn
from torchvision.models import resnet50


class BrainRadImageNet(nn.Module):
    def __init__(self, weights_path="models/resnet50_torch.pt", num_classes=3):
        super().__init__()
        self.model = resnet50(weights=None)

        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            print(f"  Loaded RadImageNet weights from {weights_path}")
        else:
            print(f"  WARNING: RadImageNet weights not found at {weights_path}")

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)
