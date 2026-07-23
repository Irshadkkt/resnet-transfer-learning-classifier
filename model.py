import torch
import torch.nn as nn
from torchvision import models

class Network(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def save_model(self):
        torch.save(self.state_dict(), "model.pkl")
