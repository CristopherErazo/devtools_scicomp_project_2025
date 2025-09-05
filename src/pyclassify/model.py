import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # here insert convolutional blocks
            # First conv
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # Conv1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 2,),
            # Second conv
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),  # Conv2
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Third conv
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            # Fourth conv
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            # Fifth conv
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)


        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # FC1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # FC2
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)  # FC3
        )
    def forward(self, x):
        x = self.avgpool(self.features(x)).flatten(start_dim=1)
        logits = self.classifier(x)
        return logits
