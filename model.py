import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()

        # Load pre-trained EfficientNetV2 model
        self.backbone = EfficientNet.from_pretrained('efficientnetv2-b0')

        # Modify the classifier head for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone._fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
def initialize_model(num_classes):
    model = CustomEfficientNet(num_classes)
    return model

