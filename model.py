import torch
import torch.nn as nn

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()

        # Load pre-trained EfficientNetV2 model
        self.backbone = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=True, dropout=0.0, stochastic_depth=0.0)

        # Modify the classifier to your desired number of classes
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.Dropout(0.5),      # Dropout for regularization
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def initialize_model(num_classes):
    model = CustomEfficientNet(num_classes)
    return model



