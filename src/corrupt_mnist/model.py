import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.flatten = nn.Flatten()
        self.outlinear = nn.Linear(128, 10)

    
    def forward(self, img) -> torch.tensor:
        img = img.unsqueeze(1)
        self.cnn_features = self.seq(img)
        self.out = self.flatten(self.cnn_features)
        self.out = self.outlinear(self.out)
        return self.out

if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")