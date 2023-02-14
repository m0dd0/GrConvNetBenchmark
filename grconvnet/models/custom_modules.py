import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class ResidualBlock(nn.Module):
    """A residual block with dropout option"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(
        self, x_in: TensorType["n_samples", 128, 56, 56]
    ) -> TensorType["n_samples", 128, 56, 56]:
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
