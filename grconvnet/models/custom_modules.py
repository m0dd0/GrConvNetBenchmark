"""_summary_"""

from abc import ABC
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


# NOTE: dimensions of type annotations assume the parametrization used in the paper


class GraspModel(nn.Module, ABC):
    """An abstract model for grasp network in a common format."""

    def compute_loss(
        self, xc: TensorType["n_samples", 4, 224, 224], yc: TensorType["n_samples"]
    ) -> Dict[str, float]:
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            "loss": p_loss + cos_loss + sin_loss + width_loss,
            "losses": {
                "p_loss": p_loss,
                "cos_loss": cos_loss,
                "sin_loss": sin_loss,
                "width_loss": width_loss,
            },
            "pred": {
                "pos": pos_pred,
                "cos": cos_pred,
                "sin": sin_pred,
                "width": width_pred,
            },
        }

    # def predict(self, xc):
    #     pos_pred, cos_pred, sin_pred, width_pred = self(xc)
    #     return {
    #         'pos': pos_pred,
    #         'cos': cos_pred,
    #         'sin': sin_pred,
    #         'width': width_pred
    #     }


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
