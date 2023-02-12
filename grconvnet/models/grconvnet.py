"""_summary_"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from .custom_modules import GraspModel, ResidualBlock

# NOTE: dimensions of type annotations assume the parametrization used in the paper


class GenerativeResnet(GraspModel):
    @classmethod
    def from_state_dict_path(
        cls, model_path: Path = None, device: str = None
    ) -> "GenerativeResnet":
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        if model_path is None:
            model_path = (
                Path(__file__).parent.parent
                / "checkpoints"
                / "cornell-randsplit-rgbd-grconvnet3-drop1-ch32"
                / "epoch_15_iou_97.pt"
            )

        model = cls()
        model.load_state_dict(torch.jit.load(model_path).state_dict())
        model.to(device)

        return model

    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 1,
        hidden_channels: int = 32,
        dropout: bool = False,
        dropout_prob: float = 0.0,
    ):
        """Base parametrizable class for trying out different hyperparameters.

        Args:
            input_channels (int, optional): Number of input channels of the incoming
                n-channel image. Defaults to 4 for RGBD image.
            output_channels (int, optional): Number of channels for the output of
                sin, cos, quality, width images. Defaults to 1.
            hidden_channels (int, optional): Number of channels after the frst
                convolution. Channel size increases up to hidden_channels*4 in
                the deeper layers. Defaults to 32.
            dropout (bool, optional): Whether to use fropout layers. Defaults to False.
            dropout_prob (float, optional): Probability for the dropout layers.
                Defaults to 0.0.
        """
        # super(GenerativeResnet, self).__init__()
        super().__init__()

        # compress pipeline
        self.conv1 = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=9, stride=1, padding=4
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)

        self.conv3 = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(hidden_channels * 4)

        # residual pipeline
        self.res1 = ResidualBlock(hidden_channels * 4, hidden_channels * 4)
        self.res2 = ResidualBlock(hidden_channels * 4, hidden_channels * 4)
        self.res3 = ResidualBlock(hidden_channels * 4, hidden_channels * 4)
        self.res4 = ResidualBlock(hidden_channels * 4, hidden_channels * 4)
        self.res5 = ResidualBlock(hidden_channels * 4, hidden_channels * 4)
        # self.res_pipeline = nn.Sequential(
        #     self.res1, self.res2, self.res3, self.res4, self.res5
        # )

        # decompress pipeline
        self.conv4 = nn.ConvTranspose2d(
            hidden_channels * 4,
            hidden_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn4 = nn.BatchNorm2d(hidden_channels * 2)

        self.conv5 = nn.ConvTranspose2d(
            hidden_channels * 2,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.bn5 = nn.BatchNorm2d(hidden_channels)

        self.conv6 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels, kernel_size=9, stride=1, padding=4
        )

        # output layers
        self.pos_output = nn.Conv2d(
            in_channels=hidden_channels, out_channels=output_channels, kernel_size=2
        )
        self.cos_output = nn.Conv2d(
            in_channels=hidden_channels, out_channels=output_channels, kernel_size=2
        )
        self.sin_output = nn.Conv2d(
            in_channels=hidden_channels, out_channels=output_channels, kernel_size=2
        )
        self.width_output = nn.Conv2d(
            in_channels=hidden_channels, out_channels=output_channels, kernel_size=2
        )

        # define dropout layers
        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=dropout_prob)
        self.dropout_cos = nn.Dropout(p=dropout_prob)
        self.dropout_sin = nn.Dropout(p=dropout_prob)
        self.dropout_wid = nn.Dropout(p=dropout_prob)

        # initialize all convolutional layers with xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(
        self, x_in: TensorType["n_samples", 4, 224, 224]
    ) -> TensorType["n_samples", 4, 1, 224, 224]:
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        # x = self.res_pipeline(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        network_output = torch.stack(
            (pos_output, cos_output, sin_output, width_output), 1
        )

        return network_output
