"""_summary_"""

from abc import ABC
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class GraspModel(nn.Module, ABC):
    """An abstract model for grasp network in a common format."""

    @staticmethod
    def check_model_path(model_path: Path) -> Path:
        model_path = Path(model_path)

        if not model_path.exists():
            model_path = Path(__file__).parent.parent / "checkpoints" / model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        return model_path

    @classmethod
    def from_jit(cls, jit_path: Path = None, device: str = None) -> "GraspModel":
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        jit_path = (
            jit_path
            or "cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_15_iou_97.pt"
        )

        jit_path = cls.check_model_path(jit_path)

        # we changed the jitted model class so we only take the state_dict
        # the jitted model expects the default parameters
        # normally we would simply return model=torch.jit.load(jit_path)
        model = cls()
        model.load_state_dict(torch.jit.load(jit_path).state_dict())

        model.to(device)

        return model

    @classmethod
    def from_state_dict_path(
        cls, state_dict_path: Path = None, device: str = None, **kwargs
    ) -> "GenerativeResnet":
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        state_dict_path = cls.check_model_path(state_dict_path)

        model = cls(**kwargs)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))

        model.to(device)

        return model

    @classmethod
    def from_config(cls, **kwargs) -> "GenerativeResnet":
        if "jit_path" in kwargs:
            return cls.from_jit(**kwargs)

        elif "state_dict_path" in kwargs:
            return cls.from_state_dict_path(**kwargs)

        else:
            raise ValueError("No valid config for loading model.")

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
