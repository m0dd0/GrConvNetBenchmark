from abc import ABC, abstractmethod
from typing import Any, Dict
from copy import deepcopy
import importlib

from torchtyping import TensorType

from grconvnet.datatypes import CameraData


class PipelineBase(ABC):
    def __init__(self):
        self.intermediate_results: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any], submodule_source: str) -> "Pipeline":
        config = deepcopy(config)

        cls_kwargs = {}
        for key, value in config.items():
            if isinstance(value, dict) and "class" in value:
                submodule_cls = getattr(
                    importlib.import_module(f"grconvnet.{submodule_source}"),
                    value.pop("class"),
                )
                submodule = submodule_cls(**value)
                cls_kwargs[key] = submodule

            else:
                cls_kwargs[key] = value

        return cls(**cls_kwargs)

    @abstractmethod
    def __call__(self, sample: CameraData) -> TensorType["batch"]:
        raise NotImplementedError()
