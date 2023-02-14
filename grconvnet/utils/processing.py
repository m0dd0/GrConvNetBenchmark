"""_summary_
"""

from typing import Dict, Any, Callable
from copy import deepcopy
from pathlib import Path
import importlib
import yaml

import torch

from grconvnet.datatypes import CameraData
from grconvnet.preprocessing import Preprocessor, PreprocessorBase
from grconvnet.postprocessing import Postprocessor, PostprocessorBase
from grconvnet.models import GenerativeResnet


def module_from_config(
    config: Dict[str, Any]
):  # -> Union[PreprocessorBase, PostprocessorBase, GraspModel, End2EndProcessor]:
    import_path = config.pop("class").split(".")
    module_cls = getattr(
        importlib.import_module(".".join(import_path)[:-1]), import_path[-1]
    )

    module_args = config["args"]

    return module_cls(**module_args)


class End2EndProcessor:
    @staticmethod
    def check_model_path(model_path: Path) -> Path:
        model_path = Path(model_path)

        if not model_path.exists():
            model_path = Path(__file__).parent.parent / "checkpoints" / model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        return model_path

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "End2EndProcessor":
        config = deepcopy(config)

        # load the model
        if "jit" in config["model"]:
            jit_path = cls.check_model_path(config["model"].pop("jit"))
            device = config["model"].pop("device", None)

            if len(config["model"]) > 0:
                raise ValueError(
                    "If a jit model is used, only jit path and device can be specified. No other model parameters are allowed."
                )

            model = torch.jit.load(jit_path)

            if device is not None:
                model = model.to(device)

        else:
            state_dict_path = cls.check_model_path(
                config["model"].pop("state_dict_path")
            )
            model_cls = getattr(
                importlib.import_module("grconvnet.models"),
                config["models"].pop("class"),
            )
            device = config["model"].pop("device", None)
            model_args = config["model"]
            model = model_cls.from_state_dict_path(
                state_dict_path, device, **model_args
            )

        # load the preprocessor
        preprocessor_cls = getattr(
            importlib.import_module("grconvnet.preprocessing"),
            config["preprocessor"].pop("class"),
        )
        preprocessor = preprocessor_cls.from_config(config["preprocessor"])

        # load the postprocessor
        postprocessor_cls = getattr(
            importlib.import_module("grconvnet.postprocessing"),
            config["postprocessor"].pop("class"),
        )
        postprocessor = postprocessor_cls.from_config(config["postprocessor"])

        # img2world converter
        if "img2world_converter" in config:
            img2world_converter_cls = getattr(
                importlib.import_module("grconvnet.postprocessing"),
                config["img2world_converter"].pop("class"),
            )
            img2world_converter = img2world_converter_cls.from_config(
                config["img2world_converter"]
            )
        else:
            img2world_converter = None

        return cls(
            model=model,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            img2world_converter=img2world_converter,
        )

    @classmethod
    def from_config_path(cls, config_path: Path) -> "End2EndProcessor":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return cls.from_config(config)

    def __init__(
        self,
        model: GenerativeResnet = None,
        preprocessor: PreprocessorBase = None,
        postprocessor: PostprocessorBase = None,
        img2world_converter: Callable = None,
    ):
        # TODO use or syntax for default values
        if model is None:
            model = GenerativeResnet.from_state_dict_path()
        self.model = model

        if preprocessor is None:
            preprocessor = Preprocessor()
        self.preprocessor = preprocessor

        if postprocessor is None:
            postprocessor = Postprocessor()
        self.postprocessor = postprocessor

        self.img2world_converter = img2world_converter

    def __call__(self, sample: CameraData) -> Dict[str, Any]:
        input_tensor = self.preprocessor(sample)
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(input_tensor)

        grasps_img = self.postprocessor(prediction)

        # TODO save all intermediate results of the img2world converter
        grasps_world = []
        if self.img2world_converter is not None:
            grasps_world = [
                self.img2world_converter(g, sample.depth) for g in grasps_img
            ]

        process_data = {
            "preprocessor": self.preprocessor.intermediate_results,
            "postprocessor": self.postprocessor.intermediate_results,
            "img2world_converter": self.img2world_converter.intermediate_results
            if self.img2world_converter is not None
            else None,
            "model_input": input_tensor,
            "sample": sample,
            "grasps_img": grasps_img,
            "grasps_world": grasps_world,
        }

        return process_data


# class PipelineCompose:
#     def __init__(self, pipelines: List[Tuple[str, Callable]]):
#         self.pipelines = pipelines

#         self.intermediate_results: Dict[str, Any] = {}

#     def __call__(self, sample: CameraData) -> Dict[str, Any]:
#         self.intermediate_results = {}

#         for result_name, pipeline in self.pipelines:
#             sample = pipeline(sample)
#             self.intermediate_results[result_name] = sample

#             if hasattr(pipeline, "intermediate_results"):
#                 for res_name, res in pipeline.intermediate_results.items():
#                     if res_name in self.intermediate_results:
#                         raise ValueError(
#                             f"Name {res_name} already in intermediate results"
#                         )
#                     self.intermediate_results[res_name] = res

#         return sample
