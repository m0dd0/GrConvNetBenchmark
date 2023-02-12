"""_summary_
"""

from typing import Dict, Any, Callable

import torch

from grconvnet.datatypes import CameraData
from grconvnet.preprocessing import Preprocessor, PreprocessorBase
from grconvnet.postprocessing import Postprocessor, PostprocessorBase
from grconvnet.models import GenerativeResnet


class End2EndProcessor:
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
