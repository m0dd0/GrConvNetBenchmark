"""_summary_
"""

from typing import Dict, Any, Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from grconvnet.dataloading.datasets import CornellDataset, YCBSimulationData
from grconvnet.datatypes import CameraData
from grconvnet.preprocessing import Preprocessor, PreprocessorBase
from grconvnet.postprocessing import Postprocessor, PostprocessorBase
from grconvnet.models import GenerativeResnet
from grconvnet.utils import visualization as vis
from grconvnet.utils.export import Exporter
from grconvnet.utils.config import module_from_config


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
            model = GenerativeResnet.from_jit()
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


def process_dataset(
    dataset,
    e2e_processor: End2EndProcessor,
    exporter: Exporter,
):
    for sample in dataset:
        print(f"Processing sample {sample.name}...")

        process_data = e2e_processor(sample)

        fig = vis.overview_fig(
            fig=plt.figure(figsize=(20, 20)),
            original_rgb=vis.make_tensor_displayable(
                process_data["sample"].rgb, True, True
            ),
            preprocessed_rgb=vis.make_tensor_displayable(
                process_data["preprocessor"]["rgb_masked"], True, True
            ),
            q_img=vis.make_tensor_displayable(
                process_data["postprocessor"]["q_img"], False, False
            ),
            angle_img=vis.make_tensor_displayable(
                process_data["postprocessor"]["angle_img"], False, False
            ),
            width_img=vis.make_tensor_displayable(
                process_data["postprocessor"]["width_img"], False, False
            ),
            image_grasps=process_data["grasps_img"],
            world_grasps=process_data["grasps_world"]
            if sample.cam_intrinsics is not None
            else None,
            cam_intrinsics=sample.cam_intrinsics,
            cam_rot=sample.cam_rot,
            cam_pos=sample.cam_pos,
        )
        plt.close(fig)

        export_data = {
            "rgb_cropped": process_data["preprocessor"]["rgb_cropped"],
            "depth_cropped": process_data["preprocessor"]["depth_cropped"],
            "rgb_masked": process_data["preprocessor"]["rgb_masked"],
            "q_img": process_data["postprocessor"]["q_img"],
            "angle_img": process_data["postprocessor"]["angle_img"],
            "width_img": process_data["postprocessor"]["width_img"],
            "grasps_img": process_data["grasps_img"],
            "grasps_world": process_data["grasps_world"]
            if sample.cam_intrinsics is not None
            else None,
            "model_input": process_data["model_input"],
            "overview": fig,
        }

        _ = exporter(export_data, f"{process_data['sample'].name}")


if __name__ == "__main__":
    # instantiate dataset
    dataset_path = Path("/home/moritz/Documents/cornell")
    dataset = CornellDataset(dataset_path)
    # dataset_path = Path("/home/moritz/Documents/ycb_sim_data_1")
    # dataset = YCBSimulationData(dataset_path)

    # load config
    config_path = (
        Path(__file__).parent.parent / "configs" / "cornell_inference_standard.yaml"
    )
    # config_path = (
    #     Path(__file__).parent.parent / "configs" / "ycb_inference_standard.yaml"
    # )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # instantiate exporter
    # export_path = Path(__file__).parent.parent / "results" / "cornell_standard"
    export_path = Path(__file__).parent.parent / "results" / "ycb_1"
    exporter = Exporter(export_dir=export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    # save config
    with open(export_path / "inference_config.yaml", "w") as f:
        yaml.dump(config, f)

    # instantiate e2e processor
    e2e_processor = module_from_config(config)
    # TODO add img2world converter and decropper for ycb processing

    process_dataset(dataset, e2e_processor, exporter)
