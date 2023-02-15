"""_summary_
"""

from typing import Dict, Any, Callable, List
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import yaml

from grconvnet.dataloading.datasets import CornellDataset, YCBSimulationData
from grconvnet.datatypes import CameraData
from grconvnet.preprocessing import Preprocessor, PreprocessorBase
from grconvnet.postprocessing import (
    Postprocessor,
    PostprocessorBase,
    Img2WorldCoordConverter,
    Decropper,
)
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
        self.model = model or GenerativeResnet.from_jit()
        self.preprocessor = preprocessor or Preprocessor()
        self.postprocessor = postprocessor or Postprocessor()
        self.img2world_converter = img2world_converter

    def __call__(self, samples: List[CameraData]) -> List[Dict[str, Any]]:
        # batched preprocessing
        input_batch = []
        preprocessor_results = []
        for sample in samples:
            input_batch.append(self.preprocessor(sample))
            preprocessor_results.append(
                deepcopy(self.preprocessor.intermediate_results)
            )
        input_batch = torch.stack(input_batch)

        # batched inference
        with torch.no_grad():
            predictions = self.model(input_batch)

        # batched postprocessing
        grasps_img_batch = []
        posprocessor_results = []
        for output in predictions:
            grasps_img_batch.append(self.postprocessor(output))
            posprocessor_results.append(
                deepcopy(self.postprocessor.intermediate_results)
            )

        # batched img2world conversion
        grasps_world_batch = []
        # img2world_converter_results = [] # TODO support this
        if self.img2world_converter is not None:
            for i_sample in range(len(samples)):
                grasps_img = grasps_img_batch[i_sample]
                sample = samples[i_sample]
                grasps_world = [
                    self.img2world_converter(g_img, sample.depth)
                    for g_img in grasps_img
                ]
                grasps_world_batch.append(grasps_world)

        # batched data collection
        process_data_batch = []
        for i_sample in range(len(samples)):
            process_data = {
                "preprocessor": preprocessor_results[i_sample],
                "postprocessor": posprocessor_results[i_sample],
                # "img2world_converter": img2world_converter_results[i_sample]
                # if self.img2world_converter is not None
                # else None,
                "model_input": input_batch[i_sample],
                "sample": samples[i_sample],
                "grasps_img": grasps_img_batch[i_sample],
                "grasps_world": grasps_world_batch[i_sample]
                if self.img2world_converter is not None
                else None,
            }
            process_data_batch.append(process_data)

        return process_data_batch


def process_dataset(
    dataset,
    e2e_processor: End2EndProcessor,
    exporter: Exporter,
    batch_size=10,
):
    for i_batch in range((len(dataset) // batch_size) + 1):
        j_start = i_batch * batch_size
        j_end = min((i_batch + 1) * batch_size, len(dataset))
        batch = [dataset[j] for j in range(j_start, j_end)]
        print(f"Processing samples {j_start}...{j_end-1}")

        process_data_batch = e2e_processor(batch)

        for process_data in process_data_batch:
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
                if process_data["sample"].cam_intrinsics is not None
                else None,
                cam_intrinsics=process_data["sample"].cam_intrinsics,
                cam_rot=process_data["sample"].cam_rot,
                cam_pos=process_data["sample"].cam_pos,
            )
            plt.close(fig)

            export_data = {
                "original_rgb": process_data["sample"].rgb,
                "rgb_cropped": process_data["preprocessor"]["rgb_cropped"],
                "depth_cropped": process_data["preprocessor"]["depth_cropped"],
                "rgb_masked": process_data["preprocessor"]["rgb_masked"],
                "q_img": process_data["postprocessor"]["q_img"],
                "angle_img": process_data["postprocessor"]["angle_img"],
                "width_img": process_data["postprocessor"]["width_img"],
                "grasps_img": process_data["grasps_img"],
                "grasps_world": process_data["grasps_world"]
                if process_data["sample"].cam_intrinsics is not None
                else None,
                "model_input": process_data["model_input"],
                "overview": fig,
            }

            _ = exporter(export_data, f"{process_data['sample'].name}")


def process_cornell(dataset_path, config_path, export_path, batch_size):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = CornellDataset(dataset_path)

    e2e_processor = module_from_config(config)

    exporter = Exporter(export_dir=export_path)

    export_path.mkdir(parents=True, exist_ok=True)
    with open(export_path / "inference_config.yaml", "w") as f:
        yaml.dump(config, f)

    process_dataset(dataset, e2e_processor, exporter, batch_size)


def process_ycb(dataset_path, config_path, export_path, batch_size):
    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = YCBSimulationData(dataset_path)

    # instantiate e2e processor
    sample = dataset[0]
    e2e_processor = module_from_config(config)
    e2e_processor.img2world_converter.coord_converter = Img2WorldCoordConverter(
        sample.cam_intrinsics, sample.cam_rot, sample.cam_pos
    )
    e2e_processor.img2world_converter.decropper = Decropper(
        resized_in_preprocess=config["preprocessor"]["resize"],
        original_img_size=sample.rgb.shape[1:],
    )

    exporter = Exporter(export_dir=export_path)

    # save config
    export_path.mkdir(parents=True, exist_ok=True)
    with open(export_path / "inference_config.yaml", "w") as f:
        yaml.dump(config, f)

    process_dataset(dataset, e2e_processor, exporter, batch_size)


if __name__ == "__main__":
    for i in range(1, 2):
        process_ycb(
            dataset_path=Path.home() / "Documents" / f"ycb_sim_data_{i}",
            config_path=Path(__file__).parent.parent / "configs" / "ycb_inference.yaml",
            export_path=Path(__file__).parent.parent / "results" / f"ycb_{i}",
            batch_size=100,
        )
