from torch.utils.data import DataLoader
import torch
import yaml
from tqdm import tqdm
from matplotlib import pyplot as plt

from grconvnet.models import GenerativeResnet
from grconvnet.postprocessing import Postprocessor, Img2WorldConverter
from grconvnet.utils.config import module_from_config
from grconvnet.utils.misc import get_root_dir
from grconvnet.utils.export import Exporter
from grconvnet.utils import visualization as vis


def process_dataset(
    dataloader: DataLoader,
    model: GenerativeResnet,
    postprocessor: Postprocessor,
    img2world_converter: Img2WorldConverter,
    exporter: Exporter,
    device: str,
):
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        preprocessor_results = list(dataloader.dataset.transform.intermediate_results)[
            -len(batch) :
        ]
        samples = [res["initial_sample"] for res in preprocessor_results]

        with torch.no_grad():
            prediction_batch = model(batch)

        grasps_img_batch = [postprocessor(pred) for pred in prediction_batch]
        postprocessor_results = list(postprocessor.intermediate_results)[-len(batch) :]

        grasps_world_batch = []
        for sample, gs_img in zip(samples, grasps_img_batch):
            grasps_world_batch.append(
                [
                    img2world_converter(
                        g_img,
                        sample.depth,
                        sample.cam_intrinsics,
                        sample.cam_rot,
                        sample.cam_pos,
                    )
                    for g_img in gs_img
                ]
            )

        for sample, pre_result, post_result, gs_img, gs_world in zip(
            samples,
            preprocessor_results,
            postprocessor_results,
            grasps_img_batch,
            grasps_world_batch,
        ):
            fig = vis.overview_fig(
                fig=plt.figure(figsize=(20, 20)),
                original_rgb=vis.make_tensor_displayable(sample.rgb, True, True),
                preprocessed_rgb=vis.make_tensor_displayable(
                    pre_result["rgb_masked"], True, True
                ),
                q_img=vis.make_tensor_displayable(post_result["q_img"], False, False),
                angle_img=vis.make_tensor_displayable(
                    post_result["angle_img"], False, False
                ),
                width_img=vis.make_tensor_displayable(
                    post_result["width_img"], False, False
                ),
                image_grasps=gs_img,
                world_grasps=gs_world,
                cam_intrinsics=sample.cam_intrinsics,
                cam_rot=sample.cam_rot,
                cam_pos=sample.cam_pos,
            )
            plt.close(fig)

            export_data = {
                "grasps_img": gs_img,
                "grasps_world": gs_world,
                "cam_intrinsics": sample.cam_intrinsics,
                "cam_pos": sample.cam_pos,
                "cam_rot": sample.cam_rot,
                "overview": fig,
            }

            _ = exporter(export_data, f"{sample.name}")


if __name__ == "__main__":
    with open(get_root_dir() / "configs" / "ycb_inference.yaml") as f:
        config = yaml.safe_load(f)

    dataloader = module_from_config(config["dataloader"])
    model = module_from_config(config["model"])
    postprocessor = module_from_config(config["postprocessor"])
    img2world_converter = module_from_config(config["img2world_converter"])
    exporter = module_from_config(config["exporter"])

    process_dataset(
        dataloader,
        model,
        postprocessor,
        img2world_converter,
        exporter,
        config["model"]["device"],
    )
