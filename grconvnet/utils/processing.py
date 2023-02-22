# TODO rewrite
# def process_dataset(
#     dataset,
#     e2e_processor: End2EndProcessor,
#     exporter: Exporter,
#     batch_size=10,
# ):
#     for i_batch in range((len(dataset) // batch_size) + 1):
#         j_start = i_batch * batch_size
#         j_end = min((i_batch + 1) * batch_size, len(dataset))
#         batch = [dataset[j] for j in range(j_start, j_end)]
#         print(f"Processing samples {j_start}...{j_end-1}")

#         process_data_batch = e2e_processor(batch)

#         for process_data in process_data_batch:
#             fig = vis.overview_fig(
#                 fig=plt.figure(figsize=(20, 20)),
#                 original_rgb=vis.make_tensor_displayable(
#                     process_data["sample"].rgb, True, True
#                 ),
#                 preprocessed_rgb=vis.make_tensor_displayable(
#                     process_data["preprocessor"]["rgb_masked"], True, True
#                 ),
#                 q_img=vis.make_tensor_displayable(
#                     process_data["postprocessor"]["q_img"], False, False
#                 ),
#                 angle_img=vis.make_tensor_displayable(
#                     process_data["postprocessor"]["angle_img"], False, False
#                 ),
#                 width_img=vis.make_tensor_displayable(
#                     process_data["postprocessor"]["width_img"], False, False
#                 ),
#                 image_grasps=process_data["grasps_img"],
#                 world_grasps=process_data["grasps_world"]
#                 if process_data["sample"].cam_intrinsics is not None
#                 else None,
#                 cam_intrinsics=process_data["sample"].cam_intrinsics,
#                 cam_rot=process_data["sample"].cam_rot,
#                 cam_pos=process_data["sample"].cam_pos,
#             )
#             plt.close(fig)

#             export_data = {
#                 "original_rgb": process_data["sample"].rgb,
#                 "rgb_cropped": process_data["preprocessor"]["rgb_cropped"],
#                 "depth_cropped": process_data["preprocessor"]["depth_cropped"],
#                 "rgb_masked": process_data["preprocessor"]["rgb_masked"],
#                 "q_img": process_data["postprocessor"]["q_img"],
#                 "angle_img": process_data["postprocessor"]["angle_img"],
#                 "width_img": process_data["postprocessor"]["width_img"],
#                 "grasps_img": process_data["grasps_img"],
#                 "grasps_world": process_data["grasps_world"]
#                 if process_data["sample"].cam_intrinsics is not None
#                 else None,
#                 "cam_intrinsics": process_data["sample"].cam_intrinsics
#                 if process_data["sample"].cam_intrinsics is not None
#                 else None,
#                 "cam_pos": process_data["sample"].cam_pos
#                 if process_data["sample"].cam_pos is not None
#                 else None,
#                 "cam_rot": process_data["sample"].cam_rot
#                 if process_data["sample"].cam_rot is not None
#                 else None,
#                 "model_input": process_data["model_input"],
#                 "overview": fig,
#             }

#             _ = exporter(export_data, f"{process_data['sample'].name}")


# def process_cornell(dataset_path, config_path, export_path, batch_size):
#     with open(config_path) as f:
#         config = yaml.safe_load(f)

#     dataset = CornellDataset(dataset_path)

#     e2e_processor = module_from_config(config)

#     exporter = Exporter(export_dir=export_path)

#     export_path.mkdir(parents=True, exist_ok=True)
#     with open(export_path / "inference_config.yaml", "w") as f:
#         yaml.dump(config, f)

#     process_dataset(dataset, e2e_processor, exporter, batch_size)


# def process_ycb(dataset_path, config_path, export_path, batch_size):
#     # load config
#     with open(config_path) as f:
#         config = yaml.safe_load(f)

#     dataset = YCBSimulationData(dataset_path)

#     # instantiate e2e processor
#     sample = dataset[0]
#     e2e_processor = module_from_config(config)
#     e2e_processor.img2world_converter.coord_converter = Img2WorldCoordConverter(
#         sample.cam_intrinsics, sample.cam_rot, sample.cam_pos
#     )
#     e2e_processor.img2world_converter.decropper = Decropper(
#         resized_in_preprocess=config["preprocessor"]["resize"],
#         original_img_size=sample.rgb.shape[1:],
#     )

#     exporter = Exporter(export_dir=export_path)

#     # save config
#     export_path.mkdir(parents=True, exist_ok=True)
#     with open(export_path / "inference_config.yaml", "w") as f:
#         yaml.dump(config, f)

#     process_dataset(dataset, e2e_processor, exporter, batch_size)


# if __name__ == "__main__":
#     batch_size = 10
#     config_suffix = "_no_pos_mask"
#     # config_suffix = ""
#     i = 3

#     process_ycb(
#         dataset_path=Path.home() / "Documents" / f"ycb_sim_data_{i}",
#         config_path=Path(__file__).parent.parent
#         / "configs"
#         / f"ycb_inference{config_suffix}.yaml",
#         export_path=Path(__file__).parent.parent
#         / "results"
#         / f"ycb_{i}_b{batch_size}{config_suffix}",
#         batch_size=batch_size,
#     )
