# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import glob
import itertools
import sys
import uuid

from pathlib import Path

import numpy as np
import rerun
import rerun.blueprint as rrb
import torch
import torchvision

from cubifyanything.batching import Sensors
from cubifyanything.boxes import GeneralInstance3DBoxes

# from cubifyanything.capture_stream import CaptureDataset
from cubifyanything.color import random_color
from cubifyanything.cubify_transformer import make_cubify_transformer
from cubifyanything.dataset import FakeCubifyAnythingDataset
from cubifyanything.instances import Instances3D
from cubifyanything.preprocessor import Augmentor, Preprocessor
from PIL import Image
from scipy.spatial.transform import Rotation

from surreal.fov3d.utils.file_io import ObbCsvWriter2
from surreal.fov3d.utils.obb import ObbTW
from surreal.fov3d.utils.pose import PoseTW, rotation_from_euler


def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    try:
        return src.to(dst)
    except:
        return src.to(dst.device)


def move_to_current_device(x, t):
    if isinstance(x, (list, tuple)):
        return [move_device_like(x_, t) for x_ in x]

    return move_device_like(x, t)


def move_input_to_current_device(batched_input: Sensors, t: torch.Tensor):
    # Assume only two levels of nesting for now.
    return {
        name: {name_: move_to_current_device(m, t) for name_, m in s.items()}
        for name, s in batched_input.items()
    }


# A global dictionary we use to for consistent colors for instances across frames.
ID_TO_COLOR = {}


def log_instances(
    instances,
    prefix,
    boxes_3d_name="gt_boxes_3d",
    ids_name="gt_ids",
    log_instances_name="instances",
    **kwargs,
):
    global ID_TO_COLOR
    boxes_3d = instances.get(boxes_3d_name)

    colors = []
    if instances.has(ids_name):
        ids = instances.get(ids_name)
        for id_ in ids:
            ID_TO_COLOR[id_] = ID_TO_COLOR.get(id_, random_color(rgb=True))
            colors.append(ID_TO_COLOR[id_])
    else:
        ids = None
        colors = [random_color(rgb=True) for _ in range(len(instances))]

    quaternions = [
        rerun.Quaternion(xyzw=Rotation.from_matrix(r).as_quat())
        for r in boxes_3d.R.cpu().numpy()
    ]

    # Hard-code these suffixes.
    rerun.log(
        f"{prefix}/{log_instances_name}",
        rerun.Boxes3D(
            centers=boxes_3d.gravity_center.cpu().numpy(),
            sizes=boxes_3d.dims.cpu().numpy(),
            quaternions=quaternions,
            colors=colors,
            labels=ids,
            show_labels=False,
        ),
        **kwargs,
    )


# def load_data_and_visualize(dataset):
#     blueprint = rrb.Blueprint(
#         rrb.Vertical(
#             contents=[
#                 rrb.Spatial3DView(
#                     name="World",
#                     origin="/world"),
#                 rrb.Horizontal(
#                     contents=[
#                         rrb.Spatial2DView(
#                             name="Image",
#                             origin="/device/wide/image",
#                             contents=[
#                                 "+ $origin/**",
#                                 "+ /device/wide/instances/**"
#                             ]),
#                         rrb.Spatial2DView(
#                             name="Depth",
#                             origin="/device/wide/depth"),
#                         rrb.Spatial2DView(
#                             name="Depth (GT)",
#                             origin="/device/gt/depth"),
#                     ],
#                     name="Wide")
#             ]))

#     recording = None
#     video_id = None
#     for sample in dataset:
#         sample_video_id = sample["meta"]["video_id"]
#         if (recording is None) or (video_id != sample_video_id):
#             new_recording = rerun.new_recording(
#                 application_id=str(sample_video_id), recording_id=uuid.uuid4(), make_default=True)

#             new_recording.send_blueprint(blueprint, make_active=True)
#             rerun.spawn()

#             recording = new_recording
#             video_id = sample_video_id

#         # Check for the world. Note that this may not show if --every-nth-frame is used.
#         if "world" in sample:
#             world_instances = sample["world"]["instances"]
#             log_instances(world_instances, prefix="/world", static=True)
#             continue

#         rerun.set_time_seconds("pts", sample["meta"]["timestamp"], recording=recording)

#         # -> channels last.
#         image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)

#         camera = rerun.Pinhole(
#             image_from_camera=sample["sensor_info"].wide.image.K[-1].numpy(), resolution=sample["sensor_info"].wide.image.size)

#         # Log this to both the device (per-frame) and to the world.
#         rerun.log("/device/wide/image", rerun.Image(image).compress())
#         rerun.log("/device/wide/image", camera)

#         # RT here corresponds to the laser-scanner space, as registered to the capture device, so this allows us
#         # to visualize the camera with respect to the annotation space.
#         RT = sample["sensor_info"].gt.RT[-1].numpy()
#         pose_transform = rerun.Transform3D(
#             translation=RT[:3, 3],
#             rotation=rerun.Quaternion(xyzw=Rotation.from_matrix(RT[:3, :3]).as_quat()))

#         rerun.log("/world/image", pose_transform)
#         rerun.log("/world/image", camera)
#         rerun.log("/world/image/image", rerun.Image(image, opacity=0.5))

#         rerun.log("/device/wide/depth", rerun.DepthImage(sample["wide"]["depth"][-1].numpy()))
#         rerun.log("/device/gt/depth", rerun.DepthImage(sample["gt"]["depth"][-1].numpy()))

#         per_frame_instances = sample["wide"]["instances"]
#         log_instances(per_frame_instances, prefix="/device/wide")

# def get_camera_coords(depth):
#     height, width = depth.shape
#     device = depth.device

#     # camera xy.
#     camera_coords = torch.stack(
#         torch.meshgrid(
#             torch.arange(0, width, device=device),
#             torch.arange(0, height, device=device), indexing="xy"),
#         dim=-1)

#     return camera_coords

# def unproject(depth, K, RT, max_depth=10.0):
#     camera_coords = get_camera_coords(depth) * depth[..., None]

#     intrinsics_4x4 = torch.eye(4, device=depth.device)
#     intrinsics_4x4[:3, :3] = K

#     valid = depth > 0
#     if max_depth is not None:
#         valid &= (depth < max_depth)

#     depth = depth[..., None]
#     uvd = torch.cat((camera_coords, depth, torch.ones_like(depth)), dim=-1)

#     camera_xyz =  torch.linalg.inv(intrinsics_4x4) @ uvd.view(-1, 4).T
#     world_xyz = RT @ camera_xyz

#     return world_xyz.T[..., :-1].reshape(uvd.shape[0], uvd.shape[1], 3), valid


# HERE !!!!
def load_data_and_execute_model(
    model, dataset, augmentor, preprocessor, score_thresh=0.0, viz_on_gt_points=False
):
    is_depth_model = "wide/depth" in augmentor.measurement_keys
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Spatial3DView(
                    name="World",
                    contents=["+ $origin/**", "+ /device/wide/pred_instances/**"],
                    origin="/world",
                ),
                rrb.Horizontal(
                    contents=(
                        [
                            rrb.Spatial2DView(
                                name="Image",
                                origin="/device/wide/image",
                                contents=[
                                    "+ $origin/**",
                                    "+ /device/wide/pred_instances/**",
                                ],
                            )
                        ]
                        + (
                            [
                                # Only show this for RGB-D.
                                rrb.Spatial2DView(
                                    name="Depth", origin="/device/wide/depth"
                                )
                            ]
                            if is_depth_model
                            else []
                        )
                    ),
                    name="Wide",
                ),
            ]
        )
    )

    recording = None
    video_id = None

    root_dir = dataset.root_dir
    out_path = os.path.join(root_dir, "cutr_frame_obbs.csv")
    writer = ObbCsvWriter2(out_path)

    device = model.pixel_mean
    count = 0
    for ii, sample in enumerate(dataset):
        if count > 120:
            break
        if ii % 15 != 0:
            continue

        # sample_video_id = sample["meta"]["video_id"]
        sample_video_id = 0
        if (recording is None) or (video_id != sample_video_id):
            new_recording = rerun.new_recording(
                application_id=str(sample_video_id),
                recording_id=uuid.uuid4(),
                make_default=True,
            )
            new_recording.send_blueprint(blueprint, make_active=True)
            rerun.spawn()

            recording = new_recording
            video_id = sample_video_id

            # Keep things in image space, so adjust accordingly.
            rerun.log("/world", rerun.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        # float in seconds? e.g. 273954.896831666
        timestamp = sample["meta"]["timestamp"]
        # image is 768x1024x3 numpy image, [0, 255], uint8
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)
        # intrinsics K is 3x3 numpy array, float32
        sensor_info = sample["sensor_info"].wide.image.K[-1].numpy()
        # resolution is tuple of (width, height) [1024, 768]
        resolution = sample["sensor_info"].wide.image.size

        rerun.set_time_seconds("pts", timestamp, recording=recording)
        color_camera = rerun.Pinhole(
            image_from_camera=sensor_info, resolution=resolution
        )

        packaged = augmentor.package(sample)

        packaged = move_input_to_current_device(packaged, device)
        packaged = preprocessor.preprocess([packaged])

        with torch.no_grad():
            pred_instances = model(packaged)[0]

        pred_instances = pred_instances[pred_instances.scores >= score_thresh]

        # Hold off on logging anything until now, since the delay might confuse the user in the visualizer.
        rerun.log("/device/wide/image", rerun.Image(image).compress())
        rerun.log("/device/wide/image", color_camera)

        # if is_depth_model:
        #     rerun.log("/device/wide/depth", rerun.DepthImage(sample["wide"]["depth"][-1].numpy()))
        #     rerun.log("/device/wide/depth", depth_camera)

        # if xyzrgb is not None:
        #     rerun.log("/world/xyz", rerun.Points3D(positions=xyzrgb[..., :3], colors=xyzrgb[..., 3:], radii=None))

        log_instances(
            pred_instances,
            prefix="/device/wide",
            boxes_3d_name="pred_boxes_3d",
            ids_name=None,
            log_instances_name="pred_instances",
        )

        T_wc = sample["T_wc"].cpu().float()

        R = pred_instances.pred_boxes_3d.R.clone().cpu().float()

        center = pred_instances.pred_boxes_3d.center.clone().cpu().float()
        whl = pred_instances.pred_boxes_3d.whl.clone().cpu().float()
        T_co = PoseTW.from_Rt(R, center)
        # T_co = PoseTW.from_Rt(R, center).inverse()

        T_wo = T_wc @ T_co
        # T_wo = T_co


        # rotate by 90 degrees around z-axis, not sure why but probably a convention differnece.
        R_w = T_wo.R
        R_nw_w = rotation_from_euler(torch.tensor([0, 0, np.pi / 2]).reshape(1, 3))
        R_nw = R_nw_w @ R_w
        T_wo = PoseTW.from_Rt(R_nw, T_wo.t)



        #T_nw_w = rotation_from_euler(torch.tensor([0, 0, np.pi / 2], device=device))
        #T_nw_w = rotation_from_euler(torch.tensor([np.pi / 2, 0, 0], device=device))
        ## Rotate world by -90 degrees around x-axis.
        #roll = -np.pi / 2
        #R_nw_w = rotation_from_euler(torch.tensor([[roll, 0.0, 0.0]]))[0]
        #T_nw_w = PoseTW.from_Rt(R_nw_w, torch.zeros(3))
        #T_wo = T_nw_w @ T_wo  # rotate by 90 degrees around x-axis

        bb3_sc = whl
        xmin = -bb3_sc[:, 0] / 2
        xmax = bb3_sc[:, 0] / 2
        ymin = -bb3_sc[:, 1] / 2
        ymax = bb3_sc[:, 1] / 2
        zmin = -bb3_sc[:, 2] / 2
        zmax = bb3_sc[:, 2] / 2
        sz = torch.stack([xmin, xmax, ymin, ymax, zmin, zmax], dim=-1)
        N = sz.shape[0]
        inst_ids = torch.arange(N).float()
        sem_ids = 32 + torch.zeros(N)  # 32 is Anything
        all_obbs = ObbTW.from_lmc(
            bb3_object=sz, T_world_object=T_wo, sem_id=sem_ids, inst_id=inst_ids
        )
        sem_id_to_name = {32: "Anything"}
        writer.write(all_obbs, int(timestamp * 1e9), sem_id_to_name)
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "dataset_path",
    #     help='Path to the directory containing the .tar files, the full path to a single tar file (recommended), or a path to a txt file containing HTTP links. Using the value "stream" will attempt to stream from your device using the NeRFCapture app',
    # )
    parser.add_argument("--model-path", help="Path to the model to load")
    parser.add_argument(
        "--no-depth", default=False, action="store_true", help="Skip loading depth."
    )
    parser.add_argument("--score-thresh", default=0.25, help="Threshold for detections")
    parser.add_argument(
        "--every-nth-frame", default=None, type=int, help="Load every `n` frames"
    )
    parser.add_argument(
        "--viz-only",
        default=False,
        action="store_true",
        help="Skip loading a model and only visualize data.",
    )
    parser.add_argument(
        "--viz-on-gt-points",
        default=False,
        action="store_true",
        help="Backproject the GT depth to form a point cloud in order to visualize the predictions",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Which device to push the model to (cpu, mps, cuda)",
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        help="Subset of videos to execute on. By default, all. Ignored if a tar file is explicitly given or in stream mode.",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="nym_loc10_newbasemap_463617026552443",
    )

    args = parser.parse_args()
    print("Command Line Args:", args)

    # dataset_path = args.dataset_path
    # use_cache = False

    # if dataset_path == "stream":
    #     # dataset = CaptureDataset()
    #     pass
    # else:
    #     dataset_files = []

    #     # Allow the user to specify a single tar or a txt file containing an http link per line.
    #     if os.path.isfile(dataset_path):
    #         if dataset_path.endswith(".txt"):
    #             with open(dataset_path, "r") as dataset_file:
    #                 dataset_files = [l.strip() for l in dataset_file.readlines()]

    #             # Cache these files locally to prevent repeated downlods.
    #             use_cache = True
    #         else:
    #             args.video_ids = None
    #             dataset_files = [dataset_path]
    #     else:
    #         # Try to glob all files matching ca1m-*.tar
    #         dataset_files = glob.glob(os.path.join(dataset_path, "ca1m-*.tar"))
    #         if len(dataset_files) == 0:
    #             raise ValueError(
    #                 f"Failed to find any .tar files matching ca1m- prefix at {dataset_path}"
    #             )

    #     if args.video_ids is not None:
    #         dataset_files = [
    #             df
    #             for df in dataset_files
    #             if Path(df).with_suffix("").name.split("-")[-1] in args.video_ids
    #         ]

    #     if len(dataset_files) == 0:
    #         raise ValueError("No data was found")

        # dataset = CubifyAnythingDataset(
        #     [
        #         Path(df).as_uri() if not df.startswith("https://") else df
        #         for df in dataset_files
        #     ],
        #     args.seq,
        #     yield_world_instances=args.viz_only,
        #     load_arkit_depth=not args.no_depth,
        #     use_cache=use_cache,
        # )
    dataset = FakeCubifyAnythingDataset(args.seq)

    # if args.viz_only:
    #     if args.every_nth_frame is not None:
    #         dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)

    #     load_data_and_visualize(dataset)
    #     sys.exit(0)

    assert args.model_path is not None
    checkpoint = torch.load(args.model_path, map_location=args.device or "cpu")["model"]

    # Figure out which model this is based on the weights.

    # Basic detection of the actual ViT backbone being used (for our setup, dimension is 1:1 with which ViT).
    backbone_embedding_dimension = checkpoint[
        "backbone.0.patch_embed.proj.weight"
    ].shape[0]

    # We need to detect RGB or RGB only models so we can disable sending depth.
    is_depth_model = any(
        k.startswith("backbone.0.patch_embed_depth.") for k in checkpoint.keys()
    )

    model = make_cubify_transformer(
        dimension=backbone_embedding_dimension, depth_model=is_depth_model
    ).eval()
    model.load_state_dict(checkpoint)

    # # No need for ARKit depth if running an RGB only model.
    # dataset.load_arkit_depth = is_depth_model
    # if args.every_nth_frame is not None:
    #     dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)

    augmentor = Augmentor(
        ("wide/image", "wide/depth") if is_depth_model else ("wide/image",)
    )
    preprocessor = Preprocessor()

    if args.device is not None:
        model = model.to(args.device)

    load_data_and_execute_model(
        model,
        dataset,
        augmentor,
        preprocessor,
        score_thresh=args.score_thresh,
        viz_on_gt_points=args.viz_on_gt_points,
    )
