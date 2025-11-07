# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import functools
import io
import json
import os

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple

import numpy as np
import tifffile
import torch
import webdataset

from cubifyanything.boxes import BoxDOF, GeneralInstance3DBoxes
from cubifyanything.capture_stream import get_camera_to_gravity_transform
from cubifyanything.instances import Instances3D
from cubifyanything.measurement import DepthMeasurementInfo, ImageMeasurementInfo
from cubifyanything.orientation import ImageOrientation
from cubifyanything.sensor import PosedSensorInfo, SensorArrayInfo, SensorInfo
from PIL import Image
from read import Loader

from surreal.fov3d.utils.sst_loader import SSTLoader

from webdataset.cache import cached_url_opener
from webdataset.handlers import reraise_exception


def custom_pipe_cleaner(spec):
    # This should only be called when using links directly to the MLR CDN, so assume some stuff.
    return "/".join(Path(spec).parts[-2:])


custom_cached_url_opener = functools.partial(
    cached_url_opener, cache_dir="data", url_to_name=custom_pipe_cleaner
)

PREFIX_SEPARATOR = "."
WORLD_PREFIX = "world"


def split_into_prefix_suffix(name):
    return name.split(PREFIX_SEPARATOR)[:2]


# All samples should be stored with keys like [video_id]/[integer timestamp].[sensor_name]/[measurement_name] (or world/).
def group_by_video_and_timestamp(
    data: Iterable[Dict[str, Any]],
    keys: Callable[[str], Tuple[str, str]] = split_into_prefix_suffix,
    lcase: bool = True,
    suffixes: Optional[Set[str]] = None,
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[Dict[str, Any]]:
    return webdataset.tariterators.group_by_keys(data, keys, lcase, suffixes, handler)


TIME_SCALE = 1e9
MM_TO_M = 1000.0


# Parsers.
def parse_json(data, key):
    return json.loads(data[key].decode("utf-8"))


def parse_size(data):
    return tuple(int(x) for x in data.decode("utf-8").strip("[]").split(", "))


def parse_transform_3x3(data):
    return torch.tensor(np.array(json.loads(data)).reshape(3, 3).astype(np.float32))


def parse_transform_4x4(data):
    return torch.tensor(np.array(json.loads(data)).reshape(4, 4).astype(np.float32))


def read_image_bytes(image_bytes, expected_size, channels_first=True):
    if image_bytes.startswith(b"\x89PNG"):
        # PNG.
        image = np.array(Image.open(io.BytesIO(image_bytes)))
    elif image_bytes.startswith(b"II*\x00") or image_bytes.startswith(b"MM\x00*"):
        # TIFF.
        image = tifffile.imread(io.BytesIO(image_bytes))
    else:
        raise ValueError("Unknown image format")

    assert (image.shape[1], image.shape[0]) == expected_size

    if channels_first and (image.ndim > 2):
        image = np.moveaxis(image, -1, 0)

    return torch.tensor(image)


def read_instances(data):
    instances_data = json.loads(data)
    instances = Instances3D()

    if len(instances_data) == 0:
        # Empty.
        instances.set("gt_ids", [])
        instances.set("gt_names", [])
        instances.set("gt_boxes_3d", empty_box(box_type))
        for src_key_2d, dst_key_2d in [
            ("box_2d_rend", "gt_boxes_2d_trunc"),
            ("box_2d_proj", "gt_boxes_2d_proj"),
        ]:
            instances.set(dst_key_2d, np.empty((0, 4)))

        return instances

    instances.set("gt_ids", [bi["id"] for bi in instances_data])
    instances.set("gt_names", [bi["category"] for bi in instances_data])
    instances.set(
        "gt_boxes_3d",
        GeneralInstance3DBoxes(
            np.concatenate(
                (
                    np.array([bi["position"] for bi in instances_data]),
                    np.array([bi["scale"] for bi in instances_data]),
                ),
                axis=-1,
            ),
            np.array([bi["R"] for bi in instances_data]),
        ),
    )

    return instances


class CubifyAnythingDataset(webdataset.DataPipeline):
    def __init__(
        self,
        url,
        box_dof=BoxDOF.GravityAligned,
        yield_world_instances=False,
        load_arkit_depth=True,
        use_cache=False,
    ):
        self._url = url
        self._yield_world_instances = yield_world_instances
        self._use_cache = use_cache

        super(CubifyAnythingDataset, self).__init__(
            webdataset.SimpleShardList(url),
            (
                custom_cached_url_opener
                if self._use_cache
                else webdataset.tariterators.url_opener
            ),
            webdataset.tariterators.tar_file_expander,
            group_by_video_and_timestamp,
            self._map_samples,
        )

        self.load_arkit_depth = load_arkit_depth

        self.first_timestamp = None

        # root_dir = os.path.expanduser(f"~/boxy_data/tutorial_office")
        root_dir = os.path.expanduser(
            f"~/boxy_data/nym_loc10_newbasemap_463617026552443"
        )
        # self.loader = SSTLoader(
        #    root_dir,
        #    camera="slaml",
        #    with_calib=True,
        #    with_traj=True,
        #    with_sdp=False,
        # )
        self.loader2 = Loader(root_dir, camera="slaml")  # slaml or rgb

    def _map_sample(self, sample):
        video_id, timestamp = sample["__key__"].split("/")
        video_id = int(video_id)
        if timestamp == "world":
            return dict(
                world=dict(instances=read_instances(sample["gt/instances"])),
                meta=dict(video_id=video_id),
            )

        # gt_depth_size = parse_size(sample["_gt/depth/size"])

        timestamp = float(timestamp) / 1e9

        # At this point, everything is in camera coordinates.
        wide = PosedSensorInfo()
        wide.RT = torch.eye(4)[None]

        # wide.image = ImageMeasurementInfo(
        #    size=parse_size(sample["_wide/image/size"]),
        #    K=parse_transform_3x3(sample["wide/image/k"])[None],
        # )
        # root_dir = os.path.expanduser(f"~/boxy_data/aeo_seq21_188862233995371")

        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        idx = int((timestamp - self.first_timestamp) * 10.0)
        idx += 30  # skip first 3 seconds

        img_torch, cam, T_wr = self.loader2.load_one(idx)
        print("USING OLD LOADER")

        # out = self.loader.load_one(idx, pinhole=True, unrotate=True)
        # img_torch = (out["img"] * 255).byte()
        # cam = out["cam"]
        # T_wr = out["T_world_rig"]
        # print("USING SST LOADER")

        # print(T_wr.t)
        # print(T_wc.t)
        # exit(1)

        HH = img_torch.shape[2]
        WW = img_torch.shape[3]
        img_size = (HH, WW)
        K = torch.eye(3)
        K[0, 0] = cam.f[0]
        K[1, 1] = cam.f[1]
        K[0, 2] = cam.c[0]
        K[1, 2] = cam.c[1]
        K = K[None]

        T_wc = T_wr @ cam.T_camera_rig.inverse()
        print("T_world_rig", T_wr.t)
        print("T_cam_rig", cam.T_camera_rig.t)
        print("f", K)
        print(f"height width {HH} {WW}")

        # wide.image = ImageMeasurementInfo(
        #    size=parse_size(sample["_wide/image/size"]),
        #    K=parse_transform_3x3(sample["wide/image/k"])[None],
        # )
        wide.image = ImageMeasurementInfo(size=img_size, K=K)

        # if self.load_arkit_depth:
        #    wide.depth = DepthMeasurementInfo(
        #        size=parse_size(sample["_wide/depth/size"]),
        #        K=parse_transform_3x3(sample["wide/depth/k"])[None])
        gt = PosedSensorInfo()
        gt.RT = parse_transform_4x4(sample["gt/rt"])[None]

        # torch.set_printoptions(precision=4, sci_mode=False)
        ## TODO(dd): set this! need to load trajectory.
        # wide.T_gravity = parse_transform_3x3(sample["wide/t_gravity"])[None]
        # print(wide.T_gravity)
        # T_gravity2 = get_camera_to_gravity_transform(gt.RT, ImageOrientation.UPRIGHT, target=ImageOrientation.UPRIGHT)
        # print(T_gravity2)
        T_gravity3 = get_camera_to_gravity_transform(
            T_wc.matrix, ImageOrientation.UPRIGHT, target=ImageOrientation.UPRIGHT
        )
        # print(T_gravity3)
        wide.T_gravity = T_gravity3[None]

        sensor_info = SensorArrayInfo()
        sensor_info.wide = wide
        # sensor_info.gt = gt

        # img_torch = img_torch[None]

        result = dict(
            sensor_info=sensor_info,
            wide=dict(
                image=img_torch,
            ),
            # instances=read_instances(sample["wide/instances"])),
            # gt=dict(
            #    # NOTE: 0.0 values here correspond to failed registration areas.
            #    depth=read_image_bytes(sample["gt/depth"], expected_size=gt.depth.size)[None].float() / MM_TO_M),
            meta=dict(video_id=video_id, timestamp=timestamp),
        )
        # result = dict(
        #    sensor_info=sensor_info,
        #    wide=dict(
        #        image=read_image_bytes(sample["wide/image"], expected_size=wide.image.size)[None],
        #        instances=read_instances(sample["wide/instances"])),
        #    #gt=dict(
        #    #    # NOTE: 0.0 values here correspond to failed registration areas.
        #    #    depth=read_image_bytes(sample["gt/depth"], expected_size=gt.depth.size)[None].float() / MM_TO_M),
        #    meta=dict(video_id=video_id, timestamp=timestamp))

        # if self.load_arkit_depth:
        #    result["wide"]["depth"] = read_image_bytes(sample["wide/depth"], expected_size=wide.depth.size)[None].float() / MM_TO_M

        return result

    def _map_samples(self, samples):
        for sample in samples:
            # Don't map the world instances unless requested to (since these are timeless).
            if sample["__key__"].endswith("/world"):
                if not self._yield_world_instances:
                    continue

            yield self._map_sample(sample)


class AriaCubifyLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # root_dir = os.path.expanduser(f"~/boxy_data/tutorial_office")

        # self.loader = SSTLoader(
        #    root_dir,
        #    camera="slaml",
        #    with_calib=True,
        #    with_traj=True,
        #    with_sdp=False,
        # )
        self.loader2 = Loader(root_dir, camera="slaml")  # slaml or rgb
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.idx
        img_torch, cam, T_wr, timestamp = self.loader2.load_one(idx)
        print("USING OLD LOADER")
        video_id = 0

        # out = self.loader.load_one(idx, pinhole=True, unrotate=True)
        # img_torch = (out["img"] * 255).byte()
        # cam = out["cam"]
        # T_wr = out["T_world_rig"]
        # print("USING SST LOADER

        # print(T_wr.t)
        # print(T_wc.t)
        # exit(1)

        HH = img_torch.shape[2]
        WW = img_torch.shape[3]
        img_size = (HH, WW)
        K = torch.eye(3)
        K[0, 0] = cam.f[0]
        K[1, 1] = cam.f[1]
        K[0, 2] = cam.c[0]
        K[1, 2] = cam.c[1]
        K = K[None]

        T_wc = T_wr @ cam.T_camera_rig.inverse()
        print("T_world_rig", T_wr.t)
        print("T_cam_rig", cam.T_camera_rig.t)
        print("f", K)
        print(f"height width {HH} {WW}")

        # wide.image = ImageMeasurementInfo(
        #    size=parse_size(sample["_wide/image/size"]),
        #    K=parse_transform_3x3(sample["wide/image/k"])[None],
        # )
        # At this point, everything is in camera coordinates.
        wide = PosedSensorInfo()
        wide.RT = torch.eye(4)[None]
        wide.image = ImageMeasurementInfo(size=img_size, K=K)

        # if self.load_arkit_depth:
        #    wide.depth = DepthMeasurementInfo(
        #        size=parse_size(sample["_wide/depth/size"]),
        #        K=parse_transform_3x3(sample["wide/depth/k"])[None])
        # gt = PosedSensorInfo()
        # gt.RT = parse_transform_4x4(sample["gt/rt"])[None]

        # torch.set_printoptions(precision=4, sci_mode=False)
        ## TODO(dd): set this! need to load trajectory.
        # wide.T_gravity = parse_transform_3x3(sample["wide/t_gravity"])[None]
        # print(wide.T_gravity)
        # T_gravity2 = get_camera_to_gravity_transform(gt.RT, ImageOrientation.UPRIGHT, target=ImageOrientation.UPRIGHT)
        # print(T_gravity2)
        T_gravity3 = get_camera_to_gravity_transform(
            T_wc.matrix, ImageOrientation.UPRIGHT, target=ImageOrientation.UPRIGHT
        )
        # print(T_gravity3)
        wide.T_gravity = T_gravity3[None]

        sensor_info = SensorArrayInfo()
        sensor_info.wide = wide
        # sensor_info.gt = gt

        # img_torch = img_torch[None]

        result = dict(
            sensor_info=sensor_info,
            wide=dict(
                image=img_torch,
            ),
            # instances=read_instances(sample["wide/instances"])),
            # gt=dict(
            #    # NOTE: 0.0 values here correspond to failed registration areas.
            #    depth=read_image_bytes(sample["gt/depth"], expected_size=gt.depth.size)[None].float() / MM_TO_M),
            meta=dict(video_id=video_id, timestamp=timestamp),
        )
        self.idx += 1
        return result


if __name__ == "__main__":
    # dataset = CubifyAnythingDataset("file:/tmp/lupine-train-49739919.tar")
    pass
