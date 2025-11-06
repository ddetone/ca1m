"""
Dataset to stream RGB-D data from the NeRFCapture iOS App -> Cubify Transformer

Adapted from SplaTaM: https://github.com/spla-tam/SplaTAM
"""

import time

# import cyclonedds.idl as idl
# import cyclonedds.idl.annotations as annotate
# import cyclonedds.idl.types as types

from dataclasses import dataclass

import numpy as np
import torch

from cubifyanything.boxes import DepthInstance3DBoxes
from cubifyanything.measurement import DepthMeasurementInfo, ImageMeasurementInfo
from cubifyanything.orientation import ImageOrientation, ROT_Z, rotate_tensor
from cubifyanything.sensor import PosedSensorInfo, SensorArrayInfo, SensorInfo

# from cyclonedds.domain import DomainParticipant, Domain
# from cyclonedds.core import Qos, Policy
# from cyclonedds.sub import DataReader
# from cyclonedds.topic import Topic
# from cyclonedds.util import duration

from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import IterableDataset

# # DDS
# # ==================================================================================================
# @dataclass
# @annotate.final
# @annotate.autoid("sequential")
# class CaptureFrame(idl.IdlStruct, typename="CaptureData.CaptureFrame"):
#     id: types.uint32
#     annotate.key("id")
#     timestamp: types.float64
#     fl_x: types.float32
#     fl_y: types.float32
#     cx: types.float32
#     cy: types.float32
#     transform_matrix: types.array[types.float32, 16]
#     width: types.uint32
#     height: types.uint32
#     image: types.sequence[types.uint8]
#     has_depth: bool
#     depth_width: types.uint32
#     depth_height: types.uint32
#     depth_scale: types.float32
#     depth_image: types.sequence[types.uint8]

# # 8 MB seems to work for me, but not 10 MB.
# dds_config = """<?xml version="1.0" encoding="UTF-8" ?> \
# <CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd"> \
#     <Domain id="any"> \
#         <Internal> \
#             <MinimumSocketReceiveBufferSize>8MB</MinimumSocketReceiveBufferSize> \
#         </Internal> \
#         <Tracing> \
#             <Verbosity>config</Verbosity> \
#             <OutputFile>stdout</OutputFile> \
#         </Tracing> \
#     </Domain> \
# </CycloneDDS> \
# """

T_RW_to_VW = (
    np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    .reshape((4, 4))
    .astype(np.float32)
)

T_RC_to_VC = (
    np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    .reshape((4, 4))
    .astype(np.float32)
)

T_VC_to_RC = (
    np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    .reshape((4, 4))
    .astype(np.float32)
)


def compute_VC2VW_from_RC2RW(T_RC_to_RW):
    T_vc2rw = np.matmul(T_RC_to_RW, T_VC_to_RC)
    T_vc2vw = np.matmul(T_RW_to_VW, T_vc2rw)
    return T_vc2vw


def get_camera_to_gravity_transform(pose, current, target=ImageOrientation.UPRIGHT):
    z_rot_4x4 = torch.eye(4).float()
    z_rot_4x4[:3, :3] = ROT_Z[(current, target)]
    pose = pose @ torch.linalg.inv(z_rot_4x4.to(pose))

    # This is somewhat lazy.
    fake_corners = DepthInstance3DBoxes(
        np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    ).corners[:, [1, 5, 4, 0, 2, 6, 7, 3]]
    fake_corners = torch.cat(
        (fake_corners, torch.ones_like(fake_corners[..., :1])), dim=-1
    ).to(pose)

    fake_corners = (torch.linalg.inv(pose) @ fake_corners.permute(0, 2, 1)).permute(
        0, 2, 1
    )[..., :3]
    fake_basis = torch.stack(
        [
            (fake_corners[:, 1] - fake_corners[:, 0])
            / torch.linalg.norm(fake_corners[:, 1] - fake_corners[:, 0], dim=-1)[
                :, None
            ],
            (fake_corners[:, 3] - fake_corners[:, 0])
            / torch.linalg.norm(fake_corners[:, 3] - fake_corners[:, 0], dim=-1)[
                :, None
            ],
            (fake_corners[:, 4] - fake_corners[:, 0])
            / torch.linalg.norm(fake_corners[:, 4] - fake_corners[:, 0], dim=-1)[
                :, None
            ],
        ],
        dim=1,
    ).permute(0, 2, 1)

    # this gets applied _after_ predictions to put it in camera space.
    T = Rotation.from_euler(
        "xz", Rotation.from_matrix(fake_basis[-1].cpu().numpy()).as_euler("yxz")[1:]
    ).as_matrix()

    return torch.tensor(T).to(pose)


MAX_LONG_SIDE = 1024


# Acts like CubifyAnythingDataset but reads from the NeRFCapture stream.
class CaptureDataset(IterableDataset):
    def __init__(self, load_arkit_depth=True):
        super(CaptureDataset, self).__init__()

        self.load_arkit_depth = load_arkit_depth

        self.domain = Domain(domain_id=0, config=dds_config)
        self.participant = DomainParticipant()
        self.qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1))
        )
        self.topic = Topic(self.participant, "Frames", CaptureFrame, qos=self.qos)
        self.reader = DataReader(self.participant, self.topic)

    def __iter__(self):
        print("Waiting for frames...")
        video_id = 0

        # Start DDS Loop
        while True:
            sample = self.reader.read_next()
            if not sample:
                print("Still waiting...")
                time.sleep(0.05)
                continue

            result = dict(wide=dict())
            wide = PosedSensorInfo()

            # OK, we have a frame. Fill on the requisite data/fields.
            image_info = ImageMeasurementInfo(
                size=(sample.width, sample.height),
                K=torch.tensor(
                    [
                        [sample.fl_x, 0.0, sample.cx],
                        [0.0, sample.fl_y, sample.cy],
                        [0.0, 0.0, 1.0],
                    ]
                )[None],
            )

            print(image_info.size)

            image = np.asarray(sample.image, dtype=np.uint8).reshape(
                (sample.height, sample.width, 3)
            )
            wide.image = image_info
            result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

            if self.load_arkit_depth and not sample.has_depth:
                raise ValueError(
                    "Depth was not found, you likely can only run the RGB only model with your device"
                )

            depth_info = None
            if sample.has_depth:
                # We'll eventually ensure this is 1/4.
                rgb_depth_ratio = sample.width / sample.depth_width
                depth_info = DepthMeasurementInfo(
                    size=(sample.depth_width, sample.depth_height),
                    K=torch.tensor(
                        [
                            [
                                sample.fl_x / rgb_depth_ratio,
                                0.0,
                                sample.cx / rgb_depth_ratio,
                            ],
                            [
                                0.0,
                                sample.fl_y / rgb_depth_ratio,
                                sample.cy / rgb_depth_ratio,
                            ],
                            [0.0, 0.0, 1.0],
                        ]
                    )[None],
                )

                # Is this an encoding thing?
                depth_scale = sample.depth_scale
                print(depth_scale)
                wide.depth = depth_info

                # If I understand this correctly, it looks like this might just want the lower 16 bits?
                depth = torch.tensor(
                    np.asarray(sample.depth_image, dtype=np.uint8)
                    .view(dtype=np.float32)
                    .reshape((sample.depth_height, sample.depth_width))
                )[None].float()
                result["wide"]["depth"] = depth

                desired_image_size = (4 * depth_info.size[0], 4 * depth_info.size[1])
                wide.image = wide.image.resize(desired_image_size)
                result["wide"]["image"] = torch.tensor(
                    np.moveaxis(
                        np.array(Image.fromarray(image).resize(desired_image_size)),
                        -1,
                        0,
                    )
                )[None]
            else:
                # Even for RGB-only, only support a certain long size.
                if max(wide.image.size) > MAX_LONG_SIDE:
                    scale_factor = MAX_LONG_SIDE / max(wide.image.size)

                    new_size = (
                        int(wide.image.size[0] * scale_factor),
                        int(wide.image.size[1] * scale_factor),
                    )
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(
                        np.moveaxis(
                            np.array(Image.fromarray(image).resize(new_size)), -1, 0
                        )
                    )[None]

            # ARKit sends W2C?
            # While we don't necessarily care about pose, we use it to derive the orientation
            # and T_gravity.
            RT = torch.tensor(
                compute_VC2VW_from_RC2RW(
                    np.asarray(sample.transform_matrix)
                    .astype(np.float32)
                    .reshape((4, 4))
                    .T
                )
            )
            wide.RT = RT[None]

            current_orientation = wide.orientation
            target_orientation = ImageOrientation.UPRIGHT

            T_gravity = get_camera_to_gravity_transform(
                wide.RT[-1], current_orientation, target=target_orientation
            )
            wide = wide.orient(current_orientation, target_orientation)

            result["wide"]["image"] = rotate_tensor(
                result["wide"]["image"], current_orientation, target=target_orientation
            )
            if wide.has("depth"):
                result["wide"]["depth"] = rotate_tensor(
                    result["wide"]["depth"],
                    current_orientation,
                    target=target_orientation,
                )

            # No need for pose anymore.
            wide.RT = torch.eye(4)[None]
            wide.T_gravity = T_gravity[None]

            sensor_info = SensorArrayInfo()
            sensor_info.wide = wide

            result["meta"] = dict(video_id=video_id, timestamp=sample.timestamp)
            result["sensor_info"] = sensor_info

            yield result
