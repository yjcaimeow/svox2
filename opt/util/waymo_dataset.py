# Standard NeRF Blender dataset loader
from .util import Rays, Intrin, select_or_shuffle_rays
from .dataset_base import DatasetBase
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
from os import path
import imageio
from tqdm import tqdm
import cv2
import json
import numpy as np

# import tensorflow.compat.v1 as tf
import tensorflow as tf
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#tf.enable_eager_execution()
# from waymo_open_dataset.utils import range_image_utils
# from waymo_open_dataset.utils import transform_utils
# from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses, c2w

class WaymoDataset(DatasetBase):
    """
    NeRF dataset loader
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,
        factor: int = 8,
        scale : Optional[float] = None,
        permutation: bool = True,
        white_bkgd: bool = True,
        n_images = None,
        **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 100.0
            #scene_scale = 2/3
        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        all_c2w = []
        all_gt = []

        split_name = split if split != "test_train" else "train"

        dataset=tf.data.TFRecordDataset('/home/xschen/yjcai/segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord', compression_type='')
        all_imgs, all_poses = [], []
        for index, data in enumerate(dataset):
            #if index>=30 :
            #    break
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            ''' image load '''
            front_camera = frame.images[0]
            data = frame.context
            pose_vehicle2world   = np.reshape(np.array(frame.pose.transform, np.float32), (4, 4))
            img = (np.array(tf.image.decode_jpeg(front_camera.image)) / 255.).astype(np.float32)

            if index == 0:
                intrinsic = data.camera_calibrations[0].intrinsic
                pose_camera2vehicle= np.array(data.camera_calibrations[0].extrinsic.transform,dtype=np.float32).reshape(4,4) #camera-vehicle from the sensor frame to the vehicle frame.
                pose_vehicle2camera = np.linalg.inv(pose_camera2vehicle).astype(np.float32)
                focal = intrinsic[0]
                K = np.array([ \
                            [intrinsic[0], 0, intrinsic[2]], \
                            [0, intrinsic[0], intrinsic[3]], \
                            [0, 0, 1]], dtype=np.float32)
                W, H = data.camera_calibrations[0].width, data.camera_calibrations[0].height

                hwf = np.reshape([H, W, focal, 0], [4,1])
                hwf = hwf[None,:]

            undist_img = cv2.undistort(img, K, np.asarray(intrinsic[4:9]), None, K)

            pose_camera2world = pose_vehicle2world @ pose_camera2vehicle
            all_imgs.append(undist_img)
            all_poses.append(pose_camera2world)
        self.split = split
        if self.split == "train":
            self.gt = torch.from_numpy(np.asarray(all_imgs, dtype=np.float32)).cuda()
        imgs = np.asarray(all_imgs, dtype=np.float32)
        poses = np.asarray(all_poses, dtype=np.float32)
        poses = np.concatenate([-poses[:, :, 1:2], poses[:, :, 2:3], -poses[:, :, 0:1], poses[:, :, 3:4]/scene_scale], 2)

        self.n_images, self.h_full, self.w_full, _ = imgs.shape
        #self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        
        hwf = np.repeat(hwf, self.n_images, axis=0)
        poses = np.concatenate([poses, hwf], 2)
        poses, _ = recenter_poses(poses)
        
        self.c2w = torch.from_numpy(poses[:,:,:4]).float().cuda()

        self.intrins_full : Intrin = Intrin(focal, focal,
                                            intrinsic[2],
                                            intrinsic[3])

        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full//factor, self.w_full//factor
            #self.intrins : Intrin = self.intrins_full
            self.intrins : Intrin = Intrin(focal/factor, focal/factor,
                                            intrinsic[2]/factor,
                                            intrinsic[3]/factor)

            imgs_half_res = np.zeros((imgs.shape[0], self.h, self.w, 3))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img,  (self.w, self.h), interpolation=cv2.INTER_AREA)
            self.gt = torch.from_numpy(np.asarray(imgs_half_res, dtype=np.float32)).cuda()
        print (self.split)
        print (self.h, self.w)
        print (self.intrins)
        self.should_use_background = False  # Give warning

