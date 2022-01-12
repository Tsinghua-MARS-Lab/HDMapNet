import os

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes

from torch.utils.data import Dataset
from data.rasterize import preprocess_map

from .vector_map import VectorizedLocalMap

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


class HDMapNetDataset(Dataset):
    def __init__(self, version, dataroot, xbound=[-30., 30., 0.15], ybound=[-15., 15., 0.15]):
        super(HDMapNetDataset, self).__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)

    def __len__(self):
        return len(self.nusc.sample)

    def get_imgs(self, rec):
        imgs = []
        trans = []
        rots = []
        intrins = []
        for cam in CAMS:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            imgs.append(img)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(torch.Tensor(sens['translation']))
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
            intrins.append(torch.Tensor(sens['camera_intrinsic']))
        return imgs, trans, rots, intrins

    def __getitem__(self, idx):
        rec = self.nusc.sample[idx]
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        imgs, trans, rots, intrins = self.get_imgs(rec)

        return imgs, torch.stack(trans), torch.stack(rots), torch.stack(intrins), vectors


class HDMapNetSemanticDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, xbound=[-30., 30., 0.15], ybound=[-15., 15., 0.15], max_channel=3, thickness=5, angle_class=36):
        super(HDMapNetSemanticDataset, self).__init__(version, dataroot, xbound, ybound)
        self.max_channel = max_channel
        self.thickness = thickness
        self.angle_class = angle_class

    def __getitem__(self, idx):
        rec = self.nusc.sample[idx]
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, self.max_channel, self.thickness, self.angle_class)

        imgs, trans, rots, intrins = self.get_imgs(rec)

        return imgs, torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.Tensor(semantic_masks), \
               torch.Tensor(instance_masks), torch.Tensor(forward_masks), torch.Tensor(backward_masks)
