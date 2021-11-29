import json
import numpy as np

import torch

from data.dataset import HDMapNetDataset
from data.rasterize import rasterize_map
from nuscenes.utils.splits import create_splits_scenes


class HDMapNetEvalDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, eval_set, result_path, thickness, max_line_count=100, max_channel=3, xbound=[-30., 30., 0.15], ybound=[-15., 15., 0.15]):
        super(HDMapNetEvalDataset, self).__init__(version, dataroot, xbound, ybound)
        scenes = create_splits_scenes()[eval_set]
        with open(result_path, 'r') as f:
            self.prediction = json.load(f)
        self.samples = [samp for samp in self.nusc.sample if self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        self.max_line_count = max_line_count
        self.max_channel = max_channel
        self.thickness = thickness

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        gt_vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])

        gt_map, _ = rasterize_map(gt_vectors, self.patch_size, self.canvas_size, self.max_channel, self.thickness)
        if self.prediction['meta']['vector']:
            pred_vectors = self.prediction['results'][rec['token']]
            pred_map, confidence_level = rasterize_map(pred_vectors, self.patch_size, self.canvas_size, self.max_channel, self.thickness)
        else:
            pred_map = np.array(self.prediction['results'][rec['token']]['map'])
            confidence_level = self.prediction['results'][rec['token']]['confidence_level']

        confidence_level = torch.tensor(confidence_level + [-1] * (self.max_line_count - len(confidence_level)))

        return pred_map, confidence_level, gt_map
