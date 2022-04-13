import os
import tqdm
import argparse
from copy import deepcopy
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from data.dataset import HDMapNetSemanticDataset


def redundant_filter(mask, kernel=25):
    M, N = mask.shape
    for i in range(M):
        for j in range(N):
            if mask[i, j] != 0:
                var = deepcopy(mask[i, j])
                local_mask = mask[
                             max(0, i - kernel // 2):min(M, i + kernel // 2 + 1),
                             max(0, j - kernel // 2):min(N, j + kernel // 2 + 1)]
                local_mask[local_mask == mask[i, j]] = 0
                mask[i, j] = var
    return mask


def vis_label(dataroot, version, xbound, ybound, thickness, angle_class):
    data_conf = {
        'image_size': (900, 1600),
        'xbound': xbound,
        'ybound': ybound,
        'thickness': thickness,
        'angle_class': angle_class,
    }

    color_map = np.random.randint(0, 256, (256, 3))
    color_map[0] = np.array([0, 0, 0])

    dataset = HDMapNetSemanticDataset(version=version, dataroot=dataroot, data_conf=data_conf, is_train=False)
    gt_path = os.path.join(dataroot, 'samples', 'semanticGT')

    if not os.path.exists(gt_path):
        os.mkdir(gt_path)

    for idx in tqdm.tqdm(range(dataset.__len__())):
        rec = dataset.nusc.sample[idx]
        semantic_mask, instance_mask, forward_mask, backward_mask, _ = dataset.get_semantic_map(rec)

        lidar_top_path = dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])

        base_path = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0]
        base_path = os.path.join(gt_path, base_path)
        semantic_path = os.path.join(base_path, "SEMANTIC.png")
        instance_path = os.path.join(base_path, "INSTANCE.png")
        direction_path = os.path.join(base_path, "DIRECTION.png")

        if not os.path.exists(base_path):
            os.mkdir(base_path)

        semantic_mask = semantic_mask.astype('uint8') * 255
        semantic_mask = np.moveaxis(semantic_mask, 0, -1)
        Image.fromarray(semantic_mask).save(semantic_path)

        instance_mask = instance_mask.sum(0).astype('uint8')
        instance_color_mask = color_map[instance_mask].astype('uint8')
        Image.fromarray(instance_color_mask).save(instance_path)

        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        plt.xlim(0, 400)
        plt.ylim(200, 0)
        R = 1
        arr_width = 1

        forward_mask = redundant_filter(forward_mask)
        coords = np.where(forward_mask != 0)
        coords = np.stack([coords[1], coords[0]], -1)
        for coord in coords:
            x = coord[0]
            y = coord[1]
            angle = np.deg2rad((forward_mask[y, x] - 1) * 10)
            dx = R * np.cos(angle)
            dy = R * np.sin(angle)
            plt.arrow(x=x+2, y=y+2, dx=dx, dy=dy, width=arr_width, head_width=3 * arr_width, head_length=5 * arr_width, facecolor=(1, 0, 0, 0.5))

        backward_mask = redundant_filter(backward_mask)
        coords = np.where(backward_mask != 0)
        coords = np.stack([coords[1], coords[0]], -1)
        for coord in coords:
            x = coord[0]
            y = coord[1]
            angle = np.deg2rad((backward_mask[y, x] - 1) * 10)
            dx = R * np.cos(angle)
            dy = R * np.sin(angle)
            plt.arrow(x=x-2, y=y-2, dx=dx, dy=dy, width=arr_width, head_width=3 * arr_width, head_length=5 * arr_width, facecolor=(0, 0, 1, 0.5))

        plt.savefig(direction_path)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local HD Map Demo.')
    parser.add_argument('dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--angle_class", type=int, default=36)
    args = parser.parse_args()

    vis_label(args.dataroot, args.version, args.xbound, args.ybound, args.thickness, args.angle_class)
