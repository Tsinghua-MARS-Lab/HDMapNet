import argparse
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import tqdm
import torch

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def vis_segmentation(model, val_loader):
    model.eval()
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(val_loader):

            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            semantic = semantic.softmax(1).cpu().numpy()
            semantic[semantic_gt < 0.1] = np.nan

            for si in range(semantic.shape[0]):
                plt.figure(figsize=(4, 2))
                plt.imshow(semantic[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                plt.imshow(semantic[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                plt.imshow(semantic[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)

                # fig.axes.get_xaxis().set_visible(False)
                # fig.axes.get_yaxis().set_visible(False)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                plt.axis('off')

                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                plt.close()


def vis_vector(model, val_loader, angle_class):
    model.eval()
    car_img = Image.open('icon/car.png')

    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt) in enumerate(val_loader):

            segmentation, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                       post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                       lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())

            for si in range(segmentation.shape[0]):
                coords, _, _ = vectorize(segmentation[si], embedding[si], direction[si], angle_class)

                for coord in coords:
                    plt.plot(coord[:, 0], coord[:, 1], linewidth=5)

                plt.xlim((0, segmentation.shape[3]))
                plt.ylim((0, segmentation.shape[2]))
                plt.imshow(car_img, extent=[segmentation.shape[3]//2-15, segmentation.shape[3]//2+15, segmentation.shape[2]//2-12, segmentation.shape[2]//2+12])

                img_name = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', img_name)
                plt.savefig(img_name)
                plt.close()


def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    vis_vector(model, val_loader, args.angle_class)
    # vis_segmentation(model, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
