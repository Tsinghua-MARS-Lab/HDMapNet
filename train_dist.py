import os
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse

import torch
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from loss import SimpleLoss, DiscriminativeLoss

from data.dataset import semantic_dataset_ddp
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from model import get_model
from evaluate import onehot_encoding, eval_iou

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)


def train(args):

    # 初始化各进程环境
    init_distributed_mode(args=args)
    rank = args.rank
    device = torch.device(args.device)
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

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

    train_loader, val_loader = semantic_dataset_ddp(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)

    if args.finetune:
        logging.info(f'Load pretrained weight from {args.modelf}...')
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(args.modelf, map_location=device), strict=False)
        for name, param in model.named_parameters():
            if 'bevencode.up' in name or 'bevencode.layer3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        logging.info('Load finish !')

    model.to(device)
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = StepLR(opt, 10, 0.1)
    writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).to(device)

    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).to(device)
    direction_loss_fn = torch.nn.BCELoss(reduction='none').to(device)

    model.train()
    best_iou = 0
    counter = 0
    last_idx = len(train_loader) - 1
    for epoch in range(args.nepochs):
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(train_loader):
            t0 = time()
            opt.zero_grad()

            semantic, embedding, direction = model(imgs.to(device), trans.to(device), rots.to(device), intrins.to(device),
                                                   post_trans.to(device), post_rots.to(device), lidar_data.to(device),
                                                   lidar_mask.to(device), car_trans.to(device), yaw_pitch_roll.to(device))

            semantic_gt = semantic_gt.to(device).float()
            instance_gt = instance_gt.to(device)
            seg_loss = loss_fn(semantic, semantic_gt)
            if args.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.to(device)
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            final_loss = seg_loss * args.scale_seg + var_loss * args.scale_var + dist_loss * args.scale_dist + direction_loss * args.scale_direction
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()
            if rank == 0:
                if counter % 100 == 0:
                    intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                    iou = intersects / (union + 1e-7)
                    logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                                f"Time: {t1-t0:>7.4f}    "
                                f"seg_loss: {seg_loss}, var_loss: {var_loss}, dist_loss: {dist_loss}, reg_loss: {reg_loss}, Loss: {final_loss.item():>7.4f}    "
                                f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

                    write_log(writer, iou, 'train', counter)
                    writer.add_scalar('train/step_time', t1 - t0, counter)
                    writer.add_scalar('train/seg_loss', seg_loss, counter)
                    writer.add_scalar('train/var_loss', var_loss, counter)
                    writer.add_scalar('train/dist_loss', dist_loss, counter)
                    writer.add_scalar('train/reg_loss', reg_loss, counter)
                    writer.add_scalar('train/direction_loss', direction_loss, counter)
                    writer.add_scalar('train/final_loss', final_loss, counter)
                    writer.add_scalar('train/angle_diff', angle_diff, counter)
        if rank == 0:
            iou = eval_iou(model, val_loader)
            logger.info(f"EVAL[{epoch:>2d}]:    "
                        f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

            write_log(writer, iou, 'eval', counter)
            if args.save_best:
                if iou[1:].numpy().sum() > best_iou:
                    best_iou = iou[1:].numpy().sum()
                    model_name = os.path.join(args.logdir, f"model_{epoch}_best.pt")
                    torch.save(model.module.state_dict(), model_name)
                    logger.info(f"{model_name} saved")
            else:
                model_name = os.path.join(args.logdir, f"model_{epoch}.pt")
                torch.save(model.module.state_dict(), model_name)
                logger.info(f"{model_name} saved")
        model.train()
        sched.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs_single_seg')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes_trainval/')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_fusion')
    parser.add_argument("--save_best", type=bool, default=True)

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # finetune config
    parser.add_argument('--finetune', default=False)
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', default=False)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', default=False)
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    train(args)
