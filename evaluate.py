import torch
import tqdm

from evaluation.dataset import HDMapNetEvalDataset
from evaluation.chamfer_distance import semantic_mask_chamfer_dist_cum
from evaluation.AP import instance_mask_AP
from evaluation.iou import get_batch_iou

SAMPLED_RECALLS = torch.linspace(0.1, 1, 10)
THRESHOLDS = [2, 4, 6]


def get_val_info(args):
    dataset = HDMapNetEvalDataset(args.version, args.dataroot, args.eval_set,
                                  args.result_path, max_channel=args.max_channel, thickness=args.thickness,
                                  xbound=args.xbound, ybound=args.ybound)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bsz, shuffle=False, drop_last=False)

    total_CD1 = torch.zeros(args.max_channel)
    total_CD2 = torch.zeros(args.max_channel)
    total_CD_num1 = torch.zeros(args.max_channel)
    total_CD_num2 = torch.zeros(args.max_channel)
    total_intersect = torch.zeros(args.max_channel)
    total_union = torch.zeros(args.max_channel)
    AP_matrix = torch.zeros((args.max_channel, len(THRESHOLDS)))
    AP_count_matrix = torch.zeros((args.max_channel, len(THRESHOLDS)))

    print('running eval...')
    for pred_map, confidence_level, gt_map in tqdm.tqdm(data_loader):
        # iou
        intersect, union = get_batch_iou(pred_map, gt_map)
        CD1, CD2, num1, num2 = semantic_mask_chamfer_dist_cum(pred_map, gt_map, args.xbound[2], args.ybound[2])

        instance_mask_AP(AP_matrix, AP_count_matrix, pred_map, gt_map, args.xbound[2], args.ybound[2],
                         confidence_level, THRESHOLDS, sampled_recalls=SAMPLED_RECALLS)

        total_intersect += intersect
        total_union += union
        total_CD1 += CD1
        total_CD2 += CD2
        total_CD_num1 += num1
        total_CD_num2 += num2

    return {
        'iou': total_intersect / total_union,
        'CD_pred (precision)': total_CD1 / total_CD_num1,
        'CD_label (recall)': total_CD2 / total_CD_num2,
        'chamfer_distance': (total_CD1 + total_CD2) / (total_CD_num1 + total_CD_num2),
        'Average_precision': AP_matrix / AP_count_matrix,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate nuScenes local HD Map Construction Results.')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--eval_set', type=str, default='mini_val', choices=['train', 'val', 'test', 'mini_train', 'mini_val'])
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--max_channel', type=int, default=3)
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])

    args = parser.parse_args()

    print(get_val_info(args))
