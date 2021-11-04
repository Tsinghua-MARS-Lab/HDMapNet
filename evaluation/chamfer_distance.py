import torch


def chamfer_distance(source_pc, target_pc, threshold, cum=False, bidirectional=True):
    dist = torch.cdist(source_pc.float(), target_pc.float())
    dist1, _ = torch.min(dist, 2)
    dist2, _ = torch.min(dist, 1)
    if cum:
        len1 = dist1.shape[-1]
        len2 = dist2.shape[-1]
        dist1 = dist1.sum(-1)
        dist2 = dist2.sum(-1)
        return dist1, dist2, len1, len2
    dist1 = dist1.mean(-1)
    dist2 = dist2.mean(-1)
    if bidirectional:
        return min((dist1 + dist2) / 2, threshold)
    else:
        return min(dist1, threshold), min(dist2, threshold)


def semantic_mask_chamfer_dist_cum(seg_pred, seg_label, scale_x, scale_y, threshold):
    # seg_label: N, C, H, W
    # seg_pred: N, C, H, W
    N, C, H, W = seg_label.shape

    cum_CD1 = torch.zeros(C, device=seg_label.device)
    cum_CD2 = torch.zeros(C, device=seg_label.device)
    cum_num1 = torch.zeros(C, device=seg_label.device)
    cum_num2 = torch.zeros(C, device=seg_label.device)
    for n in range(N):
        for c in range(C):
            pred_pc_x, pred_pc_y = torch.where(seg_pred[n, c] != 0)
            label_pc_x, label_pc_y = torch.where(seg_label[n, c] != 0)
            pred_pc_x = pred_pc_x.float() * scale_x
            pred_pc_y = pred_pc_y.float() * scale_y
            label_pc_x = label_pc_x.float() * scale_x
            label_pc_y = label_pc_y.float() * scale_y
            if len(pred_pc_x) == 0 and len(label_pc_x) == 0:
                continue

            if len(label_pc_x) == 0:
                cum_CD1[c] += len(pred_pc_x) * threshold
                cum_num1[c] += len(pred_pc_x)
                continue

            if len(pred_pc_x) == 0:
                cum_CD2[c] += len(label_pc_x) * threshold
                cum_num2[c] += len(label_pc_x)
                continue

            pred_pc_coords = torch.stack([pred_pc_x, pred_pc_y], -1).float()
            label_pc_coords = torch.stack([label_pc_x, label_pc_y], -1).float()
            CD1, CD2, len1, len2 = chamfer_distance(pred_pc_coords[None], label_pc_coords[None], threshold=threshold, cum=True)
            cum_CD1[c] += CD1.item()
            cum_CD2[c] += CD2.item()
            cum_num1[c] += len1
            cum_num2[c] += len2
    return cum_CD1, cum_CD2, cum_num1, cum_num2
