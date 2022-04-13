import torch
from .chamfer_distance import chamfer_distance


def get_line_instances_from_mask(mask, scale_x, scale_y):
    # mask: H, W
    # instance: {1: (N1, 2), 2: (N2, 2), ..., k1: (N_k1, 2)}
    indices = torch.unique(mask)
    instances = {}
    for idx in indices:
        idx = idx.item()
        if idx == 0:
            continue
        pc_x, pc_y = torch.where(mask == idx)
        pc_x = pc_x.float() * scale_x
        pc_y = pc_y.float() * scale_y
        coords = torch.stack([pc_x, pc_y], -1)
        instances[idx] = coords
    return instances


def line_matching_by_CD(inst_pred_lines, inst_pred_confidence, inst_label_lines, threshold):
    # inst_pred_line: a list of points {1: (M1, 2), 2: (M2, 2), ..., k2: (M_k2, 2)}
    # inst_pred_confidence: a list of confidence [c1, c2, ..., ck2]
    # inst_label_line: a list of points {1: (N1, 2), 2: (N2, 2), ..., k1: (N_k1, 2)}
    # return: a list of {'pred': (M, 2), 'label': (N, 2), 'confidence': scalar}
    pred_num = len(inst_pred_lines)
    label_num = len(inst_label_lines)
    CD = torch.zeros((pred_num, label_num)).cuda()

    inst_pred_lines_keys = [*inst_pred_lines]
    inst_label_lines_keys = [*inst_label_lines]
    for i, key_pred in enumerate(inst_pred_lines_keys):
        for j, key_label in enumerate(inst_label_lines_keys):
            CD[i, j] = chamfer_distance(inst_pred_lines[key_pred][None], inst_label_lines[key_label][None], bidirectional=True, threshold=threshold)

    pred_taken = torch.zeros(pred_num, dtype=torch.bool).cuda()
    label_taken = torch.zeros(label_num, dtype=torch.bool).cuda()
    matched_list = []
    if pred_num > 0 and label_num > 0:
        while True:
            idx = torch.argmin(CD)
            i, j = (idx // CD.shape[1]).item(), (idx % CD.shape[1]).item()
            if CD[i, j] >= threshold:
                break
            matched_list.append({
                'pred': inst_pred_lines[inst_pred_lines_keys[i]],
                'confidence': inst_pred_confidence[inst_pred_lines_keys[i]],
                'label': inst_label_lines[inst_label_lines_keys[j]],
                'CD': CD[i, j].item(),
            })
            pred_taken[i] = True
            label_taken[j] = True
            CD[i, :] = threshold
            CD[:, j] = threshold

    for i in range(pred_num):
        if not pred_taken[i]:
            matched_list.append({
                'pred': inst_pred_lines[inst_pred_lines_keys[i]],
                'confidence': inst_pred_confidence[inst_pred_lines_keys[i]],
                'label': None,
                'CD': threshold,
            })

    for j in range(label_num):
        if not label_taken[j]:
            matched_list.append({
                'pred': None,
                'confidence': 0,
                'label': inst_label_lines[inst_label_lines_keys[j]],
                'CD': threshold,
            })

    return matched_list


def instance_mask_AP(AP_matrix, AP_count_matrix, inst_pred_mask, inst_label_mask, scale_x, scale_y, confidence, thresholds, sampled_recalls):
    # inst_pred: N, C, H, W
    # inst_label: N, C, H, W
    # confidence: N, max_instance_num
    N, C, H, W = inst_label_mask.shape
    for n in range(N):
        for c in range(C):
            inst_pred_lines = get_line_instances_from_mask(inst_pred_mask[n, c], scale_x, scale_y)
            inst_label_lines = get_line_instances_from_mask(inst_label_mask[n, c], scale_x, scale_y)
            if len(inst_pred_lines) == 0 and len(inst_label_lines) == 0:
                continue
            AP_matrix[c] += single_instance_line_AP(inst_pred_lines, confidence[n], inst_label_lines, thresholds, sampled_recalls=sampled_recalls)
            AP_count_matrix[c] += 1


def single_instance_line_AP(inst_pred_lines, inst_pred_confidence, inst_label_lines, thresholds, sampled_recalls):
    # inst_pred_line: a list of points {1: (M1, 2), 2: (M2, 2), ..., k2: (M_k2, 2)}
    # inst_pred_confidence: a list of confidence [c1, c2, ..., ck2]
    # inst_label_line: a list of points {1: (N1, 2), 2: (N2, 2), ..., k1: (N_k1, 2)}
    # thresholds: threshold of chamfer distance to identify TP
    num_thres = len(thresholds)
    AP_thres = torch.zeros(num_thres).cuda()
    for t in range(num_thres):
        matching_list = line_matching_by_CD(inst_pred_lines, inst_pred_confidence, inst_label_lines, thresholds[t])
        precision, recall = get_precision_recall_curve_by_confidence(matching_list, len(inst_label_lines), thresholds[t])
        precision, recall = smooth_PR_curve(precision, recall)
        AP = calc_AP_from_precision_recall(precision, recall, sampled_recalls)
        AP_thres[t] = AP
    return AP_thres


def get_precision_recall_curve_by_confidence(matching_list, num_gt, threshold):
    matching_list = sorted(matching_list, key=lambda x: x['confidence'])

    TP = [0]
    FP = [0]
    for match_item in matching_list:
        pred = match_item['pred']
        label = match_item['label']
        dist = match_item['CD']

        if pred is None:
            continue

        if label is None:
            TP.append(TP[-1])
            FP.append(FP[-1] + 1)
            continue

        if dist < threshold:
            TP.append(TP[-1] + 1)
            FP.append(FP[-1])
        else:
            TP.append(TP[-1])
            FP.append(FP[-1] + 1)

    TP = torch.tensor(TP[1:])
    FP = torch.tensor(FP[1:])

    precision = TP / (TP + FP)
    recall = TP / num_gt
    return precision, recall


def smooth_PR_curve(precision, recall):
    idx = torch.argsort(recall)
    recall = recall[idx]
    precision = precision[idx]
    length = len(precision)
    for i in range(length-1, 0, -1):
        precision[:i][precision[:i] < precision[i]] = precision[i]
    return precision, recall


def calc_AP_from_precision_recall(precision, recall, sampled_recalls):
    acc_precision = 0.
    total = len(sampled_recalls)
    for r in sampled_recalls:
        idx = torch.where(recall >= r)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        acc_precision += precision[idx]
    return acc_precision / total
