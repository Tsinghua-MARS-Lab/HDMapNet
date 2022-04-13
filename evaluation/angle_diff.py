import torch


def onehot_encoding_spread(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-1, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-2, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+1, max=logits.shape[dim]-1), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+2, max=logits.shape[dim]-1), 1)

    return one_hot


def get_pred_top2_direction(direction, dim=1):
    direction = torch.softmax(direction, dim)
    idx1 = torch.argmax(direction, dim)
    idx1_onehot_spread = onehot_encoding_spread(direction, dim)
    idx1_onehot_spread = idx1_onehot_spread.bool()
    direction[idx1_onehot_spread] = 0
    idx2 = torch.argmax(direction, dim)
    direction = torch.stack([idx1, idx2], dim) - 1
    return direction


def calc_angle_diff(pred_mask, gt_mask, angle_class):
    per_angle = float(360. / angle_class)
    eval_mask = 1 - gt_mask[:, 0]
    pred_direction = get_pred_top2_direction(pred_mask, dim=1).float()
    gt_direction = (torch.topk(gt_mask, 2, dim=1)[1] - 1).float()

    pred_direction *= per_angle
    gt_direction *= per_angle
    pred_direction = pred_direction[:, :, None, :, :].repeat(1, 1, 2, 1, 1)
    gt_direction = gt_direction[:, None, :, :, :].repeat(1, 2, 1, 1, 1)
    diff_mask = torch.abs(pred_direction - gt_direction)
    diff_mask = torch.min(diff_mask, 360 - diff_mask)
    diff_mask = torch.min(diff_mask[:, 0, 0] + diff_mask[:, 1, 1], diff_mask[:, 1, 0] + diff_mask[:, 0, 1]) / 2
    return ((eval_mask * diff_mask).sum() / (eval_mask.sum() + 1e-6)).item()
