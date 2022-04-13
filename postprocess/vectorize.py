import numpy as np
import torch
import torch.nn as nn

from .cluster import LaneNetPostProcessor
from .connect import sort_points_by_dist, connect_by_direction


def onehot_encoding(logits, dim=0):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


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


def vectorize(segmentation, embedding, direction, angle_class):
    segmentation = segmentation.softmax(0)
    embedding = embedding.cpu()
    direction = direction.permute(1, 2, 0).cpu()
    direction = get_pred_top2_direction(direction, dim=-1)

    max_pool_1 = nn.MaxPool2d((1, 5), padding=(0, 2), stride=1)
    avg_pool_1 = nn.AvgPool2d((9, 5), padding=(4, 2), stride=1)
    max_pool_2 = nn.MaxPool2d((5, 1), padding=(2, 0), stride=1)
    avg_pool_2 = nn.AvgPool2d((5, 9), padding=(2, 4), stride=1)
    post_processor = LaneNetPostProcessor(dbscan_eps=1.5, postprocess_min_samples=50)

    oh_pred = onehot_encoding(segmentation).cpu().numpy()
    confidences = []
    line_types = []
    simplified_coords = []
    for i in range(1, oh_pred.shape[0]):
        single_mask = oh_pred[i].astype('uint8')
        single_embedding = embedding.permute(1, 2, 0)

        single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask, single_embedding)
        if single_class_inst_mask is None:
            continue

        num_inst = len(single_class_inst_coords)

        prob = segmentation[i]
        prob[single_class_inst_mask == 0] = 0
        nms_mask_1 = ((max_pool_1(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_1 = avg_pool_1(prob.unsqueeze(0))[0].cpu().numpy()
        nms_mask_2 = ((max_pool_2(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_2 = avg_pool_2(prob.unsqueeze(0))[0].cpu().numpy()
        vertical_mask = avg_mask_1 > avg_mask_2
        horizontal_mask = ~vertical_mask
        nms_mask = (vertical_mask & nms_mask_1) | (horizontal_mask & nms_mask_2)

        for j in range(1, num_inst + 1):
            full_idx = np.where((single_class_inst_mask == j))
            full_lane_coord = np.vstack((full_idx[1], full_idx[0])).transpose()
            confidence = prob[single_class_inst_mask == j].mean().item()

            idx = np.where(nms_mask & (single_class_inst_mask == j))
            if len(idx[0]) == 0:
                continue
            lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

            range_0 = np.max(full_lane_coord[:, 0]) - np.min(full_lane_coord[:, 0])
            range_1 = np.max(full_lane_coord[:, 1]) - np.min(full_lane_coord[:, 1])
            if range_0 > range_1:
                lane_coordinate = sorted(lane_coordinate, key=lambda x: x[0])
            else:
                lane_coordinate = sorted(lane_coordinate, key=lambda x: x[1])

            lane_coordinate = np.stack(lane_coordinate)
            lane_coordinate = sort_points_by_dist(lane_coordinate)
            lane_coordinate = lane_coordinate.astype('int32')
            lane_coordinate = connect_by_direction(lane_coordinate, direction, step=7, per_deg=360 / angle_class)

            simplified_coords.append(lane_coordinate)
            confidences.append(confidence)
            line_types.append(i-1)

    return simplified_coords, confidences, line_types
