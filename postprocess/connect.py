import math
import random
import numpy as np
from copy import deepcopy

import torch


def sort_points_by_dist(coords):
    coords = coords.astype('float')
    num_points = coords.shape[0]
    diff_matrix = np.repeat(coords[:, None], num_points, 1) - coords
    # x_range = np.max(np.abs(diff_matrix[..., 0]))
    # y_range = np.max(np.abs(diff_matrix[..., 1]))
    # diff_matrix[..., 1] *= x_range / y_range
    dist_matrix = np.sqrt(((diff_matrix) ** 2).sum(-1))
    dist_matrix_full = deepcopy(dist_matrix)
    direction_matrix = diff_matrix / (dist_matrix.reshape(num_points, num_points, 1) + 1e-6)

    sorted_points = [coords[0]]
    sorted_indices = [0]
    dist_matrix[:, 0] = np.inf

    last_direction = np.array([0, 0])
    for i in range(num_points - 1):
        last_idx = sorted_indices[-1]
        dist_metric = dist_matrix[last_idx] - 0 * (last_direction * direction_matrix[last_idx]).sum(-1)
        idx = np.argmin(dist_metric) % num_points
        new_direction = direction_matrix[last_idx, idx]
        if dist_metric[idx] > 3 and min(dist_matrix_full[idx][sorted_indices]) < 5:
            dist_matrix[:, idx] = np.inf
            continue
        if dist_metric[idx] > 10 and i > num_points * 0.9:
            break
        sorted_points.append(coords[idx])
        sorted_indices.append(idx)
        dist_matrix[:, idx] = np.inf
        last_direction = new_direction

    return np.stack(sorted_points, 0)


def connect_by_step(coords, direction_mask, sorted_points, taken_direction, step=5, per_deg=10):
    while True:
        last_point = tuple(np.flip(sorted_points[-1]))
        if not taken_direction[last_point][0]:
            direction = direction_mask[last_point][0]
            taken_direction[last_point][0] = True
        elif not taken_direction[last_point][1]:
            direction = direction_mask[last_point][1]
            taken_direction[last_point][1] = True
        else:
            break

        if direction == -1:
            continue

        deg = per_deg * direction
        vector_to_target = step * np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
        last_point = deepcopy(sorted_points[-1])

        # NMS
        coords = coords[np.linalg.norm(coords - last_point, axis=-1) > step-1]

        if len(coords) == 0:
            break

        target_point = np.array([last_point[0] + vector_to_target[0], last_point[1] + vector_to_target[1]])
        dist_metric = np.linalg.norm(coords - target_point, axis=-1)
        idx = np.argmin(dist_metric)

        if dist_metric[idx] > 50:
           continue

        sorted_points.append(deepcopy(coords[idx]))

        vector_to_next = coords[idx] - last_point
        deg = np.rad2deg(math.atan2(vector_to_next[1], vector_to_next[0]))
        inverse_deg = (180 + deg) % 360
        target_direction = per_deg * direction_mask[tuple(np.flip(sorted_points[-1]))]
        tmp = np.abs(target_direction - inverse_deg)
        tmp = torch.min(tmp, 360 - tmp)
        taken = np.argmin(tmp)
        taken_direction[tuple(np.flip(sorted_points[-1]))][taken] = True


def connect_by_direction(coords, direction_mask, step=5, per_deg=10):
    sorted_points = [deepcopy(coords[random.randint(0, coords.shape[0]-1)])]
    taken_direction = np.zeros_like(direction_mask, dtype=np.bool)

    connect_by_step(coords, direction_mask, sorted_points, taken_direction, step, per_deg)
    sorted_points.reverse()
    connect_by_step(coords, direction_mask, sorted_points, taken_direction, step, per_deg)
    return np.stack(sorted_points, 0)
