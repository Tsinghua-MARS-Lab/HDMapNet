import torch
import math

from data.const import FOV_ANGLES


def is_pts_in_fov(x, y):
    fov, offset = FOV_ANGLES['CAM_FRONT']['fov'], FOV_ANGLES['CAM_FRONT']['offset']
    tan_low, tan_high = math.tan((-fov / 2 + offset) / 180 * math.pi), math.tan((fov / 2 + offset) / 180 * math.pi)
    return (y >= x * tan_low) & (y <= x * tan_high)

def get_fov_mask(xbound, ybound):
    
    patch_h = ybound[1] - ybound[0]
    patch_w = xbound[1]
    canvas_h = int(patch_h / ybound[2])
    canvas_w = int(patch_w / xbound[2])

    fov_mask = []
    for idx_h in range(canvas_h):
        horizontal_mask = []
        for idx_w in range(canvas_w):
            x, y = idx_w, idx_h - canvas_h / 2
            horizontal_mask.append(is_pts_in_fov(x, y))
            pass
        fov_mask.append(horizontal_mask)

    return torch.Tensor(fov_mask).bool()

def get_canvas_w(xbound):
    return int(xbound[1] / xbound[2])

def fov_postprocess(patch, xbound, ybound):
    fov_mask = get_fov_mask(xbound, ybound)

    canvas_w = get_canvas_w(xbound)
    patch_clip = patch[..., canvas_w:]

    rep_shape = [*patch_clip.shape[:-2], 1, 1]
    fov_mask = fov_mask.repeat(rep_shape)
    
    return patch_clip * fov_mask
