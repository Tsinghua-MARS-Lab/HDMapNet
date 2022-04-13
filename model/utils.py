import torch

def plane_grid_2d(xbound, ybound):
    xmin, xmax = xbound[0], xbound[1]
    num_x = int((xbound[1] - xbound[0]) / xbound[2])
    ymin, ymax = ybound[0], ybound[1]
    num_y = int((ybound[1] - ybound[0]) / ybound[2])

    y = torch.linspace(xmin, xmax, num_x).cuda()
    x = torch.linspace(ymin, ymax, num_y).cuda()
    y, x = torch.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    coords = torch.stack([x, y], axis=0)
    return coords


def cam_to_pixel(points, xbound, ybound):
    new_points = torch.zeros_like(points)
    new_points[..., 0] = (points[..., 0] - xbound[0]) / xbound[2]
    new_points[..., 1] = (points[..., 1] - ybound[0]) / ybound[2]
    return new_points


def get_rot_2d(yaw):
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)
    rot = torch.zeros(list(yaw.shape) + [2, 2]).cuda()
    rot[..., 0, 0] = cos_yaw
    rot[..., 0, 1] = sin_yaw
    rot[..., 1, 0] = -sin_yaw
    rot[..., 1, 1] = cos_yaw
    return rot


