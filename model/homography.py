import numpy as np
import torch
import torch.nn as nn
import cv2

CAM_FL = 0
CAM_F = 1
CAM_FR = 2
CAM_BL = 3
CAM_B = 4
CAM_BR = 5


# =========================================================
# Projections
# =========================================================
def rotation_from_euler(rolls, pitchs, yaws, cuda=True):
    """
    Get rotation matrix
    Args:
        roll, pitch, yaw:       In degrees

    Returns:
        R:          [B, 4, 4]
    """
    B = len(rolls)

    si, sj, sk = torch.sin(rolls), torch.sin(pitchs), torch.sin(yaws)
    ci, cj, ck = torch.cos(rolls), torch.cos(pitchs), torch.cos(yaws)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    if cuda:
        R = R.cuda()
    R[:, 0, 0] = cj * ck
    R[:, 0, 1] = sj * sc - cs
    R[:, 0, 2] = sj * cc + ss
    R[:, 1, 0] = cj * sk
    R[:, 1, 1] = sj * ss + cc
    R[:, 1, 2] = sj * cs - sc
    R[:, 2, 0] = -sj
    R[:, 2, 1] = cj * si
    R[:, 2, 2] = cj * ci
    return R


def perspective(cam_coords, proj_mat, h, w, extrinsic, offset=None):
    """
    P = proj_mat @ (x, y, z, 1)
    Project cam2pixel

    Args:
        cam_coords:         [B, 4, npoints]
        proj_mat:           [B, 4, 4]

    Returns:
        pix coords:         [B, h, w, 2]
    """
    eps = 1e-7
    pix_coords = proj_mat @ cam_coords

    N, _, _ = pix_coords.shape

    if extrinsic:
        pix_coords[:, 0] += offset[0] / 2
        pix_coords[:, 2] -= offset[1] / 8
        pix_coords = torch.stack([pix_coords[:, 2], pix_coords[:, 0]], axis=1)
    else:
        pix_coords = pix_coords[:, :2, :] / (pix_coords[:, 2, :][:, None, :] + eps)
    pix_coords = pix_coords.view(N, 2, h, w)
    pix_coords = pix_coords.permute(0, 2, 3, 1).contiguous()
    return pix_coords


def bilinear_sampler(imgs, pix_coords):
    """
    Construct a new image by bilinear sampling from the input image.
    Args:
        imgs:                   [B, H, W, C]
        pix_coords:             [B, h, w, 2]
    :return:
        sampled image           [B, h, w, c]
    """
    B, img_h, img_w, img_c = imgs.shape
    B, pix_h, pix_w, pix_c = pix_coords.shape
    out_shape = (B, pix_h, pix_w, img_c)

    pix_x, pix_y = torch.split(pix_coords, 1, dim=-1)  # [B, pix_h, pix_w, 1]

    # Rounding
    pix_x0 = torch.floor(pix_x)
    pix_x1 = pix_x0 + 1
    pix_y0 = torch.floor(pix_y)
    pix_y1 = pix_y0 + 1

    # Clip within image boundary
    y_max = (img_h - 1)
    x_max = (img_w - 1)

    pix_x0 = torch.clip(pix_x0, 0, x_max)
    pix_y0 = torch.clip(pix_y0, 0, y_max)
    pix_x1 = torch.clip(pix_x1, 0, x_max)
    pix_y1 = torch.clip(pix_y1, 0, y_max)

    # Weights [B, pix_h, pix_w, 1]
    wt_x0 = pix_x1 - pix_x
    wt_x1 = pix_x - pix_x0
    wt_y0 = pix_y1 - pix_y
    wt_y1 = pix_y - pix_y0

    # indices in the image to sample from
    dim = img_w

    # Apply the lower and upper bound pix coord
    base_y0 = pix_y0 * dim
    base_y1 = pix_y1 * dim

    # 4 corner vert ices
    idx00 = (pix_x0 + base_y0).view(B, -1, 1).repeat(1, 1, img_c).long()
    idx01 = (pix_x0 + base_y1).view(B, -1, 1).repeat(1, 1, img_c).long()
    idx10 = (pix_x1 + base_y0).view(B, -1, 1).repeat(1, 1, img_c).long()
    idx11 = (pix_x1 + base_y1).view(B, -1, 1).repeat(1, 1, img_c).long()

    # Gather pixels from image using vertices
    imgs_flat = imgs.reshape([B, -1, img_c])

    im00 = torch.gather(imgs_flat, 1, idx00).reshape(out_shape)
    im01 = torch.gather(imgs_flat, 1, idx01).reshape(out_shape)
    im10 = torch.gather(imgs_flat, 1, idx10).reshape(out_shape)
    im11 = torch.gather(imgs_flat, 1, idx11).reshape(out_shape)

    # Apply weights [pix_h, pix_w, 1]
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    return output


def plane_grid(xbound, ybound, zs, yaws, rolls, pitchs, cuda=True):
    B = len(zs)

    xmin, xmax = xbound[0], xbound[1]
    num_x = int((xbound[1] - xbound[0]) / xbound[2])
    ymin, ymax = ybound[0], ybound[1]
    num_y = int((ybound[1] - ybound[0]) / ybound[2])

    # y = torch.linspace(xmin, xmax, num_x, dtype=torch.double).cuda()
    # x = torch.linspace(ymin, ymax, num_y, dtype=torch.double).cuda()
    y = torch.linspace(xmin, xmax, num_x)
    x = torch.linspace(ymin, ymax, num_y)
    if cuda:
        x = x.cuda()
        y = y.cuda()

    y, x = torch.meshgrid(x, y)

    x = x.flatten()
    y = y.flatten()

    x = x.unsqueeze(0).repeat(B, 1)
    y = y.unsqueeze(0).repeat(B, 1)

    # z = torch.ones_like(x, dtype=torch.double).cuda() * zs.view(-1, 1)
    # d = torch.ones_like(x, dtype=torch.double).cuda()
    z = torch.ones_like(x) * zs.view(-1, 1)
    d = torch.ones_like(x)
    if cuda:
        z = z.cuda()
        d = d.cuda()

    coords = torch.stack([x, y, z, d], axis=1)

    rotation_matrix = rotation_from_euler(pitchs, rolls, yaws, cuda)

    coords = rotation_matrix @ coords
    return coords


def ipm_from_parameters(image, xyz, K, RT, target_h, target_w, extrinsic, post_RT=None):
    """
    :param image: [B, H, W, C]
    :param xyz: [B, 4, npoints]
    :param K: [B, 4, 4]
    :param RT: [B, 4, 4]
    :param target_h: int
    :param target_w: int
    :return: warped_images: [B, target_h, target_w, C]
    """
    P = K @ RT
    if post_RT is not None:
        P = post_RT @ P
    P = P.reshape(-1, 4, 4)
    pixel_coords = perspective(xyz, P, target_h, target_w, extrinsic, image.shape[1:3])
    image2 = bilinear_sampler(image, pixel_coords)
    image2 = image2.type_as(image)
    return image2


class PlaneEstimationModule(nn.Module):
    def __init__(self, N, C):
        super(PlaneEstimationModule, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Linear(N*C, 3)

        self.linear.weight.data.fill_(0.)
        self.linear.bias.data.fill_(0.)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        x = self.max_pool(x)
        x = x.view(B, N*C)
        x = self.linear(x)
        z, pitch, roll = x[:, 0], x[:, 1], x[:, 2]
        return z, pitch, roll


class IPM(nn.Module):
    def __init__(self, xbound, ybound, N, C, z_roll_pitch=False, visual=False, extrinsic=False, cuda=True):
        super(IPM, self).__init__()
        self.visual = visual
        self.z_roll_pitch = z_roll_pitch
        self.xbound = xbound
        self.ybound = ybound
        self.extrinsic = extrinsic
        self.w = int((xbound[1] - xbound[0]) / xbound[2])
        self.h = int((ybound[1] - ybound[0]) / ybound[2])

        if z_roll_pitch:
            self.plane_esti = PlaneEstimationModule(N, C)
        else:
            zs = torch.tensor([0.]).cuda()
            yaws = torch.tensor([0.]).cuda()
            rolls = torch.tensor([0.]).cuda()
            pitchs = torch.tensor([0.]).cuda()
            self.planes = plane_grid(self.xbound, self.ybound, zs, yaws, rolls, pitchs)[0]

        tri_mask = np.zeros((self.h, self.w))
        vertices = np.array([[0, 0], [0, self.h], [self.w, self.h]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.fillPoly(tri_mask, [pts], color=1.)
        self.tri_mask = torch.tensor(tri_mask[None, :, :, None])
        self.flipped_tri_mask = torch.flip(self.tri_mask, [2]).bool()
        if cuda:
            self.tri_mask = self.tri_mask.cuda()
            self.flipped_tri_mask = self.flipped_tri_mask.cuda()
        self.tri_mask = self.tri_mask.bool()

    def mask_warped(self, warped_fv_images):
        warped_fv_images[:, CAM_F, :, :self.w//2, :] *= 0  # CAM_FRONT
        warped_fv_images[:, CAM_FL] *= self.flipped_tri_mask  # CAM_FRONT_LEFT
        warped_fv_images[:, CAM_FR] *= ~self.tri_mask  # CAM_FRONT_RIGHT
        warped_fv_images[:, CAM_B, :, self.w//2:, :] *= 0  # CAM_BACK
        warped_fv_images[:, CAM_BL] *= self.tri_mask  # CAM_BACK_LEFT
        warped_fv_images[:, CAM_BR] *= ~self.flipped_tri_mask  # CAM_BACK_RIGHT
        return warped_fv_images

    def forward(self, images, Ks, RTs, translation, yaw_roll_pitch, post_RTs=None):
        images = images.permute(0, 1, 3, 4, 2).contiguous()
        B, N, H, W, C = images.shape

        if self.z_roll_pitch:
            # z, roll, pitch = self.plane_esti(images)
            zs = translation[:, 2]
            rolls = yaw_roll_pitch[:, 1]
            pitchs = yaw_roll_pitch[:, 2]
            # zs += z
            # rolls += roll
            # pitchs += pitch
            planes = plane_grid(self.xbound, self.ybound, zs, torch.zeros_like(rolls), rolls, pitchs)
            planes = planes.repeat(N, 1, 1)
        else:
            planes = self.planes

        images = images.reshape(B*N, H, W, C)
        warped_fv_images = ipm_from_parameters(images, planes, Ks, RTs, self.h, self.w, self.extrinsic, post_RTs)
        warped_fv_images = warped_fv_images.reshape((B, N, self.h, self.w, C))
        if self.visual:
            warped_fv_images = self.mask_warped(warped_fv_images)

        if self.visual:
            warped_topdown = warped_fv_images[:, CAM_F] + warped_fv_images[:, CAM_B]  # CAM_FRONT + CAM_BACK
            warped_mask = warped_topdown == 0
            warped_topdown[warped_mask] = warped_fv_images[:, CAM_FL][warped_mask] + warped_fv_images[:, CAM_FR][warped_mask]
            warped_mask = warped_topdown == 0
            warped_topdown[warped_mask] = warped_fv_images[:, CAM_BL][warped_mask] + warped_fv_images[:, CAM_BR][warped_mask]
            return warped_topdown.permute(0, 3, 1, 2).contiguous()
        else:
            warped_topdown, _ = warped_fv_images.max(1)
            warped_topdown = warped_topdown.permute(0, 3, 1, 2).contiguous()
            warped_topdown = warped_topdown.view(B, C, self.h, self.w)
            return warped_topdown


