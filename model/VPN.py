import torch
import torch.nn as nn

from .base import CamEncode, BevEncode
from .pointpillar import PointPillarEncoder


class TransformModule(nn.Module):
    def __init__(self, dim, num_view=6):
        super(TransformModule, self).__init__()
        self.num_view = num_view
        self.dim = dim
        self.mat_list = nn.ModuleList()
        for i in range(self.num_view):
            fc_transform = nn.Sequential(
                        nn.Linear(dim * dim, dim * dim),
                        nn.ReLU(),
                        nn.Linear(dim * dim, dim * dim),
                        nn.ReLU()
                    )
            self.mat_list += [fc_transform]

    def forward(self, x):
        # shape x: B, V, C, H, W
        x = x.view(list(x.size()[:3]) + [self.dim * self.dim,])
        view_comb = self.mat_list[0](x[:, 0])
        for index in range(x.size(1))[1:]:
            view_comb += self.mat_list[index](x[:, index])
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.dim, self.dim])
        return view_comb


class VPNModel(nn.Module):
    def __init__(self, outC, camC=64, instance_seg=True, embedded_dim=16, extrinsic=False, lidar=False, xbound=None, ybound=None, zbound=None):
        super(VPNModel, self).__init__()
        self.camC = camC
        self.extrinsic = extrinsic
        self.downsample = 16

        self.camencode = CamEncode(camC)
        self.view_fusion = TransformModule(dim=(8, 22))
        self.up_sampler = nn.Upsample(size=(200, 400), mode='bilinear', align_corners=True)
        self.lidar = lidar
        if lidar:
            self.pp = PointPillarEncoder(128, xbound, ybound, zbound)
            self.bevencode = BevEncode(inC=camC+128, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
        else:
            self.bevencode = BevEncode(inC=camC, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)


    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        # Ks[:, :, :3, :3] = intrins

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def forward(self, points, points_mask, x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll):
        x = self.get_cam_feats(x)
        x = self.view_fusion(x)
        topdown = x.mean(1)
        topdown = self.up_sampler(topdown)
        if self.lidar:
            lidar_feature = self.pp(points, points_mask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        return self.bevencode(topdown)
