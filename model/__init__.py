from .hdmapnet import HDMapNet, HDMapNetMono
from .ipm_net import IPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar

def get_model(method, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    if method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_cam':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)
    elif method == 'HDMapNet_lidar':
        model = PointPillar(data_conf, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    elif method == 'HDMapNet-mono':
        model = HDMapNetMono(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class)
    else:
        raise NotImplementedError

    return model
