MAP = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
CLASS2LABEL = {
    'road_divider': 0,
    'lane_divider': 0,
    'ped_crossing': 1,
    'contours': 2,
    'others': -1
}
NUM_CLASSES = 3
IMG_ORIGIN_H = 900
IMG_ORIGIN_W = 1600

# FOV angles in degree
FOV_ANGLES = {
    'CAM_FRONT_LEFT': {'fov': 70, 'offset': 55},
    'CAM_FRONT': {'fov': 70, 'offset': 0},
    'CAM_FRONT_RIGHT': {'fov': 70, 'offset': -55},
    'CAM_BACK_LEFT': {'fov': 70, 'offset': 110},
    'CAM_BACK': {'fov': 110, 'offset': 180},
    'CAM_BACK_RIGHT': {'fov': 70, 'offset': -110}
}