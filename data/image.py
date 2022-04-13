import numpy as np
from PIL import Image

import torch
import torchvision

class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))


def img_transform(img, resize, resize_dims):
    post_rot2 = torch.eye(2)
    post_tran2 = torch.zeros(2)

    img = img.resize(resize_dims)

    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2
    post_tran2 = rot_resize @ post_tran2

    post_tran = torch.zeros(3)
    post_rot = torch.eye(3)
    post_tran[:2] = post_tran2
    post_rot[:2, :2] = post_rot2
    return img, post_rot, post_tran


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

# def img_transform(img, resize, resize_dims, crop, flip, rotate):
#     post_rot2 = torch.eye(2)
#     post_tran2 = torch.zeros(2)

#     # adjust image
#     img = img.resize(resize_dims)
#     img = img.crop(crop)
#     if flip:
#         img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
#     img = img.rotate(rotate)

#     # post-homography transformation
#     post_rot2 *= resize
#     post_tran2 -= torch.Tensor(crop[:2])
#     if flip:
#         A = torch.Tensor([[-1, 0], [0, 1]])
#         b = torch.Tensor([crop[2] - crop[0], 0])
#         post_rot2 = A.matmul(post_rot2)
#         post_tran2 = A.matmul(post_tran2) + b
#     A = get_rot(rotate/180*np.pi)
#     b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
#     b = A.matmul(-b) + b
#     post_rot2 = A.matmul(post_rot2)
#     post_tran2 = A.matmul(post_tran2) + b

#     post_tran = torch.zeros(3)
#     post_rot = torch.eye(3)
#     post_tran[:2] = post_tran2
#     post_rot[:2, :2] = post_rot2
#     return img, post_rot, post_tran

