from .base import DatasetRGB
import torch
from typing import Tuple, Any
from easydict import EasyDict as edict

class Dataset(DatasetRGB):
    def __init__(self, opt: edict, target: str="train") -> None:
        super().__init__(opt, target)
        # self.intr = torch.tensor(self.opt.data.image_size if self.target=="train" else self.opt.render.image_size)
        assert self.opt.data.image_size[0] == self.opt.render.image_size[0] \
            and self.opt.data.image_size[1] == self.opt.render.image_size[1], "Must be matched between the data.image_size and render.image_size"

        self.intr = torch.tensor(self.opt.intrinsic).float()

    def get_camera(self, idx: int) -> Tuple[Any, torch.Tensor]:
        pose = self.raw_poses[idx]
        return self.intr, pose
