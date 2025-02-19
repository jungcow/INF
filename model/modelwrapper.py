import os, importlib
from util import log, update_timer
import torch
import torch.utils.tensorboard
from easydict import EasyDict as edict
import time
import tqdm
from typing import Union, List, Tuple, Optional, Any
import util
import transforms
import util_vis
import numpy as np
import json
from . import density
import torch.nn.functional as torch_F

class ModelWrapper():
    """model is to control the whole training process
    """
    def __init__(self, model_list, opt_list) -> None:
        # tensorboard
        self.tb_writers = [torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path,flush_secs=10) 
                           for opt in opt_list]
        # variables to control training process
        self.ep, self.iter_start, self.status = 0, 0, "base"
        # variables to control whether to optimize density/color field, whether to use sheduler for the optimizers
        self.field_req, self.pose_req, self.sched_f, self.sched_p = True, True, False, False
        self.optim_lr = {}

        self.model_list = [m.Model(opt, idx) for idx, (opt, m) in enumerate(zip(opt_list, model_list))]

    def build_network(self, opt_list) -> None:
        """
        build renderer class in self.renderer and pose class in self.pose. 
        And restore checkpoints.

        Args:
            opt (edict[str, Any]): opt
        """
        tmp_opt = opt_list[0]
        log.info("building networks...")
        # renderer = importlib.import_module(f"model.{tmp_opt.model}")
        self.renderer = Renderer(tmp_opt).to(tmp_opt.device) # shared field (density, color fields)
        if tmp_opt.poses:
            for opt, m in zip(opt_list, self.model_list):
                m.build_network(self, opt) # Pose initialization and restore checkpoint

    def set_optimizer(self, opt_list) -> None:
        """
        set optim_f for field optimizer and optim_p for pose optimizer.
        learning rate will be restore from checkpoints or reset for training lidar poses during training process 

        Args:
            opt (edict[str, Any]): opt
        """
        tmp_opt = opt_list[0]
        log.info("setting up optimizers...")
        # optimizer for color/density field
        self.field_lr = tmp_opt.lr.field
        self.optim_f = torch.optim.Adam(params=self.renderer.field.parameters(),lr=self.field_lr)
        # optimizer for lidar poses/extrinsic parameters
        if tmp_opt.poses:
            for opt, m in zip(opt_list, self.model_list):
                m.set_optimizer(self, opt) # set pose_lr and optimizer for self.pose.parameters()

    def create_dataset(self, opt_list) -> None:
        """
        create dataset variables. Basically data are not loaded unless necessary.

        Args:
            opt (edict[str, Any]): opt
        """

        log.info("creating dataset...")
        for opt, m in zip(opt_list, self.model_list):
            m.create_dataset(self, opt) # set train_data (poses(lidar_poses), intrinsic)

    def train(self, opt_list) -> None:
        """
        the whole traning process

        Args:
            opt (edict[str, Any]): opt
        """
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)

        tmp_opt = opt_list[0]
        
        iter = tmp_opt.train.iteration
        loader = tqdm.trange(self.iter_start, iter, desc="calib", leave=True)
        self.sched_f, self.sched_p = False, False
        
        train_data_list = []
        var_list = []
        for idx, (opt, m) in enumerate(zip(opt_list, self.model_list)):
            m.train_data.prefetch_all_data()
            var = m.train_data.all
            var_list.append(var)

        for self.it in loader:
            idx = self.it % 4
            # train iteration
            var = var_list[idx]
            opt = opt_list[idx]
            m = self.model_list[idx]
            self.train_iteration(m, opt, var, loader)

    def train_iteration(self, model, opt: edict[str, Any], var: edict, loader: tqdm.std.tqdm) -> None:
        """
        in every training iteration, 
        - we retrieve poses, 
        - trasform rays to world space 
        - and render with these rays
        - and update the parameters

        Args:
            model: Multicolor
            opt (edict[str, Any]): opt
            var (edict[str, torch.Tensor]): edict of used training data
            loader (tqdm.std.tqdm): tqdm progress bar
        """
        # --- before train iteration ---
        
        self.renderer.train()
        self.timer.it_start = time.time()
        self.field_req and self.optim_f.zero_grad()
        self.pose_req and model.optim_p.zero_grad()

        # --- train iteration ---

        # get rays, render and calculate loss
        dirs, origins, gt = model.get_rays(self, opt, var) # in child model
        res = self.renderer(opt, dirs, origins, mode="train") # in wrapper
        self.loss = model.compute_loss(self, opt, gt, res) # in child model
        # optimizer
        self.loss.all.backward()
        self.field_req and self.optim_f.step() # in wrapper
        self.field_req and self.sched_f and self.sched_f.step() # in wrapper
        self.pose_req and model.optim_p.step() # in wrapper
        self.pose_req and self.sched_p and self.sched_p.step() # in wrapper

        # --- after training iteration ---
        
        if (self.it//4)==0 or ((self.it//4) + 1) % opt.freq.val == 0: 
            model.validate(self, opt) # in child model
        if (self.it//4)==0 or ((self.it//4) + 1) % opt.freq.scalar == 0: 
            model.log_scalars(self, opt, var.idx) # in child model
        if ((self.it//4) + 1) % opt.freq.ckpt == 0:
            model.save_checkpoint(self, opt, status=self.status) # in child model
        loader.set_postfix(it=(self.it//4),loss=f"{self.loss.all:.3f}")
        self.timer.it_end = time.time()
        update_timer(opt,self.timer,self.ep,len(loader))

    def end_process(self, opt_list) -> None:
        """
        close every opened files

        Args:
            opt (edict[str, Any]): opt
        """
        model_num = len(self.model_list)
        for idx in range(model_num):
            self.tb_writers[idx].flush()
            self.tb_writers[idx].close()
        log.title("PROCESS END")

class Renderer(torch.nn.Module):
    """
    renderer class is used to perform volume rendering for given rays.
    """

    def __init__(self,opt):
        super().__init__()
        self.density_field = Density(opt.density_opt)
        self.field = Color(opt)

    def sample_points(self, opt: edict[str, Any], dirs: torch.Tensor, origins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample points along the ray
        
        Args:
            dirs: shape of (..., 3)
            origins: shape of (..., 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
            1. point coordinates: shape of (..., #points every ray, 3)
            2. depths(distance) of the points on the ray: shape of (..., #points every ray)
        """
        depth_min,depth_max = opt.data.near_far
        si = dirs.shape[:-1]
        rand_samples = torch.rand(*si,opt.train.sample_intvs,device=opt.device) 
        add_ = torch.arange(opt.train.sample_intvs,device=opt.device).reshape(*[1 for _ in range(len(si))], opt.train.sample_intvs)
        rand_samples += add_.float()
        depth_samples = rand_samples/opt.train.sample_intvs*(depth_max-depth_min)+depth_min # (-1, N)
        points = depth_samples.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2)
        return points, depth_samples

    def forward(self, opt: edict[str, Any], dirs: torch.Tensor, origins: torch.Tensor, mode: str="train") -> edict[str, torch.Tensor]:
        """
        given rays and return the rendered depths, colors

        Args:
            opt (edict[str, Any]): opt
            dirs (torch.Tensor): tensor with shape [B, HW, 3]
            origins (torch.Tensor): tensor with shape [B, HW, 3]
            mode (str, optional): not used here. Defaults to "train".

        Returns:
            edict[str, torch.Tensor]: 1. rgb: tensor with shape [B, HW, 3]; 2. depth: tensor with shape [B, HW]
        """
        point_samples, depth_samples = self.sample_points(opt, dirs, origins) # [B, HW, N, 3], [B, HW, N]
        density_samples = self.density_field(opt.density_opt, point_samples)
        weights = self.density_field.composite(opt.density_opt,density_samples,depth_samples) # [B, HW, N]
        depth = (weights * depth_samples).sum(dim=-1)
        # give the farthest point color, 
        # otherwise the unreached rays will be all black
        weights[..., -1] = 1 - weights[..., :-1].sum(dim=-1)
        colors = self.field(opt, point_samples)
        rgb = (weights[..., None] * colors).sum(dim=-2)
        return edict(rgb=rgb,depth=depth) #[B, HW]

class Color(density.Density):

    def define_network(self, opt: edict[str, Any]) -> None:
        get_layer_dims = lambda layers: list(zip(layers[:-1],layers[1:]))
        input_3D_dim = 3+6*opt.arch.posenc.L_3D
        self.mlp_feat = torch.nn.ModuleList()
        L = get_layer_dims(opt.arch.layers_feat)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_3D_dim
            if li in opt.arch.skip: k_in += input_3D_dim
            linear = torch.nn.Linear(k_in,k_out)
            self.tensorflow_init_weights(opt,linear,out="first" if li==len(L)-1 else None)
            self.mlp_feat.append(linear)

    def forward(self, opt: edict[str, Any], points_3D: torch.Tensor) -> torch.Tensor: 
        """
        give colors of the sampled potins

        Args:
            opt (edict[str, Any]): opt
            points_3D (torch.Tensor): with shape [B, HW, N, 3]

        Returns:
            torch.Tensor: with shape [B, HW, N, 3]
        """
        points_enc = self.positional_encoding(opt,points_3D,L=opt.arch.posenc.L_3D)
        points_enc = torch.cat([points_3D,points_enc],dim=-1) # [B, HW, N,6L+3]
        feat = points_enc
        # extract coordinate-based features
        for li, layer in enumerate(self.mlp_feat):
            if li in opt.arch.skip: feat = torch.cat([feat, points_enc], dim=-1)
            feat = layer(feat)
            if li!=len(self.mlp_feat)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B,HW, N,3]
        return rgb

class Density(density.Density):
    def __init__(self, opt: edict[str, Any]) -> None:
        super().__init__(opt)
        # load the trained density field
        self.restore_param(opt)
        for p in self.parameters():
            p.requires_grad_(False)

    def restore_param(self, opt: edict[str, Any]) -> None:
        """
        restore the trained density field parameters
        
        Args:
            opt (edict[str, Any]): opt
        """
        load_name = f"{opt.output_path}/model.ckpt"
        assert os.path.exists(load_name), "density field not found"
        checkpoint = torch.load(load_name,map_location=opt.device)
        get_child_state_dict = lambda state_dict, key: { ".".join(k.split(".")[1:]): v for k,v in state_dict.items() if k.startswith(f"{key}.")}
        child_state_dict = get_child_state_dict(checkpoint["renderer"], "field")
        self.load_state_dict(child_state_dict)
