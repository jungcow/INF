from . import modelwrapper, density, base
from easydict import EasyDict as edict
from util import log, make_transformation, make_transformation_cuda
import torch
import tqdm
import util_vis
import os
import torch.nn.functional as torch_F
import transforms
import importlib
import imageio
import json
import util
from typing import Any, Tuple, List
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lpipsPyTorch import lpips

CAM_NUM=4

class Model():
    def __init__(self, opt, model_idx):
        self.model_idx = model_idx

    def build_network(self, super, opt) -> None:
        """
        build renderer class in self.renderer and pose class in self.pose. 
        And restore checkpoints.

        Args:
            opt (edict[str, Any]): opt
        """
        pose = importlib.import_module(f"model.{opt.model}")
        self.pose = pose.Pose(opt, self.model_idx).to(opt.device)
        self.restore_checkpoint(super, opt)

    def set_optimizer(self, super, opt) -> None:
        """
        set optim_f for field optimizer and optim_p for pose optimizer.
        learning rate will be restore from checkpoints or reset for training lidar poses during training process 

        Args:
            opt (edict[str, Any]): opt
        """
        if opt.poses:
            self.pose_lr = opt.lr.pose
            self.optim_p = torch.optim.Adam(params=self.pose.parameters(), lr=self.pose_lr)

    def create_dataset(self, super, opt) -> None:
        """
        create dataset variables. Basically data are not loaded unless necessary.

        Args:
            opt (edict[str, Any]): opt
        """
        data = importlib.import_module(f"data.{opt.data.sensor}")
        log.info("creating dataset...")
        self.train_data = data.Dataset(opt, target="train")
        if opt.model=="color" or opt.model=="multicolor":
            self.vis_data = data.Dataset(opt,target="vis")

    def get_rays(self, super, opt: edict[str, Any], var: edict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, edict[str, torch.Tensor]]:
        
        # --- pre-calculate directions for a image frame ---
        if not hasattr(self, "train_dirs"):
            self.camera = importlib.import_module(f"camera.{opt.camera}")
            self.train_dirs = util.rays_from_image(opt, opt.W, opt.H, self.camera, self.train_data.intr)
        
        # --- scale of image ---
        scale_mask = torch.zeros((opt.H, opt.W), device=opt.device, dtype=torch.bool)
        steps = [m for m, _ in opt.train.multi_scale] # opt.train.multi_scale: [[iter, scale], [iter, scale], ...] -> step:[iter, iter, ...]
        for n, s in enumerate(zip(steps, steps[1:] + [float("inf")])): 
            # [[iter1, iter2], [iter2, iter3], ..., [iter_last, inf]]
            it = super.it//CAM_NUM
            if it >= s[0] and it < s[1]:
                scale = opt.train.multi_scale[n][1]
                break
        scale_mask[::scale, ::scale] = True 
        scale_mask = scale_mask.reshape(-1)
        # --- masked points ---
        if var.mask is not None:
            scale_mask = scale_mask & var.mask
        # --- samples indices ---
        all_indices = torch.arange(opt.H*opt.W, device=opt.device)[scale_mask]
        selected = (torch.rand(opt.train.rand_rays // len(var.idx), device=opt.device) * len(all_indices)).to(int) # shape of ray per frames
        indices = all_indices[selected]
        # --- project the indices to a cosine distribution (for panorama camera)
        if opt.camera == "panorama":
            row = torch.div(indices, opt.W, rounding_mode="floor")
            column = indices % opt.W
            # inverse transform sampling
            row_c = ((torch.arcsin((row/opt.H - 0.5) * 2) / torch.pi + 0.5)* opt.H).to(int)
            new_indices = row_c * opt.W + column
        # --- remove the indices that fall in to masked area after inverse transform sampling ---
            new_indices = new_indices[var.mask[new_indices]] # [HW]
            new_indices = new_indices.cpu()
        else:
            new_indices = indices.cpu()
        # --- transform camera rays to world space
        cdirs = self.train_dirs[new_indices].to(opt.device).repeat(len(var.idx), 1, 1) # [B, HW, 3]
        c2w = self.pose(var.pose) # [B, 3, 4]
        dirs = cdirs @ c2w[..., :3].transpose(1, 2) # [B, HW, 3]
        origins = c2w[..., 3][:, None, :].expand_as(dirs) # [B, HW, 3]

        # --- ground truth rgb ---
        gt_rgb = var.image[:, new_indices].to(opt.device) # [B, HW, 3]
        return dirs, origins, edict(rgb=gt_rgb, dirs=dirs, origins=origins, cposes=c2w, lposes=var.pose)

    def compute_loss(self, super, opt: edict[str, Any], gt: edict[str, torch.Tensor], res: edict[str, torch.Tensor]) -> edict[str, torch.Tensor]:
        rgb_loss = ((gt.rgb - res.rgb) ** 2).mean()
        return edict(all=rgb_loss)

    @torch.no_grad()
    def validate(self, super, opt: edict[str, Any]) -> None:
        log.info("validating")
        super.renderer.eval()

        os.makedirs(os.path.join(opt.output_path, "figures"), exist_ok=True)
        def psnr(img1, img2):
            mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
            return 20 * torch.log10(1.0 / torch.sqrt(mse))

        def gaussian(window_size, sigma):
            gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()
        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
            return window

        def ssim(img1, img2, window_size=11, size_average=True):
            channel = img1.size(-3)
            window = create_window(window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            return _ssim(img1, img2, window, window_size, channel, size_average)

        def _ssim(img1, img2, window, window_size, channel, size_average=True):
            mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            if size_average:
                return ssim_map.mean()
            else:
                return ssim_map.mean(1).mean(1).mean(1)

        psnrs = []
        ssims = []
        lpipss = []
        for var_original in tqdm.tqdm(self.vis_data):
            var_original = edict(var_original)
            # save the rendered images
            os.makedirs(os.path.join(opt.output_path, "figures", f"{var_original.idx}_depth"), exist_ok=True)
            os.makedirs(os.path.join(opt.output_path, "figures", f"{var_original.idx}_rgb"), exist_ok=True)
            
            # render
            var_original = edict(var_original)
            pose = self.pose(var_original.pose.to(opt.device))
            var = self.render_by_slices(super, opt, pose)

            # depth result
            depth_c = util_vis.to_color_img(opt, var.depth)
            util_vis.tb_image(opt, 
                              super.tb_writers[self.model_idx], 
                              super.it//CAM_NUM + 1, 
                              "vis", f"{var_original.idx}_depth", depth_c)
            imageio.imwrite(os.path.join(opt.output_path, "figures", f"{var_original.idx}_depth", f"{super.it//CAM_NUM}.png"), util_vis.to_np_img(depth_c))
            # rgb image
            rgb_map = var.rgb.view(-1,opt.render_H,opt.render_W,3).permute(0,3,1,2) # [B,3,H,W]
            util_vis.tb_image(opt,
                              super.tb_writers[self.model_idx],
                              super.it//CAM_NUM+1,
                              "vis", f"{var_original.idx}_rgb",rgb_map)
            imageio.imwrite(os.path.join(opt.output_path, "figures", f"{var_original.idx}_rgb", f"{super.it//CAM_NUM}.png"), util_vis.to_np_img(rgb_map))
            # original image
            origin_image = var_original.image.reshape(1, opt.render_H, opt.render_W, 3).permute(0, 3, 1, 2)
            util_vis.tb_image(opt,
                              super.tb_writers[self.model_idx],
                              super.it//CAM_NUM+1,"vis",
                              f"{var_original.idx}_origin",origin_image)
            
            # breakpoint()
            psnrs.append(psnr(origin_image.cuda(), rgb_map))
            ssims.append(ssim(origin_image.cuda(), rgb_map))
            lpipss.append(lpips(origin_image.cuda(), rgb_map))

        self.psnr = torch.tensor(psnrs).mean()
        self.ssim = torch.tensor(ssims).mean()
        self.lpips = torch.tensor(lpipss).mean()


    @torch.no_grad()
    def render_by_slices(self, super, opt: edict[str, Any], pose: torch.Tensor) -> edict[str, torch.Tensor]:
        """
        render a whole frame with a pose

        Args:
            opt (edict[str, Any]):  opt
            pose (tensor): [3, 4]
            
        Returns:
            edict[str, torch.Tensor]: depth=tensor with shape [HW]; rgb=tensor with shape [HW, 3] if there is rgb
        """
        # --- pre-calculate directions for a image frame ---
        # if not hasattr(self, "train_dirs"):
        #     self.camera = importlib.import_module(f"camera.{opt.camera}")
        #     self.train_dirs = util.rays_from_image(opt, opt.W, opt.H, self.camera, self.train_data.intr)

        if not hasattr(self, "render_dirs"):
            # camera = importlib.import_module(f"camera.{opt.render_camera}")
            self.camera = importlib.import_module(f"camera.{opt.camera}")
            self.render_dirs = util.rays_from_image(opt, opt.W, opt.H, self.camera, self.train_data.intr)
        ret_all = edict(depth=[])
        if opt.model=="color" or opt.model=="multicolor":
            ret_all.update(rgb=[])
        # render the image by slices for memory considerations
        for c in range(0,opt.H * opt.W,opt.train.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.train.rand_rays,opt.H * opt.W))
            dirs = self.render_dirs[ray_idx].to(opt.device) # [HW, 3]
            # transform to world spaces
            dirs = dirs @ pose[..., :3].transpose(0, 1) # [HW, 3]
            origins = pose[:, 3].repeat(ray_idx.shape[0], 1) # [HW, 3]
            # render the rays
            ret = super.renderer(opt, dirs[None], origins[None], mode="vis") # [1, HW, 3]
            for key in ret_all: ret_all[key].append(ret[key][0]) # [HW] or [HW, 3]
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=0)
        return ret_all  

    @torch.no_grad()
    def log_scalars(self, super, opt: edict[str, Any], idx: List[int] = [0]) -> None:
        # log loss:
        for key, value in super.loss.items():
            super.tb_writers[self.model_idx].add_scalar(f"train/{key}_loss", 
                                                       value, 
                                                       super.it//CAM_NUM + 1)

        # log learning rate:
        super.field_req and super.tb_writers[self.model_idx].add_scalar(f"train/lr_field", 
                                                                      super.optim_f.param_groups[0]["lr"], 
                                                                      super.it//CAM_NUM + 1)
        super.pose_req and super.tb_writers[self.model_idx].add_scalar(f"train/lr_pose", 
                                                                     self.optim_p.param_groups[0]["lr"],
                                                                     super.it//CAM_NUM + 1)

        # log pose
        pose = self.pose.SE3()
        R = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0]
        ], dtype=np.float32) # cam_INF to cam(z-forward, x-right, y-downward)
        pose = transforms.pose.compose_pair(torch.tensor(R).float().cuda(), pose)
        euler, trans = transforms.get_ang_tra(pose)
        util_vis.tb_log_ang_tra(super.tb_writers[self.model_idx], 
                                "ext", None, euler, trans,
                                super.it//CAM_NUM + 1)
        res = {"rotation": euler, "translation": trans}
        
        # log pose error if there is ground truth pose
        if self.pose.ref_ext is not None:
            pose_e = transforms.pose.relative(pose, self.pose.ref_ext)
            euler_e, trans_e = transforms.get_ang_tra(pose_e)
            util_vis.tb_log_ang_tra(super.tb_writers[self.model_idx], "ext_error", None, euler_e, trans_e, super.it//CAM_NUM + 1)
            res.update(
                rotation_error=euler_e, 
                rotation_norm_error=np.linalg.norm(euler_e),
                translation_error=trans_e,
                translation_norm_error = np.linalg.norm(trans_e),
                PSNR = self.psnr.item(),
                SSIM = self.ssim.item(),
                LPIPS=self.lpips.item())
            
        # save in a json file
        with open(os.path.join(opt.output_path,"res.json"), "w") as f:
            json.dump(res, f)

    def save_checkpoint(self, super, opt: edict[str, Any], status: str="base") -> None:
        """
        save checkpoint, including 
        current epochs (only change when optimizing LiDAR poses), 
        current iteration, learning rates, status (only change when optimizing LiDAR poses)
        and the parameters in renderer and pose classes

        Args:
            opt (edict[str, Any]): opt
            status (str, optional): used when optimizing LiDAR poses. Defaults to "base".
        """
        checkpoint = dict(
            epoch=super.ep,
            iter=super.it//CAM_NUM,
            status=status,
            renderer=super.renderer.state_dict(),
        )
        opt.poses and checkpoint.update(pose=self.pose.state_dict())
        super.it//CAM_NUM>0 and super.field_req and super.sched_f and checkpoint.update(lr_f=super.optim_f.param_groups[0]["lr"])
        super.it//CAM_NUM>0 and super.pose_req and super.sched_p and checkpoint.update(lr_p=self.optim_p.param_groups[0]["lr"])

        torch.save(checkpoint,f"{opt.output_path}/model.ckpt")
        log.info(f"checkpoint saved: (epoch {super.ep} (iteration {super.it//CAM_NUM})") 

    def restore_checkpoint(self, super, opt: edict[str, Any]) -> None:
        """
        restore the information saved including
        current epochs (only change when optimizing LiDAR poses), 
        current iteration, learning rates, status (only change when optimizing LiDAR poses)
        and the parameters in renderer and pose classes

        Args:
            opt (edict[str, Any]): opt
        """
        log.info("resuming from previous checkpoint...")
        
        get_child_state_dict = lambda state_dict, key: { ".".join(k.split(".")[1:]): v for k,v in state_dict.items() if k.startswith(f"{key}.")}
        load_name = f"{opt.output_path}/model.ckpt"

        if not os.path.exists(load_name):
            return
        checkpoint = torch.load(load_name,map_location=opt.device)
        # load modules in renderer (mlp for density/color fields)
        child_state_dict = get_child_state_dict(checkpoint["renderer"],"field")
        if child_state_dict:
            print("restoring field...")
            super.renderer.field.load_state_dict(child_state_dict)
        # load pose parameters
        if opt.poses:
            self.pose.load_state_dict(checkpoint["pose"])
        # load train progress
        super.ep = checkpoint["epoch"]
        super.iter_start = 4*checkpoint["iter"] + self.model_idx
        super.status = checkpoint["status"]
        print(f"resuming from epoch {super.ep} (iteration {super.iter_start})")
        # load learning rate
        for key in ["lr_f", "lr_p"] :
            if key in checkpoint:
                super.optim_lr[key] = checkpoint[key]

class Pose(torch.nn.Module):
    """
    extrinsic parameters
    """
    def __init__(self, opt, model_idx):
        super().__init__()
        # --- extrinsic parameters requiring gradient---     
        self.ext = torch.nn.Embedding(1, 6).to(opt.device)
        torch.nn.init.zeros_(self.ext.weight)

        # --- load the reference extrinsic parameter ----
        ref_path = os.path.join("data", opt.data.scene, "ref_ext.json") # L2C
        if os.path.exists(ref_path):
            with open(ref_path, "r") as file:
                ref = json.load(file)
            # angles are euler angles in xyz order and in degrees
            #! multicolor model expects that ref json has multiple rot & trans transformations
            self.ref_ext = transforms.ang_tra_to_SE3(opt, ref[model_idx]["rotation"], ref[model_idx]["translation"])
            # In our reference extrinsic parameters, we use lidar-to-camera transformation. While in this work, we first project camera poses to LiDAR spaces, causing the result extrinsic parameters to be camera-to-lidar transformation. 
            # Thus, we need to inverse it here.
            self.ref_ext = transforms.pose.invert(self.ref_ext, use_inverse=True) # C2L
        else:
            self.ref_ext = None

        if 'random_noise' in opt.train:
            self.random_noise = opt.train.random_noise
            rot_noise_bound = self.random_noise[0]
            trans_noise_bound = self.random_noise[1]

            # random sign
            r_noise = [-1 ** np.random.randint(1, 3) * rot_noise_bound for _ in range(3)]
            t_noise = [-1 ** np.random.randint(1, 3) * trans_noise_bound for _ in range(3)]

            def make_transform(axis, T, degree):
                axis = axis / np.linalg.norm(axis)
                rad = np.radians(degree)
                c = np.cos(rad)
                s = np.sin(rad)
                t = 1 - c
                x = axis[0]
                y = axis[1]
                z = axis[2]
                
                return np.array([
                    [t*x*x + c, t*x*y - s*z, t*x*z + s*y, T[0]],
                    [t*x*y + s*z, t*y*y + c, t*y*z - s*x, T[1]],
                    [t*x*z - s*y, t*y*z + s*x, t*z*z + c, T[2]],
                    [0, 0, 0, 1]
                ])

            noise_transform = make_transform(axis=np.array([1, 0, 0]), T=t_noise, degree=r_noise[0]) @ \
                            make_transform(axis=np.array([0, 1, 0]), T=np.zeros(3), degree=r_noise[1]) @ \
                            make_transform(axis=np.array([0, 0, 1]), T=np.zeros(3), degree=r_noise[2])

            noise = torch.tensor(noise_transform[:3]).float().cuda()
            pose = transforms.pose.compose_pair(noise, self.ref_ext)

            """
            C2C: cam_INF to cam
            - cam_INF coordinate convention: (x: forward, y: left, z: upward)
            - general camera coordinate convention: (z-forward, x-right, y-downward)
            """
            C2C = np.array([
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0]
            ], dtype=np.float32)
            C2C_inv = transforms.pose.invert(torch.tensor(C2C).float().cuda())
            pose = transforms.pose.compose_pair(torch.tensor(C2C_inv).float().cuda(), pose)
        else:
            # --- initial value ---
            # init euler angles are in degrees, xyz arrangement
            rot = opt.extrinsic[:3] if "extrinsic" in opt else [0, 0, 0] 
            trans = opt.extrinsic[-3:] if "extrinsic" in opt else [0, 0, 0]
            pose = transforms.pose.invert(transforms.ang_tra_to_SE3(opt, rot, trans), use_inverse=True)

        self.init = transforms.lie.SE3_to_se3(pose) #(6)


    def SE3(self) -> torch.Tensor:
        """
        return SE3 matrix of ext

        Returns:
            torch.Tensor: tensor with shape [3, 4]
        """
        return transforms.lie.se3_to_SE3(self.ext.weight[0] + self.init)

    def forward(self, l2w) -> torch.Tensor:
        """
        given lidar to world poses, return camera to world poses

        Args:
            l2w (tensor): with shape [..., 3, 4]

        Returns:
            torch.Tensor: with shape [..., 3, 4]
        """
        c2l = self.SE3() # [3, 4]
        new_poses = transforms.pose.compose_pair(c2l, l2w) # l2w @ c2l = c2w
        return new_poses