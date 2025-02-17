import torch
from typing import List

def cam2img(X_cam: torch.Tensor, cam_intr: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D camera coordinates to image pixel coordinates.

    Args:
        X_cam (torch.Tensor): Camera coordinates [N, 3] (X, Y, Z in camera frame).
        cam_intr (torch.Tensor): Camera intrinsics [4] (f_x, f_y, c_x, c_y).

    Returns:
        torch.Tensor: Image coordinates [N, 2] (u, v).
    """
    f_x, f_y, c_x, c_y = cam_intr
    X, Y, Z = X_cam[..., 0], X_cam[..., 1], X_cam[..., 2]
    
    # Perspective projection to image coordinates
    u = f_x * (X / Z) + c_x
    v = f_y * (Y / Z) + c_y
    
    return torch.stack((u, v), dim=-1)  # Shape [N, 2]


def img2cam(X_img: torch.Tensor, cam_intr: torch.Tensor) -> torch.Tensor:
    """
    Convert image pixel coordinates to camera ray directions.

    Args:
        X_img (torch.Tensor): Image coordinates [N, 2] (u, v).
        cam_intr (torch.Tensor): Camera intrinsics [4] (f_x, f_y, c_x, c_y).

    Returns:
        torch.Tensor: Camera ray directions [N, 3], normalized.
    """
    f_x, f_y, c_x, c_y = cam_intr
    
    # Compute camera coordinates
    X_cam = (X_img[..., 0] - c_x) / f_x  # (u - c_x) / f_x
    Y_cam = (X_img[..., 1] - c_y) / f_y  # (v - c_y) / f_y
    Z_cam = torch.ones_like(X_cam)  # Set depth to 1
    
    directions = torch.stack((X_cam, Y_cam, Z_cam), dim=-1)
    
    # Normalize ray directions
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    return directions  # Shape [N, 3]
