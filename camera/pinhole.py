import torch

def cam2img(X: torch.Tensor, cam_intr: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D points in camera coordinates to 2D image coordinates using a pinhole model.

    Args:
        X (torch.Tensor): [..., 3] 3D points in camera coordinates (Z>0 for valid projection).
        intr (torch.Tensor): [4] = (fx, fy, cx, cy).

    Returns:
        torch.Tensor: [..., 2] image coordinates in (x, y) format. 
                      Floating-point for sub-pixel precision.
    """
    fx, fy, cx, cy = cam_intr

    # Z must not be zero or negative for valid forward projection
    z = X[..., 2] + 1e-8

    x_img = (X[..., 0] / z) * fx + cx
    y_img = (X[..., 1] / z) * fy + cy
    return torch.stack([x_img, y_img], dim=-1)

def img2cam(X: torch.Tensor, cam_intr: torch.Tensor) -> torch.Tensor:
    """
    Convert 2D image coordinates to normalized 3D camera-space rays using a pinhole model.

    Args:
        uv (torch.Tensor): [..., 2] pixel coordinates in (x, y) format.
        intr (torch.Tensor): [4] = (fx, fy, cx, cy).

    Returns:
        torch.Tensor: [..., 3] normalized direction vectors in camera coordinates.
                      +X is to the right, +Y is down, +Z is forward (commonly used convention).
    """
    fx, fy, cx, cy = cam_intr

    x_img = X[..., 0]
    y_img = X[..., 1]

    # Convert from pixel to normalized camera coordinates
    x_cam = (x_img - cx) / fx
    y_cam = (y_img - cy) / fy

    dirs = torch.stack([torch.ones_like(x_cam), -x_cam, -y_cam], dim=-1)
    return dirs
