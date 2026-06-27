import torch
import math

def generate_circle_cameras(num_cameras: int = 12, radius: float = 3.5, height: float = 0.5):
    """Generate world-to-camera view matrices for cameras arranged in a circle looking at the origin."""
    viewmats = []
    for i in range(num_cameras):
        angle = 2 * math.pi * i / num_cameras
        cam_x = radius * math.cos(angle)
        cam_z = radius * math.sin(angle)
        cam_y = height

        # Negate the camera position to make the z-axis point toward the origin (0, 0, 0)
        z_cam = torch.tensor([-cam_x, -cam_y, -cam_z], device="cuda")
        z_cam = z_cam / z_cam.norm()

        up = torch.tensor([0.0, 1.0, 0.0], device="cuda")

        # torch.linalg.cross avoids the deprecation warning from torch.cross
        x_cam = torch.linalg.cross(up, z_cam)
        x_cam = x_cam / x_cam.norm()

        y_cam = torch.linalg.cross(z_cam, x_cam)

        R = torch.stack([x_cam, y_cam, z_cam], dim=0)
        cam_pos = torch.tensor([cam_x, cam_y, cam_z], device="cuda")
        t = -R @ cam_pos

        viewmat = torch.eye(4, device="cuda")
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        viewmats.append(viewmat)
    return torch.stack(viewmats, dim=0)

def get_camera_params(num_views: int = 12, width: int = 256, height: int = 256):
    """Return (viewmats, K) — camera parameters only, independent of any model or renderer."""
    K = torch.tensor([[[200.0, 0.0, width / 2.0],
                       [0.0, 200.0, height / 2.0],
                       [0.0, 0.0, 1.0]]], dtype=torch.float32, device="cuda")
    K = K.expand(num_views, -1, -1).contiguous()
    viewmats = generate_circle_cameras(num_cameras=num_views)
    return viewmats, K

def get_synthetic_dataset(num_views: int = 12, width: int = 256, height: int = 256,
                          gaussian_model=None, renderer=None):
    """
    Generate a synthetic dataset of num_views images from cameras arranged in a circle.

    Args:
        gaussian_model: Pre-initialised teacher model (nn.Module on CUDA).
                        If None, a default VanillaGaussian is created for backward compatibility.
        renderer:       Pre-initialised renderer. If None, a default GsplatRenderer is created.
    """
    viewmats, K = get_camera_params(num_views, width, height)

    if gaussian_model is None:
        from opengs.models.vanilla_gs import VanillaGaussian
        gaussian_model = VanillaGaussian(num_points=300).cuda()
        with torch.no_grad():
            random_directions = torch.randn_like(gaussian_model.params["means"])
            random_directions /= random_directions.norm(dim=-1, keepdim=True)
            gaussian_model.params["means"].copy_(random_directions * 0.8)
            gaussian_model.params["scales"].fill_(-3.0)
            gaussian_model.params["opacities"].fill_(2.0)

    if renderer is None:
        from opengs.renderers.gsplat_renderer import GsplatRenderer
        renderer = GsplatRenderer(rasterize_mode="antialiased")

    images = []
    with torch.no_grad():
        for i in range(num_views):
            img, _ = renderer.render(
                gaussian_model=gaussian_model,
                viewmat=viewmats[i:i+1],
                K=K[i:i+1],
                width=width,
                height=height
            )
            images.append(img)

    return images, viewmats, K
