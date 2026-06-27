import torch
import torch.nn as nn
import numpy as np

# SH DC basis constant: Y_0^0 = 1/(2*sqrt(pi))
SH_C0 = 0.28209479177387814


def _num_sh_coeffs(degree: int) -> int:
    """Total number of SH coefficients up to `degree`: (degree+1)^2."""
    return (degree + 1) ** 2


def _rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert [N, 3] RGB colors (in [0, 1]) to [N, 1, 3] SH degree-0 coefficients.
    Inverse of:  RGB = SH_C0 * f_dc + 0.5
    """
    return ((rgb - 0.5) / SH_C0).unsqueeze(1)  # [N, 1, 3]


def _init_log_scales_from_xyz(xyz: torch.Tensor) -> torch.Tensor:
    """
    Estimate per-point isotropic log-scale from nearest-neighbour distance.
    Always runs on CPU in chunks to avoid GPU OOM regardless of point cloud size.
    """
    device = xyz.device
    pts = xyz.detach().cpu().float()
    n   = pts.shape[0]

    REF_SIZE = 8_000
    CHUNK    = 2_000

    if n <= REF_SIZE:
        ref = pts
    else:
        idx = torch.randperm(n)[:REF_SIZE]
        ref = pts[idx]

    nn_dists = torch.empty(n, dtype=torch.float32)

    for start in range(0, n, CHUNK):
        end   = min(start + CHUNK, n)
        chunk = pts[start:end]
        d     = torch.cdist(chunk, ref)          # [c, REF_SIZE] — CPU only
        if n <= REF_SIZE:
            # Exclude self-distances: shift diagonal per chunk offset
            for i in range(end - start):
                global_i = start + i
                if global_i < ref.shape[0]:
                    d[i, global_i] = float("inf")
        nn_dists[start:end] = d.min(dim=1).values

    nn_dists = nn_dists.clamp(min=1e-6)
    log_scale = torch.log(nn_dists).unsqueeze(-1).expand(-1, 3).contiguous()
    return log_scale.to(device)


class VanillaGaussian(nn.Module):
    """
    3D Gaussian model with Spherical Harmonics (SH) view-dependent color.

    Color is stored as two separate parameters:
      sh_dc   [N, 1, 3]      — degree-0 (DC) SH coefficient
      sh_rest [N, K-1, 3]    — degrees 1…max_sh_degree coefficients

    Keeping them separate allows the optimizer to assign different learning
    rates (DC is learned faster than higher-order coefficients).
    """

    max_sh_degree: int = 3   # class-level default; instances can override

    def __init__(self, num_points: int = 100, max_sh_degree: int = 3):
        super().__init__()
        self.num_points    = num_points
        self.max_sh_degree = max_sh_degree

        K = _num_sh_coeffs(max_sh_degree)  # 16 for degree 3

        # Random RGB → SH DC; rest initialized to 0
        rand_rgb = torch.rand(num_points, 3)
        sh_dc_init   = _rgb_to_sh_dc(rand_rgb)              # [N, 1, 3]
        sh_rest_init = torch.zeros(num_points, K - 1, 3)    # [N, 15, 3]

        self.params = nn.ParameterDict({
            "means":    nn.Parameter(torch.randn((num_points, 3)) * 0.5),
            "quats":    nn.Parameter(torch.randn((num_points, 4))),
            "scales":   nn.Parameter(torch.randn((num_points, 3)) * 0.1 - 2.0),
            "opacities":nn.Parameter(torch.randn((num_points,)) * 0.1),
            "sh_dc":    nn.Parameter(sh_dc_init),
            "sh_rest":  nn.Parameter(sh_rest_init),
        })

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def means(self):
        return self.params["means"]

    @property
    def quats(self):
        return self.params["quats"]

    @property
    def scales(self):
        return self.params["scales"]

    @property
    def opacities(self):
        return self.params["opacities"]

    @property
    def sh_dc(self):
        return self.params["sh_dc"]       # [N, 1, 3]

    @property
    def sh_rest(self):
        return self.params["sh_rest"]     # [N, K-1, 3]

    @property
    def colors(self):
        """Full SH coefficient tensor [N, K, 3] used by the renderer."""
        return torch.cat([self.sh_dc, self.sh_rest], dim=1)

    # ------------------------------------------------------------------ #
    # Activations
    # ------------------------------------------------------------------ #
    def get_normalized_quats(self):
        return self.quats / self.quats.norm(dim=-1, keepdim=True)

    def get_real_scales(self):
        return torch.exp(self.scales)

    def get_real_opacities(self):
        return torch.sigmoid(self.opacities)

    # ------------------------------------------------------------------ #
    # Factory: initialize from COLMAP sparse point cloud
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pointcloud(cls,
                        xyz: torch.Tensor,
                        rgb: torch.Tensor,
                        max_sh_degree: int = 3) -> "VanillaGaussian":
        """
        Initialize Gaussians from a COLMAP sparse point cloud.

        Args:
            xyz:            Float32 [N, 3] — world-space 3D positions.
            rgb:            Float32 [N, 3] — colors in [0, 1].
            max_sh_degree:  Maximum SH degree to store (default 3, like paper).
        """
        n  = xyz.shape[0]
        K  = _num_sh_coeffs(max_sh_degree)
        dv = xyz.device

        # Rotation: identity quaternion (w=1, x=y=z=0)
        quats = torch.zeros(n, 4, dtype=torch.float32, device=dv)
        quats[:, 0] = 1.0

        # Scale: estimated from nearest-neighbour distance (isotropic)
        log_scales = _init_log_scales_from_xyz(xyz.float())

        # Opacity: logit(0.1) ≈ -2.2 → sigmoid ≈ 0.1 (sparse start)
        opacities = torch.full((n,), -2.2, dtype=torch.float32, device=dv)

        # SH: DC from COLMAP RGB, higher degrees = 0
        sh_dc   = _rgb_to_sh_dc(rgb.float().clamp(0, 1))   # [N, 1, 3]  on dv
        sh_rest = torch.zeros(n, K - 1, 3, dtype=torch.float32, device=dv)

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.num_points    = n
        instance.max_sh_degree = max_sh_degree
        instance.params = nn.ParameterDict({
            "means":     nn.Parameter(xyz.float()),
            "quats":     nn.Parameter(quats),
            "scales":    nn.Parameter(log_scales),
            "opacities": nn.Parameter(opacities),
            "sh_dc":     nn.Parameter(sh_dc),
            "sh_rest":   nn.Parameter(sh_rest),
        })
        return instance
