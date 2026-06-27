import numpy as np
import torch


def save_ply(model, path: str):
    """
    Export a VanillaGaussian model to a standard binary PLY file compatible
    with common 3DGS viewers (SuperSplat, Gaussian Splatting Viewer, etc.).

    The PLY format mirrors the original INRIA 3DGS repository:
      - x, y, z, nx, ny, nz
      - f_dc_0, f_dc_1, f_dc_2              (SH degree-0 coefficient)
      - f_rest_0 … f_rest_44                (SH degrees 1-3, if present)
      - opacity
      - scale_0, scale_1, scale_2           (log-scale, raw)
      - rot_0, rot_1, rot_2, rot_3          (quaternion, raw)
    """
    means     = model.means.detach().cpu().numpy()          # [N, 3]
    quats     = model.quats.detach().cpu().numpy()          # [N, 4]
    scales    = model.scales.detach().cpu().numpy()         # [N, 3]
    opacities = model.opacities.detach().cpu().numpy()      # [N]
    sh_dc     = model.sh_dc.detach().cpu().numpy()          # [N, 1, 3]
    sh_rest   = model.sh_rest.detach().cpu().numpy()        # [N, K-1, 3]

    n = means.shape[0]

    f_dc  = sh_dc[:, 0, :]                          # [N, 3]
    # Interleave SH rest as (coeff_0_ch0, coeff_0_ch1, coeff_0_ch2, coeff_1_ch0, …)
    # i.e. reshape [N, K-1, 3] → [N, (K-1)*3] in C order
    f_rest = sh_rest.reshape(n, -1)                 # [N, (K-1)*3]
    n_rest = f_rest.shape[1]                        # 45 for degree-3

    normals = np.zeros((n, 3), dtype=np.float32)

    if opacities.ndim == 1:
        opacities = opacities[:, np.newaxis]        # [N, 1]

    # Stack into a single float32 block
    data = np.hstack([
        means,          # x y z            (3)
        normals,        # nx ny nz          (3)
        f_dc,           # f_dc_0..2         (3)
        f_rest,         # f_rest_0..44      (45 for deg-3)
        opacities,      # opacity           (1)
        scales,         # scale_0..2        (3)
        quats,          # rot_0..3          (4)
    ]).astype(np.float32)

    # Build PLY header
    f_rest_props = "\n".join(
        f"property float f_rest_{i}" for i in range(n_rest)
    )

    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property float nx\n"
        f"property float ny\n"
        f"property float nz\n"
        f"property float f_dc_0\n"
        f"property float f_dc_1\n"
        f"property float f_dc_2\n"
        f"{f_rest_props}\n"
        f"property float opacity\n"
        f"property float scale_0\n"
        f"property float scale_1\n"
        f"property float scale_2\n"
        f"property float rot_0\n"
        f"property float rot_1\n"
        f"property float rot_2\n"
        f"property float rot_3\n"
        f"end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
