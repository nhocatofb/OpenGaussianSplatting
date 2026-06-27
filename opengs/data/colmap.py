"""
ColmapDataset: Load standard COLMAP sparse reconstruction for 3DGS training.

Expected folder layout:
    data_dir/
    ├── images/              # input images (JPG / PNG)
    └── sparse/
        └── 0/               # COLMAP sparse model
            ├── cameras.bin  (or cameras.txt)
            ├── images.bin   (or images.txt)
            └── points3D.bin (or points3D.txt)
"""

import struct
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# COLMAP model IDs -> (name, num_params)
# ---------------------------------------------------------------------------
_CAMERA_MODELS = {
    0:  ("SIMPLE_PINHOLE", 3),     # f, cx, cy
    1:  ("PINHOLE", 4),            # fx, fy, cx, cy
    2:  ("SIMPLE_RADIAL", 4),      # f, cx, cy, k1
    3:  ("RADIAL", 5),             # f, cx, cy, k1, k2
    4:  ("OPENCV", 8),             # fx, fy, cx, cy, k1, k2, p1, p2
    5:  ("OPENCV_FISHEYE", 8),     # fx, fy, cx, cy, k1, k2, k3, k4
    6:  ("FULL_OPENCV", 12),
    7:  ("FOV", 5),
    8:  ("SIMPLE_RADIAL_FISHEYE", 4),
    9:  ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


# ---------------------------------------------------------------------------
# Binary readers
# ---------------------------------------------------------------------------

def _read_cameras_bin(path: Path) -> dict:
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id    = struct.unpack("<I", f.read(4))[0]
            model_id  = struct.unpack("<i", f.read(4))[0]
            width     = struct.unpack("<Q", f.read(8))[0]
            height    = struct.unpack("<Q", f.read(8))[0]
            model_name, n_params = _CAMERA_MODELS.get(model_id, ("UNKNOWN", 0))
            params    = np.array(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))
            cameras[cam_id] = {"model": model_name, "width": int(width),
                               "height": int(height), "params": params}
    return cameras


def _read_images_bin(path: Path) -> dict:
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec   = np.array(struct.unpack("<4d", f.read(32)))   # qw qx qy qz
            tvec   = np.array(struct.unpack("<3d", f.read(24)))
            cam_id = struct.unpack("<I", f.read(4))[0]

            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")

            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts2d * 24)  # skip x(f64) y(f64) point3D_id(i64) per point

            images[img_id] = {"qvec": qvec, "tvec": tvec,
                              "camera_id": cam_id, "name": name}
    return images


def _read_points3d_bin(path: Path) -> dict:
    points = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pt_id = struct.unpack("<Q", f.read(8))[0]
            xyz   = np.array(struct.unpack("<3d", f.read(24)))
            rgb   = np.array(struct.unpack("<3B", f.read(3)), dtype=np.float32)
            error = struct.unpack("<d", f.read(8))[0]
            n_meas = struct.unpack("<Q", f.read(8))[0]
            f.read(n_meas * 8)  # skip image_id(u32) + point2D_idx(u32) per measurement
            points[pt_id] = {"xyz": xyz, "rgb": rgb, "error": error}
    return points


# ---------------------------------------------------------------------------
# Text readers
# ---------------------------------------------------------------------------

def _read_cameras_txt(path: Path) -> dict:
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts  = line.strip().split()
            cam_id = int(parts[0])
            model  = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = np.array([float(x) for x in parts[4:]])
            cameras[cam_id] = {"model": model, "width": width,
                               "height": height, "params": params}
    return cameras


def _read_images_txt(path: Path) -> dict:
    images = {}
    with open(path, "r") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts  = lines[i].strip().split()
        img_id = int(parts[0])
        qvec   = np.array([float(x) for x in parts[1:5]])
        tvec   = np.array([float(x) for x in parts[5:8]])
        cam_id = int(parts[8])
        name   = parts[9]
        images[img_id] = {"qvec": qvec, "tvec": tvec,
                          "camera_id": cam_id, "name": name}
        i += 2  # skip the points2D line
    return images


def _read_points3d_txt(path: Path) -> dict:
    points = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            pt_id = int(parts[0])
            xyz   = np.array([float(x) for x in parts[1:4]])
            rgb   = np.array([float(x) for x in parts[4:7]])
            error = float(parts[7])
            points[pt_id] = {"xyz": xyz, "rgb": rgb, "error": error}
    return points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float32)


def _get_K(camera: dict, scale: float = 1.0) -> np.ndarray:
    """Build 3×3 intrinsic matrix from a COLMAP camera entry."""
    model  = camera["model"]
    params = camera["params"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                 "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
        fx = fy = params[0]
        cx, cy  = params[1], params[2]
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE",
                   "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
        fx, fy  = params[0], params[1]
        cx, cy  = params[2], params[3]
    else:
        fx = fy = params[0]
        cx, cy  = camera["width"] / 2.0, camera["height"] / 2.0

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    K[0] *= scale   # scale fx, cx
    K[1] *= scale   # scale fy, cy
    return K


def _has_model(p: Path) -> bool:
    """Return True if p contains a valid COLMAP cameras file."""
    return (p / "cameras.bin").exists() or (p / "cameras.txt").exists()


def _locate_sparse(data_dir: Path) -> Path:
    """
    Locate the COLMAP sparse model directory.

    Search order:
      1. sparse/0              ← default COLMAP output
      2. Any numbered subdir   ← sparse/1, sparse/2, … (pick the first found)
      3. sparse/               ← cameras placed directly in sparse/
      4. data_dir/             ← cameras placed at the dataset root
    """
    sparse_root = data_dir / "sparse"

    # 1. Canonical sparse/0
    if sparse_root.is_dir():
        if _has_model(sparse_root / "0"):
            return sparse_root / "0"

        # 2. Any other numbered / named subdirectory, sorted for reproducibility
        sub_dirs = sorted(
            [d for d in sparse_root.iterdir() if d.is_dir()],
            key=lambda d: d.name
        )
        for sub in sub_dirs:
            if _has_model(sub):
                return sub

        # 3. Cameras directly inside sparse/
        if _has_model(sparse_root):
            return sparse_root

    # 4. Cameras at the dataset root
    if _has_model(data_dir):
        return data_dir

    raise FileNotFoundError(
        f"Cannot find a COLMAP sparse model under '{data_dir}'.\n"
        f"Tried: sparse/0/, sparse/<subdir>/, sparse/, and the root.\n"
        f"Expected files: cameras.bin or cameras.txt"
    )


# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------

class ColmapDataset:
    """
    Loads a COLMAP sparse reconstruction and prepares tensors for 3DGS training.

    Args:
        data_dir:    Root directory of the dataset (contains images/ and sparse/).
        image_scale: Downscale factor applied to images and intrinsics (e.g. 0.5 = half-res).
        device:      PyTorch device string ("cuda" or "cpu").
    """

    def __init__(self, data_dir: str, image_scale: float = 1.0, device: str = "cuda"):
        self.data_dir    = Path(data_dir)
        self.image_scale = image_scale
        self.device      = device

        sparse_dir = _locate_sparse(self.data_dir)

        # Load binary first, fall back to text
        if (sparse_dir / "cameras.bin").exists():
            self._cameras    = _read_cameras_bin(sparse_dir / "cameras.bin")
            self._images_meta = _read_images_bin(sparse_dir / "images.bin")
            self._points3d   = _read_points3d_bin(sparse_dir / "points3D.bin")
            fmt = "binary"
        else:
            self._cameras    = _read_cameras_txt(sparse_dir / "cameras.txt")
            self._images_meta = _read_images_txt(sparse_dir / "images.txt")
            self._points3d   = _read_points3d_txt(sparse_dir / "points3D.txt")
            fmt = "text"

        print(f"[ColmapDataset] Loaded COLMAP model ({fmt}) from {sparse_dir}")

        # Find image folder
        img_dir = self.data_dir / "images"
        if not img_dir.exists():
            img_dir = self.data_dir
        self._img_dir = img_dir

        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        gt_images, viewmats, Ks = [], [], []
        widths, heights = [], []

        for img_id in sorted(self._images_meta.keys()):
            meta = self._images_meta[img_id]
            cam  = self._cameras[meta["camera_id"]]

            # Locate image file
            img_path = self._img_dir / meta["name"]
            if not img_path.exists():
                img_path = self._img_dir / Path(meta["name"]).name
            if not img_path.exists():
                print(f"  [WARN] Image not found: {meta['name']} — skipping")
                continue

            # Load + resize
            w0, h0 = cam["width"], cam["height"]
            w = max(1, round(w0 * self.image_scale))
            h = max(1, round(h0 * self.image_scale))
            pil_img = Image.open(img_path).convert("RGB").resize((w, h), Image.LANCZOS)
            # Keep images on CPU RAM — pushed to GPU one-at-a-time during training.
            # This avoids pre-allocating all images in VRAM (e.g. 128 × 768×576 ≈ 680 MB).
            img_t   = torch.from_numpy(
                np.array(pil_img, dtype=np.float32) / 255.0
            )   # [H, W, 3]  — CPU tensor

            # Viewmat and K stay on GPU (small, ~1 KB each)
            R = _qvec2rotmat(meta["qvec"])
            t = meta["tvec"].astype(np.float32)
            vm = np.eye(4, dtype=np.float32)
            vm[:3, :3] = R
            vm[:3, 3]  = t
            vm_t = torch.from_numpy(vm).to(self.device)

            K  = _get_K(cam, scale=self.image_scale)
            K_t = torch.from_numpy(K).to(self.device)

            gt_images.append(img_t)
            viewmats.append(vm_t)
            Ks.append(K_t)
            widths.append(w)
            heights.append(h)

        if len(gt_images) == 0:
            raise RuntimeError("No images loaded — check that images/ folder exists "
                               "and names match the COLMAP model.")

        self.gt_images = gt_images                                    # list[CPU Tensor H W 3]
        self.viewmats  = torch.stack(viewmats, dim=0)                 # [N, 4, 4]
        self.Ks        = torch.stack(Ks, dim=0)                       # [N, 3, 3]
        self.widths    = widths                                        # list[int]
        self.heights   = heights                                       # list[int]

        # Assume all images share the same resolution (common case)
        self.width  = widths[0]
        self.height = heights[0]

        print(f"[ColmapDataset] {len(gt_images)} images loaded  "
              f"({self.width}×{self.height})  |  "
              f"{len(self._points3d):,} sparse 3D points")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.gt_images)

    # ------------------------------------------------------------------
    def get_point_cloud(self):
        """
        Return (xyz, rgb) Float32 tensors from the COLMAP sparse points.
        xyz/rgb are placed on self.device (GPU) because they are used
        directly to initialise model parameters.
        """
        pts = list(self._points3d.values())
        xyz = torch.tensor(
            np.stack([p["xyz"] for p in pts], axis=0), dtype=torch.float32
        ).to(self.device)
        rgb = torch.tensor(
            np.stack([p["rgb"] for p in pts], axis=0), dtype=torch.float32
        ).to(self.device) / 255.0
        return xyz, rgb

    # ------------------------------------------------------------------
    def get_scene_scale(self) -> float:
        """
        Estimate scene extent as the radius of the bounding sphere of camera centres.
        Used to set the scene_scale for the DefaultStrategy.
        """
        cam_positions = []
        for vm in self.viewmats:
            R = vm[:3, :3]
            t = vm[:3, 3]
            pos = -(R.T @ t)          # world-space camera centre
            cam_positions.append(pos)
        centres = torch.stack(cam_positions)           # [N, 3]
        scene_centre = centres.mean(dim=0)
        radius = (centres - scene_centre).norm(dim=-1).max().item()
        return max(radius, 1.0) * 1.1                 # slight padding; never <1
