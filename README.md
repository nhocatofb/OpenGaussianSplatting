# OpenGaussianSplatting

A modular, research-friendly implementation of [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) built on top of [gsplat](https://github.com/nerfstudio-project/gsplat).

The codebase is intentionally small and readable — every component (model, renderer, loss, optimizer, data loader) lives in its own file and can be swapped or extended without touching the rest of the pipeline.

---

## Features

- Clean modular architecture — models, renderers, losses, and optimizers are fully decoupled
- Full 3DGS training pipeline on real COLMAP data (`apps/train_3d.py`)
- Progressive Spherical Harmonics activation (degree 0 → max_sh_degree during training)
- Auto-configured hyperparameters based on scene scale, resolution, and available VRAM
- PLY export compatible with common 3DGS viewers (SuperSplat, Gaussian Splatting Viewer)
- Apache 2.0 licensed — commercial use permitted

---

## Requirements

| Component | Version |
|-----------|---------|
| CUDA Driver | ≥ 12.1 — verify with `nvidia-smi` |
| PyTorch | 2.2.1+cu121 |
| gsplat | 1.5.2 |
| NumPy | 1.26.4 (must be < 2.0) |

> **Python version:** The gsplat pre-built wheels target **Python 3.10**.
> Python 3.11/3.12 users can still install, but gsplat will compile from source
> (Linux only — requires the CUDA toolkit headers).

---

## Installation

### Windows (Python 3.10 required)

```bash
# 1. Create a conda environment with Python 3.10
conda create -n opengs python=3.10 -y
conda activate opengs

# 2. Clone the repository
git clone https://github.com/nhocatofb/OpenGaussianSplatting.git
cd OpenGaussianSplatting

# 3. Install dependencies using the Windows-specific requirements file
pip install -r requirements/windows.txt

# 4. Install opengs in editable mode so local edits take effect immediately
pip install -e .
```

### Linux / Ubuntu (Python 3.10–3.12)

```bash
# 1. Create a conda environment (3.10 recommended for pre-built gsplat wheel)
conda create -n opengs python=3.10 -y
conda activate opengs

# 2. Clone the repository
git clone https://github.com/nhocatofb/OpenGaussianSplatting.git
cd OpenGaussianSplatting

# 3. Install dependencies using the Linux requirements file
pip install -r requirements/linux.txt

# 4. Install opengs in editable mode
pip install -e .
```

> On Python 3.11/3.12, gsplat compiles from source automatically. This takes a few
> extra minutes and requires the CUDA toolkit:
> `sudo apt install cuda-toolkit-12-1`

### Kaggle

Kaggle ships Python 3.12, PyTorch, and NumPy 2.x pre-installed system-wide.
**Do not reinstall torch or downgrade NumPy** — it will break the Kaggle environment.

```bash
# Step 1 — Clone the repo into your Kaggle working directory
git clone https://github.com/nhocatofb/OpenGaussianSplatting.git
cd OpenGaussianSplatting

# Step 2 — Install only the missing utilities (torch and numpy are already present)
pip install ninja jaxtyping rich Pillow

# Step 3 — Compile gsplat from source against Kaggle's existing PyTorch (~3–5 min)
#   Do NOT pin a version here — the pre-built gsplat==1.5.2 wheel targets PyTorch 2.2.1
#   and will cause ABI errors on Kaggle's newer PyTorch.
pip install gsplat

# Step 4 — Install opengs
pip install -e .
```

### Google Colab

Colab also ships PyTorch and NumPy 2.x pre-installed. Same rules apply — do not
touch torch or numpy. If Colab's pre-installed torch is outdated, restart the runtime
after installing PyTorch from the official index rather than touching numpy.

```bash
# Step 1 — Clone the repo
git clone https://github.com/nhocatofb/OpenGaussianSplatting.git
%cd OpenGaussianSplatting

# Step 2 — Install missing utilities
!pip install ninja jaxtyping rich Pillow

# Step 3 — Compile gsplat from source
!pip install gsplat

# Step 4 — Install opengs
!pip install -e .
```

> Colab sessions are ephemeral — re-run these cells after each runtime restart.

---

## Quick Sanity Test

Run this after installation to verify the full render pipeline works before
training on real data:

```bash
python - <<'EOF'
import torch
from opengs.models import VanillaGaussian
from opengs.renderers import GsplatRenderer

model    = VanillaGaussian(num_points=500).cuda()
renderer = GsplatRenderer()

viewmat = torch.eye(4, device="cuda").unsqueeze(0)
K = torch.tensor([[[200., 0., 128.],
                   [0., 200., 128.],
                   [0.,   0.,   1.]]], device="cuda")

img, _ = renderer.render(model, viewmat, K, width=256, height=256)
print(f"Render OK — output shape: {img.shape}, dtype: {img.dtype}")
EOF
```

Expected output:
```
Render OK — output shape: torch.Size([256, 256, 3]), dtype: torch.float32
```

---

## Usage

### Train on your own COLMAP scene

```
your_scene/
├── images/          ← input photos (JPG / PNG)
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

```bash
# Basic run — hyperparameters are auto-configured from the scene
python apps/train_3d.py --data_dir path/to/your_scene

# Full options
python apps/train_3d.py \
    --data_dir   path/to/your_scene \
    --output_dir path/to/output \
    --max_steps  30000 \
    --image_scale 0.5 \
    --sh_degree  0
```

Training outputs:
- `output/previews/step_XXXXX.png` — preview renders every N steps
- `output/checkpoint_XXXXX.ply` — intermediate checkpoints
- `output/point_cloud.ply` — final model (open in [SuperSplat](https://playcanvas.com/supersplat/editor))

### Use the Python API directly

```python
from opengs.data import ColmapDataset
from opengs.models import VanillaGaussian
from opengs.renderers import GsplatRenderer

dataset  = ColmapDataset("path/to/scene", image_scale=0.5, device="cuda")
xyz, rgb = dataset.get_point_cloud()
model    = VanillaGaussian.from_pointcloud(xyz, rgb, max_sh_degree=0)
renderer = GsplatRenderer(rasterize_mode="antialiased")

img, info = renderer.render(
    model,
    viewmat=dataset.viewmats[0:1],
    K=dataset.Ks[0:1],
    width=dataset.widths[0],
    height=dataset.heights[0],
    active_sh_degree=0,
)
```

---

## Project Structure

```
OpenGaussianSplatting/
├── apps/
│   └── train_3d.py          # End-to-end COLMAP training script
├── opengs/
│   ├── models/
│   │   └── vanilla_gs.py    # VanillaGaussian — Gaussian parameters + SH color
│   ├── renderers/
│   │   └── gsplat_renderer.py  # GsplatRenderer — thin wrapper around gsplat
│   ├── losses/
│   │   └── image_loss.py    # GSImageLoss — L1 + DSSIM
│   ├── optimizers/
│   │   └── densifier.py     # Adam optimizers + DefaultStrategy (split/clone/prune)
│   ├── data/
│   │   ├── colmap.py        # ColmapDataset — binary/text COLMAP reader
│   │   └── synthetic.py     # Synthetic circular camera rig for testing
│   └── utils/
│       ├── auto_config.py   # Auto-tune hyperparameters from dataset properties
│       └── ply_parser.py    # PLY export (INRIA-compatible format)
├── requirements.txt
├── pyproject.toml
├── LICENSE                  # Apache 2.0
└── NOTICE                   # Third-party attributions
```

---

## Extending the Codebase

The architecture is designed so that each component can be replaced independently:

| Want to change | File to edit or replace |
|----------------|------------------------|
| Gaussian representation | `opengs/models/vanilla_gs.py` |
| Rasterizer backend | `opengs/renderers/gsplat_renderer.py` |
| Loss function | `opengs/losses/image_loss.py` |
| Densification strategy | `opengs/optimizers/densifier.py` |
| Data source | `opengs/data/colmap.py` or add a new loader |
| Hyperparameter logic | `opengs/utils/auto_config.py` |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide: how to add a new model,
renderer, loss, data loader, or optimizer, plug it into the pipeline, and coding
conventions for pull requests.

---

## Citation

If you use this project in your research, please also cite the original 3DGS paper:

```bibtex
@article{kerbl3Dgaussians,
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal   = {ACM Transactions on Graphics},
  year      = {2023},
  volume    = {42},
  number    = {4},
  url       = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
Third-party attributions are listed in [NOTICE](NOTICE).
