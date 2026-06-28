# Contributing to OpenGaussianSplatting

This guide explains how the codebase is structured and how to add new components — a new model, renderer, loss function, data loader, or optimizer — and plug them into the training pipeline with minimal friction.

---

## Architecture Overview

Every component lives in its own file under `opengs/` and is completely decoupled from the others. Components communicate only through well-defined tensor interfaces, so you can swap any one of them without touching the rest.

```
opengs/
├── models/       ← Gaussian parameter containers
├── renderers/    ← Rasterization wrappers
├── losses/       ← Image-space loss functions
├── optimizers/   ← Adam groups + densification strategy
├── data/         ← Dataset loaders
└── utils/        ← Auto-config, PLY export
```

The training script (`apps/train_3d.py`) is the only place where components are assembled together. Think of it as the "wiring layer" — it imports from `opengs/` but contains no core logic itself.

---

## 1. Adding a New Gaussian Model

A Gaussian model is any `nn.Module` that exposes a `params: nn.ParameterDict` with the keys the renderer expects. The minimum contract is:

| Attribute / Property | Type | Description |
|---|---|---|
| `params` | `nn.ParameterDict` | Dict of learnable tensors keyed by name |
| `means` | `[N, 3]` | 3D positions |
| `get_normalized_quats()` | `[N, 4]` | Unit quaternions |
| `get_real_scales()` | `[N, 3]` | Positive scales (after `exp`) |
| `get_real_opacities()` | `[N]` | Opacities in (0, 1) (after `sigmoid`) |
| `colors` | `[N, K, 3]` | SH coefficients (K=1 for degree-0) |
| `max_sh_degree` | `int` | Maximum SH degree stored |

### Step-by-step

**1. Create the file**

```
opengs/models/my_model.py
```

```python
import torch
import torch.nn as nn

class MyGaussian(nn.Module):
    def __init__(self, num_points: int = 100, max_sh_degree: int = 0):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        K = (max_sh_degree + 1) ** 2

        self.params = nn.ParameterDict({
            "means":     nn.Parameter(torch.randn(num_points, 3)),
            "quats":     nn.Parameter(torch.randn(num_points, 4)),
            "scales":    nn.Parameter(torch.full((num_points, 3), -2.0)),
            "opacities": nn.Parameter(torch.full((num_points,), -2.2)),
            "sh_dc":     nn.Parameter(torch.zeros(num_points, 1, 3)),
            "sh_rest":   nn.Parameter(torch.zeros(num_points, K - 1, 3)),
        })

    @property
    def means(self): return self.params["means"]
    @property
    def colors(self):
        return torch.cat([self.params["sh_dc"], self.params["sh_rest"]], dim=1)

    def get_normalized_quats(self):
        return self.params["quats"] / self.params["quats"].norm(dim=-1, keepdim=True)

    def get_real_scales(self):    return torch.exp(self.params["scales"])
    def get_real_opacities(self): return torch.sigmoid(self.params["opacities"])
```

**2. Export it from the package**

In `opengs/models/__init__.py`:
```python
from .vanilla_gs import VanillaGaussian
from .my_model import MyGaussian          # add this line
```

**3. Use it in a training script**

```python
from opengs.models import MyGaussian
model = MyGaussian(num_points=10_000, max_sh_degree=0).cuda()
```

The rest of the pipeline (renderer, optimizer, strategy) works without any other changes because they only interact with the shared interface above.

---

## 2. Adding a New Renderer

A renderer is a plain Python class (no `nn.Module` required) with a single `render()` method that returns `(image [H, W, 3], info dict)`.

### Minimum interface

```python
class MyRenderer:
    def render(self, gaussian_model, viewmat, K, width, height,
               active_sh_degree=0) -> tuple[torch.Tensor, dict]:
        ...
        # info must contain at least the keys gsplat's DefaultStrategy needs:
        # "means2d", "width", "height" (set width/height after the call in train loop)
        return rendered_image, info
```

### Step-by-step

**1. Create the file**

```
opengs/renderers/my_renderer.py
```

**2. Export it**

In `opengs/renderers/__init__.py`:
```python
from .gsplat_renderer import GsplatRenderer
from .my_renderer import MyRenderer        # add this line
```

**3. Swap in the training script**

```python
from opengs.renderers import MyRenderer
renderer = MyRenderer()
```

> **Note on `info`:** If you use a custom renderer that does not produce the keys that `DefaultStrategy` needs (e.g. `"means2d"`), you will also need to switch to a custom densification strategy (see §5).

---

## 3. Adding a New Loss Function

Loss functions are `nn.Module` subclasses. Their `forward()` should return a tuple of `(total_loss, *component_losses)` — the training loop unpacks the first element for `.backward()` and logs the rest.

### Step-by-step

**1. Create the file**

```
opengs/losses/my_loss.py
```

```python
import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        l1   = (pred - gt).abs().mean()
        l2   = (pred - gt).pow(2).mean()
        total = (1 - self.alpha) * l1 + self.alpha * l2
        return total, l1, l2
```

**2. Export it**

In `opengs/losses/__init__.py`:
```python
from .image_loss import l1_loss, ssim, GSImageLoss
from .my_loss import MyLoss                # add this line
```

**3. Swap in the training script**

```python
from opengs.losses import MyLoss
loss_fn = MyLoss(alpha=0.3).cuda()

# In the training loop:
total_loss, l1, l2 = loss_fn(rendered, target_img)
```

---

## 4. Adding a New Data Loader

A dataset class only needs to populate a handful of attributes after `__init__`. There is no base class to inherit from.

### Required attributes

| Attribute | Type | Description |
|---|---|---|
| `gt_images` | `list[Tensor H×W×3]` | Ground-truth images, kept on **CPU** |
| `viewmats` | `[N, 4, 4]` GPU tensor | World-to-camera matrices |
| `Ks` | `[N, 3, 3]` GPU tensor | Intrinsic matrices |
| `widths` | `list[int]` | Per-image widths |
| `heights` | `list[int]` | Per-image heights |
| `width` / `height` | `int` | Representative resolution |

And one method:
```python
def get_point_cloud(self) -> tuple[Tensor, Tensor]:
    """Return (xyz [N,3], rgb [N,3]) on GPU for model initialization."""
```

### Step-by-step

**1. Create the file**

```
opengs/data/my_dataset.py
```

**2. Export it**

In `opengs/data/__init__.py`:
```python
from .colmap import ColmapDataset
from .my_dataset import MyDataset          # add this line
```

**3. Use it in a training script**

```python
from opengs.data import MyDataset
dataset = MyDataset("path/to/data", device="cuda")
xyz, rgb = dataset.get_point_cloud()
```

---

## 5. Adding a New Optimizer or Densification Strategy

Optimizers are created by a factory function that returns `dict[str, torch.optim.Optimizer]` — one entry per parameter group. The strategy object must implement the gsplat `Strategy` protocol.

### Custom learning rates

Edit or copy `opengs/optimizers/densifier.py`:

```python
from opengs.optimizers.densifier import create_gs_optimizers

optimizers = create_gs_optimizers(
    model.params,
    means_lr=1e-4,
    sh_dc_lr=1e-3,
    sh_rest_lr=5e-5,
)
```

### Custom densification strategy

Implement a class that matches the gsplat `Strategy` protocol:

```python
class MyStrategy:
    def initialize_state(self, scene_scale: float) -> dict:
        return {}

    def step_pre_backward(self, params, optimizers, state, step, info):
        pass  # e.g. register gradient hooks

    def step_post_backward(self, params, optimizers, state, step, info):
        pass  # e.g. split, clone, prune Gaussians
```

Then in the training script:
```python
strategy = MyStrategy()
strategy_state = strategy.initialize_state(scene_scale=cfg.scene_scale)
```

---

## 6. Writing a New Training Script

The fastest way is to copy `apps/train_3d.py` and swap components:

```python
# apps/my_experiment.py
from opengs.data    import MyDataset
from opengs.models  import MyGaussian
from opengs.renderers import MyRenderer
from opengs.losses  import MyLoss
from opengs.optimizers import create_gs_optimizers, get_default_strategy
from opengs.utils   import save_ply, auto_config

dataset  = MyDataset(...)
xyz, rgb = dataset.get_point_cloud()
model    = MyGaussian.from_pointcloud(xyz, rgb, max_sh_degree=0).cuda()
renderer = MyRenderer()
loss_fn  = MyLoss().cuda()
optimizers = create_gs_optimizers(model.params)
strategy   = get_default_strategy(...)
```

The training loop skeleton from `apps/train_3d.py` works without modification once all four objects above are in place.

---

## 7. Coding Conventions

- **One class / one file.** Each module file should contain exactly one primary class or one cohesive group of related functions.
- **No cross-component imports.** `models/` must not import from `renderers/`, `losses/`, etc. Only the training script (in `apps/`) is allowed to wire components together.
- **Tensors in, tensors out.** Public methods should accept and return plain PyTorch tensors. Avoid leaking internal implementation details (e.g. gsplat-specific objects) through public APIs.
- **Type annotations.** All public function signatures should have type annotations.
- **No silent GPU allocation in `__init__`.** Keep large tensors (images, point clouds) on CPU until they are needed for computation.

---

## 8. Pull Request Checklist

- [ ] New file placed in the correct `opengs/<module>/` subfolder
- [ ] Class or function exported from the corresponding `__init__.py`
- [ ] Public methods have type annotations and a one-line docstring
- [ ] No imports across component boundaries (model ↔ renderer ↔ loss, etc.)
- [ ] Tested end-to-end with at least the quick sanity test from the README
