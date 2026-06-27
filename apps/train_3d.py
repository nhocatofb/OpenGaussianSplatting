"""
Standard 3DGS training pipeline on real COLMAP data.

Usage:
    python apps/train_3d.py --data_dir D:/my_scene
    python apps/train_3d.py --data_dir D:/my_scene --max_steps 30000 --image_scale 0.5
    python apps/train_3d.py --data_dir D:/my_scene --output_dir D:/output/my_scene

Data layout expected inside --data_dir:
    data_dir/
    ├── images/         ← input photos (JPG / PNG)
    └── sparse/
        └── 0/          ← COLMAP sparse reconstruction
            ├── cameras.bin   (or .txt)
            ├── images.bin    (or .txt)
            └── points3D.bin  (or .txt)
"""

import os
import sys
import math
import argparse
from pathlib import Path

import torch
import torchvision

from opengs.data import ColmapDataset
from opengs.models import VanillaGaussian
from opengs.renderers import GsplatRenderer
from opengs.losses import GSImageLoss
from opengs.optimizers import create_gs_optimizers, get_default_strategy
from opengs.utils import save_ply, auto_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="3DGS training on COLMAP data")
    p.add_argument("--data_dir",    required=True,
                   help="Root directory of the COLMAP dataset")
    p.add_argument("--output_dir",  default=None,
                   help="Where to write results (default: <data_dir>/output)")
    p.add_argument("--max_steps",   type=int, default=None,
                   help="Total training iterations. Default: auto (30000, or 15000 for small scenes).")
    p.add_argument("--image_scale", type=float, default=None,
                   help="Downscale factor for images (e.g. 0.5 = half res). "
                        "Default: auto-selected based on resolution and free VRAM.")
    p.add_argument("--save_every",  type=int, default=5_000,
                   help="Save a checkpoint .ply every N steps")
    p.add_argument("--log_every",   type=int, default=500,
                   help="Print loss and save a preview image every N steps")
    p.add_argument("--lambda_dssim", type=float, default=0.2,
                   help="Weight of DSSIM loss component (default: 0.2)")
    p.add_argument("--sh_degree",   type=int, default=0, choices=[0, 1, 2, 3],
                   help="Max SH degree for view-dependent color (0–3, default: 0). "
                        "Lower = fewer parameters, faster training, less view-dependence. "
                        "0=RGB-only (3 floats/Gaussian), 3=full SH (48 floats/Gaussian).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Learning-rate scheduler (exponential decay, mirrors original 3DGS paper)
# ---------------------------------------------------------------------------

def _exp_lr_lambda(step: int, lr_init: float, lr_final: float,
                   lr_delay_steps: int, lr_delay_mult: float,
                   max_steps: int):
    """Returns a multiplicative factor for the base lr."""
    if step < lr_delay_steps:
        delay = lr_delay_mult + (1 - lr_delay_mult) * math.sin(
            0.5 * math.pi * step / lr_delay_steps)
    else:
        delay = 1.0
    t = max(0.0, min(1.0, (step - lr_delay_steps) / max(1, max_steps - lr_delay_steps)))
    log_lerp = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
    return delay * log_lerp / lr_init   # factor relative to base lr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load COLMAP dataset
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("  3DGS Training  —  COLMAP pipeline")
    print("=" * 60)
    print(f"Data dir  : {args.data_dir}")
    print(f"Output    : {output_dir}")
    print()

    # ---- Load dataset at native resolution first for auto-config ----------
    dataset = ColmapDataset(
        data_dir=args.data_dir,
        image_scale=1.0,          # probe at native res; auto_config may reduce it
        device="cuda",
    )

    # ---- Auto-configure hyperparameters from dataset properties -----------
    cfg = auto_config(
        dataset,
        user_image_scale=args.image_scale,
        user_max_steps=args.max_steps,
    )
    print()
    print(cfg.summary())
    print()

    # ---- Reload at the chosen scale if different from 1.0 -----------------
    if cfg.image_scale != 1.0:
        dataset = ColmapDataset(
            data_dir=args.data_dir,
            image_scale=cfg.image_scale,
            device="cuda",
        )

    num_views   = len(dataset)
    width       = cfg.width
    height      = cfg.height
    scene_scale = cfg.scene_scale

    # ------------------------------------------------------------------ #
    # 2. Initialize Gaussian model from COLMAP sparse point cloud
    # ------------------------------------------------------------------ #
    print("Initializing Gaussians from COLMAP sparse points...")
    xyz, rgb = dataset.get_point_cloud()
    model = VanillaGaussian.from_pointcloud(xyz, rgb, max_sh_degree=args.sh_degree)
    print(f"  → {model.num_points:,} initial Gaussians")

    renderer = GsplatRenderer(rasterize_mode="antialiased")
    loss_fn  = GSImageLoss(lambda_dssim=args.lambda_dssim).cuda()

    # ------------------------------------------------------------------ #
    # 3. Optimizers + densification strategy  (values from auto_config)
    # ------------------------------------------------------------------ #
    optimizers = create_gs_optimizers(
        model.params,
        means_lr=cfg.means_lr_init,
    )

    strategy = get_default_strategy(
        refine_start_iter=cfg.refine_start_iter,
        refine_stop_iter=cfg.refine_stop_iter,
        refine_every=cfg.refine_every,
        reset_every=cfg.reset_every,
    )
    strategy_state = strategy.initialize_state(scene_scale=cfg.scene_scale)

    # Exponential LR scheduler for positions (means)
    means_optimizer = optimizers["means"]
    def update_means_lr(step: int):
        factor = _exp_lr_lambda(
            step=step,
            lr_init=cfg.means_lr_init,
            lr_final=cfg.means_lr_final,
            lr_delay_steps=100,
            lr_delay_mult=0.01,
            max_steps=cfg.max_steps,
        )
        for pg in means_optimizer.param_groups:
            pg["lr"] = cfg.means_lr_init * factor

    # ------------------------------------------------------------------ #
    # 4. Training loop
    # ------------------------------------------------------------------ #
    print("\nStarting training...\n")

    # Shuffled round-robin queue:
    # - Every camera is visited exactly once per "epoch" before any repeats.
    # - Order is reshuffled each epoch so training doesn't become cyclic.
    # - .pop() from the end is O(1) and avoids shifting the list.
    cam_queue: list = []

    for step in range(cfg.max_steps + 1):
        if not cam_queue:
            cam_queue = torch.randperm(num_views).tolist()   # new shuffled epoch

        cam_idx = cam_queue.pop()
        # Images stay on CPU RAM; push only the current one to GPU
        target_img = dataset.gt_images[cam_idx].cuda()              # [H, W, 3]
        viewmat    = dataset.viewmats[cam_idx:cam_idx + 1]           # [1, 4, 4]  GPU
        K          = dataset.Ks[cam_idx:cam_idx + 1]                 # [1, 3, 3]  GPU
        w          = dataset.widths[cam_idx]
        h          = dataset.heights[cam_idx]

        # ---- Progressive SH activation (increase degree every 1000 steps) ----
        active_sh = min(step // 1_000, model.max_sh_degree)

        # A. Render with current active SH degree
        rendered, info = renderer.render(
            gaussian_model=model,
            viewmat=viewmat,
            K=K,
            width=w,
            height=h,
            active_sh_degree=active_sh,
        )

        # B. Pre-backward hook
        strategy.step_pre_backward(model.params, optimizers, strategy_state, step, info)

        # C. Loss — resize target only if resolution mismatch
        if rendered.shape[:2] != target_img.shape[:2]:
            target_img = torch.nn.functional.interpolate(
                target_img.permute(2, 0, 1).unsqueeze(0),
                size=rendered.shape[:2], mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        total_loss, l1, dssim = loss_fn(rendered, target_img)

        # D. Backward
        total_loss.backward()

        # E. Post-backward: densification / pruning
        info["width"]  = w
        info["height"] = h
        strategy.step_post_backward(model.params, optimizers, strategy_state, step, info)

        # F. Optimizer step + zero_grad
        update_means_lr(step)
        for opt in optimizers.values():
            opt.step()
        for opt in optimizers.values():
            opt.zero_grad()

        # ---- Logging ----
        if step % cfg.log_every == 0:
            n_gaussians = model.means.shape[0]
            print(f"Step {step:05d}/{cfg.max_steps}  |  "
                  f"Loss: {total_loss.item():.4f}  "
                  f"(L1={l1.item():.4f}  DSSIM={dssim.item():.4f})  |  "
                  f"Gaussians: {n_gaussians:,}  SH degree: {active_sh}")

            # Save preview from first camera (with full active SH degree)
            with torch.no_grad():
                preview, _ = renderer.render(
                    gaussian_model=model,
                    viewmat=dataset.viewmats[0:1],
                    K=dataset.Ks[0:1],
                    width=dataset.widths[0],
                    height=dataset.heights[0],
                    active_sh_degree=active_sh,
                )
            preview_path = preview_dir / f"step_{step:05d}.png"
            torchvision.utils.save_image(preview.permute(2, 0, 1), str(preview_path))

        # ---- Checkpoint ----
        if step > 0 and step % cfg.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_{step:05d}.ply"
            save_ply(model, str(ckpt_path))
            print(f"  [checkpoint] saved → {ckpt_path}")

    # ------------------------------------------------------------------ #
    # 5. Save final model
    # ------------------------------------------------------------------ #
    final_path = output_dir / "point_cloud.ply"
    save_ply(model, str(final_path))

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Final model  : {final_path}")
    print(f"  Previews     : {preview_dir}")
    print(f"  Gaussians    : {model.means.shape[0]:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
