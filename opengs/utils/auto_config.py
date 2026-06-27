"""
auto_config: inspect a ColmapDataset and derive sensible training hyperparameters.

All thresholds are documented inline so they are easy to tune later.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opengs.data.colmap import ColmapDataset


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # ---- resolution --------------------------------------------------------
    image_scale: float = 1.0          # applied to images + K
    width: int = 0
    height: int = 0

    # ---- core training -----------------------------------------------------
    max_steps: int = 30_000
    means_lr_init: float = 1.6e-4     # will be scaled by scene_scale
    means_lr_final: float = 1.6e-6    # will be scaled by scene_scale

    # ---- densification strategy --------------------------------------------
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3_000
    scene_scale: float = 1.0

    # ---- logging / checkpointing -------------------------------------------
    log_every: int = 500
    save_every: int = 5_000

    # ---- loss --------------------------------------------------------------
    lambda_dssim: float = 0.2

    # ---- human-readable notes from the auto-config logic -------------------
    notes: list[str] = field(default_factory=list)

    def _note(self, msg: str) -> None:
        self.notes.append(msg)

    def summary(self) -> str:
        lines = ["[AutoConfig] Training configuration:"]
        lines.append(f"  Resolution     : {self.width}×{self.height}"
                     f"  (image_scale={self.image_scale})")
        lines.append(f"  Steps          : {self.max_steps}")
        lines.append(f"  Scene scale    : {self.scene_scale:.3f}")
        lines.append(f"  Means LR       : {self.means_lr_init:.2e} → {self.means_lr_final:.2e}")
        lines.append(f"  Densify        : {self.refine_start_iter}–{self.refine_stop_iter}"
                     f"  every {self.refine_every} steps")
        lines.append(f"  Opacity reset  : every {self.reset_every} steps")
        lines.append(f"  Log/save every : {self.log_every} / {self.save_every} steps")
        if self.notes:
            lines.append("  Notes:")
            for n in self.notes:
                lines.append(f"    • {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# VRAM probe
# ---------------------------------------------------------------------------

def _free_vram_gb() -> float:
    """Return free GPU VRAM in GB, or a large number if CUDA is not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 999.0
        free, total = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:
        return 999.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def auto_config(dataset: "ColmapDataset",
                user_image_scale: float | None = None,
                user_max_steps: int | None = None) -> TrainConfig:
    """
    Inspect *dataset* and return a :class:`TrainConfig` with tuned defaults.

    Args:
        dataset:          An already-loaded ColmapDataset.
        user_image_scale: If the user explicitly set --image_scale, pass it here
                          so auto-config respects that choice (only warns, no override).
        user_max_steps:   Same for --max_steps.
    """
    cfg = TrainConfig()

    num_views    = len(dataset)
    raw_w        = dataset.width          # resolution BEFORE any scale override
    raw_h        = dataset.height
    num_points   = len(dataset._points3d)
    scene_scale  = dataset.get_scene_scale()
    free_vram    = _free_vram_gb()

    cfg.scene_scale     = scene_scale
    cfg.means_lr_init   = 1.6e-4 * scene_scale
    cfg.means_lr_final  = 1.6e-6 * scene_scale

    # ------------------------------------------------------------------ #
    # 1. Resolution / image_scale
    # ------------------------------------------------------------------ #
    raw_mpx = (raw_w * raw_h) / 1_000_000

    if user_image_scale is not None:
        # User set it explicitly — honour it, but warn if it seems risky
        cfg.image_scale = user_image_scale
        eff_mpx = raw_mpx * user_image_scale ** 2
        if eff_mpx > 2.0 and free_vram < 6.0:
            cfg._note(
                f"Effective resolution {raw_w*user_image_scale:.0f}×"
                f"{raw_h*user_image_scale:.0f} ({eff_mpx:.1f} Mpx) may stress "
                f"{free_vram:.1f} GB free VRAM. Consider --image_scale 0.25."
            )
    else:
        # Auto-select scale based on VRAM and raw resolution
        # Target: keep the render tile ≤ ~1 Mpx on ≤4 GB cards
        if free_vram >= 12.0:
            target_mpx = 4.0
        elif free_vram >= 8.0:
            target_mpx = 2.0
        elif free_vram >= 6.0:
            target_mpx = 1.0
        else:                    # ≤ 4 GB
            target_mpx = 0.5

        if raw_mpx <= target_mpx:
            cfg.image_scale = 1.0
        else:
            # Pick the largest standard scale that fits
            for scale in [1.0, 0.5, 0.25, 0.125]:
                if raw_mpx * scale ** 2 <= target_mpx:
                    cfg.image_scale = scale
                    break
            else:
                cfg.image_scale = 0.125

        if cfg.image_scale < 1.0:
            cfg._note(
                f"Image downscaled to {cfg.image_scale}× "
                f"({raw_w*cfg.image_scale:.0f}×{raw_h*cfg.image_scale:.0f}) "
                f"to fit {free_vram:.1f} GB free VRAM "
                f"(raw {raw_mpx:.1f} Mpx → target ≤{target_mpx} Mpx). "
                f"Override with --image_scale 1.0 if you have more VRAM."
            )

    cfg.width  = round(raw_w * cfg.image_scale)
    cfg.height = round(raw_h * cfg.image_scale)

    # ------------------------------------------------------------------ #
    # 2. max_steps
    # ------------------------------------------------------------------ #
    if user_max_steps is not None:
        cfg.max_steps = user_max_steps
    else:
        # More views → more diverse signal → fewer steps needed per epoch
        # but each step is more expensive → keep 30k as safe default,
        # reduce only for quick tests (few views & low-res)
        if num_views <= 20 and cfg.width * cfg.height < 200_000:
            cfg.max_steps = 15_000
            cfg._note(
                f"Only {num_views} views at low resolution — "
                f"max_steps reduced to {cfg.max_steps} (fast experiment mode). "
                f"Use --max_steps 30000 for full quality."
            )
        else:
            cfg.max_steps = 30_000

    # ------------------------------------------------------------------ #
    # 3. Densification schedule
    # ------------------------------------------------------------------ #
    # refine_stop_iter: stop at half the training steps (more conservative for
    # scenes where densification oscillates with many cameras)
    cfg.refine_stop_iter  = cfg.max_steps // 2

    if num_views >= 100:
        # Large multi-view capture: densify more aggressively early on
        cfg.refine_start_iter = 300
        cfg.refine_every      = 100
        cfg._note(
            f"{num_views} views → densification starts earlier (iter {cfg.refine_start_iter})."
        )
    elif num_views <= 30:
        # Few views: slower densification to avoid floaters
        cfg.refine_start_iter = 1_000
        cfg.refine_every      = 200
        cfg._note(
            f"Only {num_views} views → densification starts later (iter {cfg.refine_start_iter}) "
            f"to reduce floaters."
        )
    else:
        cfg.refine_start_iter = 500
        cfg.refine_every      = 100

    # ------------------------------------------------------------------ #
    # 4. Opacity reset interval
    # ------------------------------------------------------------------ #
    # reset_every controls how often needles/floaters get pruned.
    # Large scenes benefit from less frequent resets (avoid pruning valid Gaussians).
    if scene_scale > 10.0:
        cfg.reset_every = 5_000
        cfg._note(
            f"Large scene (scale={scene_scale:.1f}) → opacity reset every {cfg.reset_every} steps."
        )
    else:
        cfg.reset_every = 3_000

    # ------------------------------------------------------------------ #
    # 5. Sparse point cloud size warnings
    # ------------------------------------------------------------------ #
    if num_points < 5_000:
        cfg._note(
            f"Only {num_points:,} sparse points — initialization may be under-sampled. "
            f"Re-run COLMAP with denser matching if quality is low."
        )
    elif num_points > 500_000:
        cfg._note(
            f"{num_points:,} sparse points — init will be slow. "
            f"Consider filtering outliers with COLMAP's filter_points3D."
        )

    # ------------------------------------------------------------------ #
    # 6. log / save cadence
    # ------------------------------------------------------------------ #
    cfg.log_every  = max(100, cfg.max_steps // 60)   # ~60 log entries total
    cfg.save_every = max(1_000, cfg.max_steps // 6)  # ~6 checkpoints total

    return cfg
