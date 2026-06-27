import torch
from gsplat.strategy import DefaultStrategy


def create_gs_optimizers(params: torch.nn.ParameterDict,
                         means_lr: float    = 1.6e-4,
                         scales_lr: float   = 5e-3,
                         quats_lr: float    = 1e-3,
                         opacities_lr: float = 5e-2,
                         sh_dc_lr: float    = 2.5e-3,
                         sh_rest_lr: float  = 1.25e-4) -> dict:
    """
    Create one Adam optimizer per Gaussian parameter group.

    Keeping optimizers separate is required by gsplat's DefaultStrategy so it
    can grow/shrink parameter tensors independently during densification.

    SH learning rates follow the original 3DGS paper:
      - sh_dc   (degree-0 DC) gets the full color LR  (2.5e-3)
      - sh_rest (degrees 1-3) gets 1/20 of that        (1.25e-4)
        because higher-order coefficients need finer updates to avoid oscillation.
    """
    optimizers = {
        "means":     torch.optim.Adam([params["means"]],     lr=means_lr),
        "scales":    torch.optim.Adam([params["scales"]],    lr=scales_lr),
        "quats":     torch.optim.Adam([params["quats"]],     lr=quats_lr),
        "opacities": torch.optim.Adam([params["opacities"]], lr=opacities_lr),
        "sh_dc":     torch.optim.Adam([params["sh_dc"]],     lr=sh_dc_lr),
        "sh_rest":   torch.optim.Adam([params["sh_rest"]],   lr=sh_rest_lr),
    }
    return optimizers


def get_default_strategy(refine_start_iter: int = 500,
                         refine_stop_iter: int  = 15_000,
                         refine_every: int      = 100,
                         reset_every: int       = 3_000) -> DefaultStrategy:
    """
    Return gsplat's DefaultStrategy for Gaussian densification / pruning.
    Automatically handles Split, Clone, and Prune operations.
    """
    return DefaultStrategy(
        refine_start_iter=refine_start_iter,
        refine_stop_iter=refine_stop_iter,
        refine_every=refine_every,
        reset_every=reset_every,
    )
