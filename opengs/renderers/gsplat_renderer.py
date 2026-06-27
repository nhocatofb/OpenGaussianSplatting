import torch
from gsplat import rasterization


class GsplatRenderer:
    def __init__(self, rasterize_mode: str = "antialiased", render_mode: str = "RGB"):
        self.rasterize_mode = rasterize_mode
        self.render_mode    = render_mode

    def render(self,
               gaussian_model,
               viewmat: torch.Tensor,
               K: torch.Tensor,
               width: int,
               height: int,
               active_sh_degree: int = 0) -> tuple:
        """
        Render Gaussians using gsplat rasterization with optional SH colors.

        Args:
            gaussian_model:    VanillaGaussian instance.
            viewmat:           [1, 4, 4] world-to-camera matrix.
            K:                 [1, 3, 3] intrinsic matrix.
            width, height:     Output image size in pixels.
            active_sh_degree:  SH degree to evaluate (0 = DC only, max 3).
                               Passed directly to gsplat; higher degrees are
                               progressively activated during training.

        Returns:
            (rendered_image [H, W, 3],  info dict)
        """
        means     = gaussian_model.means
        quats     = gaussian_model.get_normalized_quats()
        scales    = gaussian_model.get_real_scales()
        opacities = gaussian_model.get_real_opacities()

        # colors: [N, K, 3] SH coefficients when active_sh_degree > 0
        #         gsplat evaluates view-dependent color internally
        colors = gaussian_model.colors   # [N, K, 3]

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=active_sh_degree,
            render_mode=self.render_mode,
            rasterize_mode=self.rasterize_mode,
            packed=False,
        )

        return render_colors[0], info   # [H, W, 3], dict
