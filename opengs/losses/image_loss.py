import torch
import torch.nn.functional as F
from math import exp

def l1_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between two images."""
    return torch.abs(img1 - img2).mean()

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Construct a 1-D Gaussian kernel."""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Build a 2-D separable Gaussian window for SSIM computation."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute the Structural Similarity Index (SSIM) between two images.
    Expects inputs in [H, W, C] layout.
    """
    channel = img1.size(-1)

    # Reformat from [H, W, C] to [1, C, H, W] for Conv2d
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)

    window = create_window(window_size, channel).to(img1.device)

    # Local means via depthwise convolution
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Standard SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class GSImageLoss(torch.nn.Module):
    def __init__(self, lambda_dssim: float = 0.2, window_size: int = 11, channels: int = 3):
        """
        Combined image loss for 3DGS training: L1 + weighted DSSIM.

        Args:
            lambda_dssim: Weight of the DSSIM loss term (default 0.2).
            window_size:  Gaussian window size for SSIM (default 11).
            channels:     Number of image channels (default 3 = RGB).
        """
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.window_size = window_size
        # Registered as a buffer so it moves automatically with .cuda() / .to(device)
        self.register_buffer("window", create_window(window_size, channels))

    def forward(self, pred_img: torch.Tensor, gt_img: torch.Tensor):
        """
        Compute the combined loss.

        Args:
            pred_img: Rendered image from the model [H, W, 3].
            gt_img:   Ground-truth reference image [H, W, 3].

        Returns:
            Tuple of (total_loss, l1_loss, dssim_loss).
        """
        l1 = l1_loss(pred_img, gt_img)

        channel = pred_img.size(-1)
        img1 = pred_img.permute(2, 0, 1).unsqueeze(0)
        img2 = gt_img.permute(2, 0, 1).unsqueeze(0)
        window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_val = ssim_map.mean()

        # DSSIM = (1 - SSIM) / 2, normalized to [0, 1]
        dssim = (1.0 - ssim_val) / 2.0
        total_loss = (1.0 - self.lambda_dssim) * l1 + self.lambda_dssim * dssim
        return total_loss, l1, dssim
