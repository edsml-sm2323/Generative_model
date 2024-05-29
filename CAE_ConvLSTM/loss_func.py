from math import exp

import torch
import torch.nn.functional as F

__all__ = ["gaussian", "create_window", "ssim", "SSIMLoss"]


def gaussian(window_size, sigma):
    """
    Generates a 1D Gaussian window.

    Parameters:
    window_size (int): The size of the Gaussian window.
    sigma (float): The standard deviation of the Gaussian.

    Returns:
    Tensor: 1D tensor representing the Gaussian window.
    """

    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    Creates a 2D Gaussian window suitable for SSIM calculation.

    Parameters:
    window_size (int): The size of the window.
    channel (int): The number of channels in the images.

    Returns:
    Tensor: A 2D Gaussian window tensor.
    """

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()
    _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Parameters:
    img1 (Tensor): The first image.
    img2 (Tensor): The second image.
    window (Tensor): The Gaussian window.
    window_size (int): The size of the Gaussian window.
    channel (int): The number of channels in the images.
    size_average (bool, optional):
    If True, returns the mean SSIM.
    Otherwise, returns the full SSIM map.

    Returns:
    Tensor: The SSIM score.
    """
    L = 1
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    a = 2 * mu1_mu2 + C1
    b = 2 * sigma12 + C2
    c = mu1_sq + mu2_sq + C1
    d = sigma1_sq + sigma2_sq + C2
    ssim_map = ((a) * (b)) / ((c) * (d))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    """
    A PyTorch module
    for computing the Structural Similarity Index (SSIM)
    loss between two images.

    Parameters:
    window_size (int, optional): The size of the Gaussian window.
    size_average (bool, optional):
    If True, returns the mean SSIM loss.
    Otherwise, returns the full SSIM loss map.

    Methods:
    forward: Computes the SSIM loss between two images.
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # print(img1.shape)
        (_, channel, _, _) = img1.size()

        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = (
                create_window(self.window_size, channel)
                .to(img1.device)
                .type(img1.dtype)
            )
            self.window = window
            self.channel = channel

        return -ssim(
            img1, img2, window, self.window_size, self.channel, self.size_average
        )
