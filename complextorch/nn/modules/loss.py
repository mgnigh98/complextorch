from typing import Optional

import torch
import torch.nn as nn
import torch.functional as F

from ... import CVTensor

__all__ = [
    "GeneralizedSplitLoss",
    "SplitL1",
    "SplitMSE",
    "SSIM",
    "SplitSSIM",
    "PerpLossSSIM",
    "CVQuadError",
    "CVFourthPowError",
    "CVCauchyError",
    "CVLogCoshError",
    "CVLogError",
]


class GeneralizedSplitLoss(nn.Module):
    """
    Generalized Split Loss Function.

    Operates on the real and imaginary parts separately and sums the losses.

    f(x, y) = f_r(x_r, y_r) + 1j * f_i(x_i, y_i)
    """

    def __init__(self, loss_r: nn.Module, loss_i: nn.Module) -> None:
        super(GeneralizedSplitLoss, self).__init__()
        self.loss_r = loss_r
        self.loss_i = loss_i

    def forward(self, x: CVTensor, y: CVTensor) -> torch.Tensor:
        return self.loss_r(x.real, y.real) + self.loss_i(x.imag, y.imag)


class GeneralizedPolarLoss(nn.Module):
    """
    Generalized Polar Loss Function.

    Operates on the magnitude and phase separately and performs weighted sum of the losses.

    g(x, y) = weight_mag * g_mag(abs(x), abs(y)) + weight_phase * g_phase(angle(x), angle(y))
    """

    def __init__(
        self,
        loss_mag: nn.Module,
        loss_phase: nn.Module,
        weight_mag: float = 1.0,
        weight_phase: float = 1.0,
    ) -> None:
        super(GeneralizedPolarLoss, self).__init__()
        self.loss_mag = loss_mag
        self.loss_phase = loss_phase

        self.weight_mag = weight_mag
        self.weight_phase = weight_phase

    def forward(self, x: CVTensor, y: CVTensor) -> torch.Tensor:
        l_mag = self.weight_mag * self.loss_mag(x.abs(), y.abs())
        l_phase = self.weight_phase * self.loss_phase(x.angle(), y.angle())
        return l_mag + l_phase


class SplitL1(GeneralizedSplitLoss):
    """
    Split L1 Loss Function.

    L1(x, y) = L1(x_r, y_r) + L1(x_i, y_i)
    """

    def __init__(self) -> None:
        super().__init__(nn.L1Loss(), nn.L1Loss())


class SplitMSE(GeneralizedSplitLoss):
    """
    Split MSE Loss Function.

    MSE(x, y) = MSE(x_r, y_r) + MSE(x_i, y_i)
    """

    def __init__(self) -> None:
        super().__init__(nn.MSELoss(), nn.MSELoss())


class SSIM(nn.Module):
    """
    Typical SSIM Loss Function

    Code modified from: https://gitlab.com/computational-imaging-lab/perp_loss
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03) -> None:
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        data_range: Optional[torch.Tensor] = None,
        full: bool = False,
    ) -> torch.Tensor:
        assert isinstance(self.w, torch.Tensor)

        if data_range is None:
            data_range = torch.ones_like(y)  # * Y.max()
            p = (self.win_size - 1) // 2
            data_range = data_range[:, :, p:-p, p:-p]
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(x, self.w)  # typing: ignore
        uy = F.conv2d(y, self.w)  #
        uxx = F.conv2d(x * x, self.w)
        uyy = F.conv2d(y * y, self.w)
        uxy = F.conv2d(x * y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if full:
            return S
        else:
            return S.mean()


class SplitSSIM(GeneralizedSplitLoss):
    """
    Split SSIM Loss Function.

    Returns sum of SSIM over real and imaginary parts separately.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03) -> None:
        super().__init__(SSIM(win_size, k1, k2), SSIM(win_size, k1, k2))

    def forward(
        self,
        x: CVTensor,
        y: CVTensor,
        data_range: Optional[torch.Tensor] = None,
        full: bool = False,
    ) -> torch.Tensor:
        return self.loss_r(
            x.real, y.real, data_range=data_range, full=full
        ) + self.loss_i(x.imag, y.imag, data_range=data_range, full=full)


class PerpLossSSIM(nn.Module):
    """
    Perpendicular SSIM Loss Function.

    M. L. Terpstra, M. Maspero, A. Sbrizzi, C. van den Berg. ⊥-loss: A symmetric loss function for magnetic resonance imaging reconstruction and image registration with deep learning.
    See Fig. 1 for perpendicular explanation
    Eq. (5)
    https://www.sciencedirect.com/science/article/pii/S1361841522001566
    Code: https://gitlab.com/computational-imaging-lab/perp_loss
    """

    def __init__(self) -> None:
        super().__init__()

        self.ssim = SSIM()
        self.param = nn.Parameter(torch.ones(1) / 2)

    def forward(self, x: CVTensor, y: CVTensor) -> torch.Tensor:
        mag_input = torch.abs(x)
        mag_target = torch.abs(y)
        cross = torch.abs(x.real * y.imag - x.imag * y.real)

        angle = torch.atan2(x.imag, x.real) - torch.atan2(y.imag, y.real)
        ploss = torch.abs(cross) / (mag_input + 1e-8)

        aligned_mask = (torch.cos(angle) < 0).bool()

        final_term = torch.zeros_like(ploss)
        final_term[aligned_mask] = mag_target[aligned_mask] + (
            mag_target[aligned_mask] - ploss[aligned_mask]
        )
        final_term[~aligned_mask] = ploss[~aligned_mask]
        ssim_loss = (1 - self.ssim(x, y)) / mag_input.shape[0]

        return (
            final_term.mean() * torch.clamp(self.param, 0, 1)
            + (1 - torch.clamp(self.param, 0, 1)) * ssim_loss
        )


class CVQuadError(nn.Module):
    """
    Complex-Valued Quadratic Error Function CVQuadError.

    CVQuadError(x, y) = 1/2 * sum(err * err.H), with err = x - y.

    Ronny Hänsch. Complex-valued multi-layer perceptrons - an application to polarimetric SAR data
    Eq. (11)
    https://www.ingentaconnect.com/content/asprs/pers/2010/00000076/00000009/art00008?crawler=true&mimetype=application/pdf
    """

    def __init__(self) -> None:
        super(CVQuadError, self).__init__()

    def forward(self, x: CVTensor, y: CVTensor) -> torch.Tensor:
        return 0.5 * ((x - y).abs() ** 2).sum()


class CVFourthPowError(nn.Module):
    """
    Complex Fourth Power Error Function CVFourthPowError.

    CVFourthPowError(x, y) = 1/2 * sum( (err * err.H)^2 ), with err = x - y.

    Ronny Hänsch. Complex-valued multi-layer perceptrons - an application to polarimetric SAR data
    Eq. (12)
    https://www.ingentaconnect.com/content/asprs/pers/2010/00000076/00000009/art00008?crawler=true&mimetype=application/pdf
    """

    def __init__(self) -> None:
        super(CVFourthPowError, self).__init__()

    def forward(self, x: CVTensor, y: CVTensor) -> torch.Tensor:
        return 0.5 * ((x - y).abs() ** 4).sum()


class CVCauchyError(nn.Module):
    """
    Complex-Valued Cauchy Error Function CVCauchyError.

    CVCauchyError(x, y) = 1/2 * sum( (err * err.H)^2 ), with err = x - y.

    Ronny Hänsch. Complex-valued multi-layer perceptrons - an application to polarimetric SAR data
    Eq. (13)
    https://www.ingentaconnect.com/content/asprs/pers/2010/00000076/00000009/art00008?crawler=true&mimetype=application/pdf
    """

    def __init__(self, c: float = 1) -> None:
        super(CVCauchyError, self).__init__()

        self.c2 = c**2

    def forward(self, x: CVTensor, y: CVTensor) -> torch.Tensor:
        return (self.c2 / 2 * torch.log(1 + ((x - y).abs() ** 2) / self.c2)).sum()


class CVLogCoshError(nn.Module):
    """
    Complex-Valued Log-Cosh Error Function CVLogCoshError.

    CVLogCoshError(x, y) = sum(ln(cosh(err * err.H)), with err = x - y.

    Ronny Hänsch. Complex-valued multi-layer perceptrons - an application to polarimetric SAR data
    Eq. (14)
    https://www.ingentaconnect.com/content/asprs/pers/2010/00000076/00000009/art00008?crawler=true&mimetype=application/pdf
    """

    def __init__(self) -> None:
        super(CVLogCoshError, self).__init__()

    def forward(self, x: CVTensor, y: CVTensor) -> torch.Tensor:
        return torch.log(torch.cosh((x - y).abs() ** 2)).sum()


class CVLogError(nn.Module):
    """
    Complex-Valued Log Error Function CVLogError.

    CVLogError(x, y) = sum(err * err.H), with err = log(x) - log(y).

    J Bassey, L Qian, X Li. A Survey of Complex-Valued Neural Networks.
    Eq. (10)
    https://arxiv.org/abs/2101.12249
    """

    def __init__(self) -> None:
        super(CVLogError, self).__init__()

    def forward(self, x: CVTensor, y: CVTensor) -> torch.Tensor:
        err = torch.log(x.complex) - torch.log(y.complex)
        return (err.abs() ** 2).sum()