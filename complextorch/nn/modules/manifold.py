from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t

from ... import CVTensor, from_polar

__all__ = ["wFMConv1d", "wFMConv2d"]


def _normalize_weights_squared(weights: torch.Tensor) -> torch.Tensor:
    return (weights**2) / torch.sum(weights**2)


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    return weights / torch.sum(weights**2)


def _normalize_rows(weights: torch.Tensor) -> torch.Tensor:
    return weights**2 / torch.sum(weights**2, dim=1, keepdim=True)


class _wFMConv2dHelper(nn.Module):
    """Helper class for wFMConv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = (1, 1),
        padding: _size_2_t = (0, 0),
        weight_dropout: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        super(_wFMConv2dHelper, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps

        prod_kernel_size = np.prod(kernel_size)

        self.dropout = nn.Dropout(weight_dropout)

        # Weight matrices
        self.weight_matrix_ang1 = nn.Parameter(
            torch.rand(in_channels, prod_kernel_size), requires_grad=True
        )

        self.weight_matrix_ang2 = nn.Parameter(
            torch.rand(out_channels, in_channels), requires_grad=True
        )

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

    def compute_output_shape(self, input_shape) -> Tuple[int]:
        return tuple(
            int(np.floor((in_shape + 2 * padding - (kernel_size - 1) - 1) / stride + 1))
            for in_shape, padding, kernel_size, stride in zip(
                input_shape, self.padding, self.kernel_size, self.stride
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, mag_ang, in_channels, *input_shape = x.shape

        assert mag_ang == 2, "Input must be complex valued in polar form (mag, ang)"
        assert in_channels == self.in_channels, "Input channels must match"

        out_channels = self.out_channels
        kernel_size = self.kernel_size
        prod_kernel_size = np.prod(kernel_size)

        output_shape = self.compute_output_shape(input_shape)
        L = np.prod(output_shape)  # Total number of unfolded blocks

        x = x.view(batch_size * 2, in_channels, *input_shape)

        # unfolded shape: (batch_size * 2, in_channels * prod_kernel_size, L)
        temporal_buckets = self.unfold(x).view(
            batch_size, 2, in_channels, prod_kernel_size, L
        )

        ### Do magnitude processing
        tb_mag = torch.log(
            temporal_buckets[:, 0]
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * L, in_channels, prod_kernel_size)
            + self.eps
        )

        # Normalize the weights
        wmm1 = _normalize_rows(self.dropout(self.weight_matrix_ang1))
        wmm2 = _normalize_rows(self.dropout(self.weight_matrix_ang2))

        out_mag = (
            torch.sum(tb_mag * wmm1, dim=2).unsqueeze(1).repeat(1, out_channels, 1)
        )

        out_mag = torch.exp(
            torch.sum(out_mag * wmm2, dim=2)
            .view(batch_size, 1, *output_shape, out_channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

        ### Do phase processing
        tb_ang = (
            temporal_buckets[:, 1]
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * L, in_channels, prod_kernel_size)
        )

        # Normalize the weights
        wma1 = _normalize_weights_squared(self.weight_matrix_ang1)
        wma2 = _normalize_weights_squared(self.weight_matrix_ang2)

        out_ang = (
            torch.sum(tb_ang * wma1, dim=2).unsqueeze(1).repeat(1, out_channels, 1)
        )

        out_ang = (
            torch.sum(out_ang * wma2, dim=2)
            .view(batch_size, 1, *output_shape, out_channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

        return torch.cat((out_mag, out_ang), dim=1)


class wFMConv2d(nn.Module):
    """
    Weighted Frechet Mean 2D Convolutional Layer.

    R Chakraborty, Y Xing, S Yu. SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold.
    Eqs. (14)-(16)
    https://arxiv.org/abs/1910.11334

    Modified from implementation: https://github.com/xingyifei2016/RotLieNet
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = (1, 1),
        padding: _size_2_t = (0, 0),
        weight_dropout: float = 0.0,
    ) -> None:
        super(wFMConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_dropout = weight_dropout

        prod_kernel_size = np.prod(kernel_size)

        # Weight matrices for magnitude and angle
        self.weight_matrix_mag = nn.Parameter(
            torch.rand(in_channels, prod_kernel_size), requires_grad=True
        )

        self.weight_matrix_ang = nn.Parameter(
            torch.rand(in_channels, prod_kernel_size), requires_grad=True
        )

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

        self.wFM_conv = _wFMConv2dHelper(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_dropout=weight_dropout,
        )

    def compute_output_shape(self, input_shape) -> Tuple[int]:
        return tuple(
            int(np.floor((in_shape + 2 * padding - (kernel_size - 1) - 1) / stride + 1))
            for in_shape, padding, kernel_size, stride in zip(
                input_shape, self.padding, self.kernel_size, self.stride
            )
        )

    def forward(self, x: CVTensor) -> CVTensor:
        batch_size, in_channels, *input_shape = x.shape

        assert in_channels == self.in_channels, "Input channels must match"

        kernel_size = self.kernel_size
        prod_kernel_size = np.prod(kernel_size)

        output_shape = self.compute_output_shape(input_shape)
        L = np.prod(output_shape)  # Total number of unfolded blocks

        self.fold = nn.Fold(
            output_size=input_shape,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # Separate magnitude and angle from CVTensor input
        (x_mag, x_ang) = x.polar

        ### Do magnitude processing
        x_mag = self.unfold(x_mag).view(batch_size, in_channels, prod_kernel_size, L)

        x_mag = (
            x_mag.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * L, in_channels, prod_kernel_size)
        )

        x_mag = x_mag + _normalize_weights_squared(self.weight_matrix_mag)

        x_mag = (
            x_mag.view(batch_size, *output_shape, in_channels * prod_kernel_size)
            .permute(0, 3, 1, 2)
            .contiguous()
            .unsqueeze(1)
        )

        ### Do phase processing
        x_ang = self.unfold(x_ang).view(batch_size, in_channels, prod_kernel_size, L)

        x_ang = (
            x_ang.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * L, in_channels, prod_kernel_size)
        )

        x_ang = x_ang * _normalize_weights(self.weight_matrix_ang)

        x_ang = (
            x_ang.view(batch_size, *output_shape, in_channels * prod_kernel_size)
            .permute(0, 3, 1, 2)
            .contiguous()
            .unsqueeze(1)
        )

        # Stack the magnitude and phase tensors
        in_fold = self.fold(
            torch.cat((x_mag, x_ang), dim=1).view(
                batch_size, 2 * in_channels * prod_kernel_size, L
            )
        ).view(batch_size, 2, in_channels, *input_shape)

        x_out = self.wFM_conv(in_fold)
        return from_polar(x_out[:, 0], x_out[:, 1])


class wFMConv1d(nn.Module):
    """
    Weighted Frechet Mean 1D Convolutional Layer.

    R Chakraborty, Y Xing, S Yu. SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold.
    Eqs. (14)-(16)
    https://arxiv.org/abs/1910.11334

    Modified from implementation: https://github.com/xingyifei2016/RotLieNet

    Note: This is a wrapper around wFMConv2d that performs a 1D convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        weight_dropout: float = 0.0,
    ) -> None:
        super(wFMConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_dropout = weight_dropout

        self.conv1d = wFMConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            weight_dropout=weight_dropout,
        )

        self.wFM_conv = self.conv1d.wFM_conv

    def forward(self, x: CVTensor) -> CVTensor:
        return self.conv1d(x.unsqueeze(-2)).squeeze()

    @property
    def weight_matrix_ang(self) -> torch.Tensor:
        return self.conv1d.weight_matrix_ang

    @property
    def weight_matrix_mag(self) -> torch.Tensor:
        return self.conv1d.weight_matrix_mag