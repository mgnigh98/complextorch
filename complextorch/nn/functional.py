import torch
import torch.nn as nn

from typing import List, Optional

from .. import CVTensor, from_polar

__all__ = [
    "apply_complex",
    "apply_complex_split",
    "apply_complex_polar",
    "inv_sqrtm2x2",
    "cv_batch_norm",
    "cv_layer_norm",
]


def apply_complex(
    real_module: nn.Module, 
    imag_module: nn.Module, 
    x: CVTensor
) -> CVTensor:
    """
    Naive complex computation between a complex-valued module defined by two
    real-valued modules and a complex-valued tensor (CVTensor).
    """
    return CVTensor(
        real_module(x.real) - imag_module(x.imag),
        real_module(x.imag) + imag_module(x.real)
    )


def apply_complex_split(r_fun, i_fun, x: CVTensor) -> CVTensor:
    return CVTensor(r_fun(x.real), i_fun(x.imag))


def apply_complex_polar(mag_fun, phase_fun, x: CVTensor) -> CVTensor:
    return from_polar(mag_fun(x.abs()), phase_fun(x.angle()))


def inv_sqrtm2x2(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    symmetric: bool = False,
):
    """
    Compute the inverse matrix square root of a 2x2 matrix: A^-1/2
    Following: https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix

    A = [ a b
          c d ]

    recall:
    A^-1 = 1/det(A) * [  d  -b
                        -c   a ]

    define:
    delta = det(A) = ad - bc
    tau = trace(A) = a + d

    s = sqrt(delta)
    t = sqrt(tau + 2s)

    A^(1/2) = 1/t * [ a+s  b
                      c    d+s ]

    therefore:
    A^(-1/2) = 1/(st) * [  d+s  -b
                          -c     a+s ]

    define:
    B = A^(-1/2) = [ w x
                     y z]

    w = 1/(st) * (d+s)
    x = 1/(st) * (-b)
    y = 1/(st) * (-c)
    z = 1/(st) * (a+s)
    """

    if symmetric:
        # If A is symmetric, b == c and x == y
        # Hence, we ignore c and y to save one multiplaction
        delta = a * d - b * b
        tau = a + d

        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)

        coeff = 1 / (s * t)

        w, z = coeff * (d + s), coeff * (a + s)
        x, y = -coeff * b, None
    else:
        delta = a * d - b * c
        tau = a + d

        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)

        coeff = 1 / (s * t)

        w, z = coeff * (d + s), coeff * (a + s)
        x, y = -coeff * b, -coeff * c

    return w, x, y, z


def _whiten2x2_batch_norm(
    x: torch.Tensor,
    training: bool = True,
    running_mean: Optional[torch.Tensor] = None,
    running_cov: Optional[torch.Tensor] = None,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    """Performs 2x2 whitening for batch normalization

    Args:
        x (torch.Tensor): Input tensor of size 2 x B x F x ...
        training (bool, optional): Boolean if in training mode. Defaults to True.
        running_mean (torch.Tensor, optional): Running mean to track. Defaults to None.
        running_cov (torch.Tensor, optional): Running covariance matrices to track. Defaults to None.
        momentum (float, optional): The weight in the exponential moving average used to keep track of the running feature statistics. Defaults to 0.1.
        eps (float, optional): The ridge coefficient to stabilize the estimate of the covariance. Defaults to 1e-5.

    Returns:
        torch.Tensor : Batch normalized data
    """
    # assume tensor is 2 x B x F x ...
    assert x.dim() >= 3

    # Axes over which to compute mean and covariance
    axes = 1, *range(3, x.dim())

    # tail shape for broadcasting ? x 1 x F x [*1]
    tail = 1, x.shape[2], *([1] * (x.dim() - 3))

    # Compute the batch mean [2, F]
    if training or running_mean is None:
        mean = x.mean(dim=axes, keepdim=True)
        if running_mean is not None:
            running_mean += momentum * (mean.data.squeeze() - running_mean)

    else:
        mean = running_mean

    # Center the batch
    x -= mean

    # Compute the batch covariance [2, 2, F]
    if training or running_cov is None:
        var = (x * x).mean(dim=axes) + eps
        v_rr, v_ii = var[0], var[1]

        v_ir = (x[0] * x[1]).mean([a - 1 for a in axes])
        if running_cov is not None:
            cov = torch.stack(
                [
                    v_rr.data,
                    v_ir.data,
                    v_ir.data,
                    v_ii.data,
                ],
                dim=0,
            ).view(2, 2, -1)
            running_cov += momentum * (cov - running_cov)

    else:
        v_rr, v_ir, v_ir, v_ii = running_cov.view(4, -1)

    # Compute inverse matrix square root for ZCA whitening
    p, q, _, s = inv_sqrtm2x2(v_rr, v_ir, None, v_ii, symmetric=True)

    # Whiten the batch
    return torch.stack(
        [
            x[0] * p.view(tail) + x[1] * q.view(tail),
            x[0] * q.view(tail) + x[1] * s.view(tail),
        ],
        dim=0,
    )


def cv_batch_norm(
    x: CVTensor,
    running_mean: Optional[torch.Tensor] = None,
    running_var: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> CVTensor:
    """
    Applies complex-valued Batch Normalization as described in
    (Trabelsi et al., 2018) for each channel across a batch of data.

    Arguments
    ---------
    x : cvtorch.CVTensor
        The input complex-valued data is expected to be at least 2d, with
        shape [B, F, ...], where `B` is the batch dimension, `F` -- the
        channels/features, `...` -- the spatial dimensions (if present).

    running_mean : torch.tensor, or None
        The tensor with running mean statistics having shape [2, F]. Ignored
        if explicitly `None`.

    running_var : torch.tensor, or None
        The tensor with running real-imaginary covariance statistics having
        shape [2, 2, F]. Ignored if explicitly `None`.

    weight : torch.tensor, default=None
        The 2x2 weight matrix of the affine transformation of real and
        imaginary parts post normalization. Has shape [2, 2, F] . Ignored
        together with `bias` if explicitly `None`.

    bias : torch.tensor, or None
        The offest (bias) of the affine transformation of real and imaginary
        parts post normalization. Has shape [2, F] . Ignored together with
        `weight` if explicitly `None`.

    training : bool, default=True
        Determines whether to update running feature statistics, if they are
        provided, or use them instead of batch computed statistics. If `False`
        then `running_mean` and `running_var` MUST be provided.

    momentum : float, default=0.1
        The weight in the exponential moving average used to keep track of the
        running feature statistics.

    eps : float, default=1e-5
        The ridge coefficient to stabilize the estimate of the real-imaginary
        covariance.
    """
    # check arguments
    assert (running_mean is None and running_var is None) or (
        running_mean is not None and running_var is not None
    )
    assert (weight is None and bias is None) or (
        weight is not None and bias is not None
    )

    # stack along the first axis
    x = torch.stack(x.rect, dim=0)

    # whiten
    z = _whiten2x2_batch_norm(
        x,
        training=training,
        running_mean=running_mean,
        running_cov=running_var,
        momentum=momentum,
        eps=eps,
    )

    # apply affine transformation
    if weight is not None and bias is not None:
        shape = 1, x.shape[2], *([1] * (x.dim() - 3))
        weight = weight.view(2, 2, *shape)
        z = torch.stack(
            [
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ],
            dim=0,
        ) + bias.view(2, *shape)

    return CVTensor(z[0], z[1])


def _whiten2x2_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    eps: float = 1e-5,
):
    """Performs 2x2 whitening for layer normalization

    Args:
        x (torch.Tensor): Input tensor of size 2 x B x F x ...
        eps (float, optional): The ridge coefficient to stabilize the estimate of the covariance. Defaults to 1e-5.

    Returns:
        torch.Tensor : Layer normalized data
    """
    # assume tensor is 2 x B x F x ...
    assert x.dim() >= 3

    # Axes over which to compute mean and covariance
    axes = [-(i + 1) for i in range(len(normalized_shape))]

    # Compute the batch mean [2, B, 1, ...] and center the batch
    mean = x.clone().mean(dim=axes, keepdim=True)
    x -= mean

    # head shape for broadcasting
    head = mean.shape[1:]

    # Compute the batch covariance [2, 2, F]
    var = (x * x).mean(dim=axes) + eps
    v_rr, v_ii = var[0], var[1]

    v_ir = (x[0] * x[1]).mean(dim=axes)

    # Compute inverse matrix square root for ZCA whitening
    p, q, _, s = inv_sqrtm2x2(v_rr, v_ir, None, v_ii, symmetric=True)

    # Whiten the batch
    return torch.stack(
        [
            x[0] * p.view(head) + x[1] * q.view(head),
            x[0] * q.view(head) + x[1] * s.view(head),
        ],
        dim=0,
    )


def cv_layer_norm(
    x: CVTensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> CVTensor:
    """Applies complex-valued Layer Normalization."""

    # stack along the first axis
    x = torch.stack(x.rect, dim=0)

    # whiten
    z = _whiten2x2_layer_norm(
        x,
        normalized_shape,
        eps=eps,
    )

    # apply affine transformation
    if weight is not None:
        shape = *([1] * (x.dim() - 1 - len(normalized_shape))), *normalized_shape
        weight = weight.view(2, 2, *shape)
        z = torch.stack(
            [
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ],
            dim=0,
        ) + bias.view(2, *shape)

    return CVTensor(z[0], z[1])