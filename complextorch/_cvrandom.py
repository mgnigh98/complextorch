import torch

__all__ = ["randn"]


def randn(*size, dtype=torch.complex64, device="cpu", requires_grad=False) -> torch.complex:
    return torch.randn(*size, dtype=dtype, device=device, requires_grad=requires_grad)
