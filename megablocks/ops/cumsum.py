# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

from megablocks.op import op

# Wrap this in a try-block with better error message and
# instructions for building the c++ operations.
try:
    import megablocks_ops as ops  # type: ignore
    MEGABLOCKS_OPS_AVAILABLE = True
except ModuleNotFoundError as e:
    MEGABLOCKS_OPS_AVAILABLE = False


# Autograd wrappers for cumsum kernels.
# NOTE: Does not support gradients.
class ExclusiveCumsumOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int):
        if not MEGABLOCKS_OPS_AVAILABLE:
            return torch.cumsum(x, dim) - 1

        if len(x.size()) == 1:
            x = x.view([1, -1])
            out = torch.empty_like(x)
            ops.exclusive_cumsum(x, 1, out)
            return out.squeeze()
        out = torch.empty_like(x)
        ops.exclusive_cumsum(x, dim, out)
        return out


exclusive_cumsum = ExclusiveCumsumOp.apply


class InclusiveCumsumOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int) -> torch.Tensor:
        if not MEGABLOCKS_OPS_AVAILABLE:
            return torch.cumsum(x, dim)

        if len(x.size()) == 1:
            x = x.view([1, -1])
            out = torch.empty_like(x)
            ops.inclusive_cumsum(x, 1, out)
            return out.squeeze()
        out = torch.empty_like(x)
        ops.inclusive_cumsum(x, dim, out)
        return out


inclusive_cumsum = InclusiveCumsumOp.apply
