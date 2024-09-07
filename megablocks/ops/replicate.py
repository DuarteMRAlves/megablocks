# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# Wrap this in a try-block with better error message and
# instructions for building the c++ operations.
try:
    import megablocks_ops as ops  # type: ignore
    MEGABLOCKS_OPS_AVAILABLE = True
except ModuleNotFoundError as e:
    MEGABLOCKS_OPS_AVAILABLE = False


# Autograd wrapper for replicate kernel.
class ReplicateOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, bins: torch.Tensor, num_outputs: int):
        ctx.save_for_backward(bins)

        if not MEGABLOCKS_OPS_AVAILABLE:
            # Copied from test to give a default behavior
            # when the custom extension is not available.
            x = x.cpu().numpy()
            bins = bins.cpu().numpy()
            out = np.zeros((x.shape[0], num_outputs))
            for batch_idx in range(x.shape[0]):
                start = 0
                for i, end in enumerate(bins):
                    value = x[batch_idx, i]
                    while start < end:
                        out[batch_idx, start] = value
                        start += 1
            return torch.from_numpy(out).cuda().half()

        out = torch.empty((x.shape[0], num_outputs), dtype=x.dtype, device=x.device)
        ops.replicate_forward(x, bins, out)
        return out

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        bins, = ctx.saved_tensors
        out = torch.empty((grad.shape[0], bins.shape[0]), dtype=grad.dtype, device=grad.device)
        ops.replicate_backward(grad, bins, out)
        return out, None, None


replicate = ReplicateOp.apply
