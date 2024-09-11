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


# Autograd wrapper for topology kernel.
# NOTE: Does not support gradients.
class TopologyOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        padded_bins: torch.Tensor,
        block_size: int,
        output_block_rows: int,
        output_block_columns: int,
    ):
        if not MEGABLOCKS_OPS_AVAILABLE:
            # Copied from test to give a default behavior
            # when the custom extension is not available.
            rows = output_block_rows
            columns = output_block_columns
            blocking = block_size

            padded_bins = padded_bins.cpu().numpy()

            out = np.zeros([rows * columns])
            start = 0
            for i in range(padded_bins.shape[0]):
                end = padded_bins[i] // blocking
                while start < end:
                    for j in range(columns):
                        out[start * columns + j] = j + i * columns
                    start += 1
            return torch.from_numpy(out).cuda().short()


        out = torch.empty(
            output_block_rows * output_block_columns,
            dtype=torch.int16,
            device=padded_bins.device,
        )
        ops.indices(
            padded_bins,
            block_size,
            output_block_rows,
            output_block_columns,
            out,
        )
        return out


topology = TopologyOp.apply
