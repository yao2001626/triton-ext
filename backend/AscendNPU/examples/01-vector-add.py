# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Vector Addition
===============
"""

import torch_npu  # noqa: F401
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=128, num_warps=4)
    return output


# %%
# Unit Test
# ---------


def test_vector_add():
    torch.manual_seed(0)
    size = 8193
    x = torch.randn(size, dtype=torch.float32, device="npu")
    y = torch.randn(size, dtype=torch.float32, device="npu")
    output_triton = add(x, y)
    output_torch = x + y
    torch.testing.assert_close(output_triton, output_torch)
    print(output_triton)
    print(output_torch)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')


# %%
# Use an irregular size to exercise the masked final block.
if __name__ == "__main__":
    test_vector_add()
    print("======Vector Addition Test Passed!======")
