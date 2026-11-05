"""Backend discovery and TTIR compilation without CANN or an NPU."""

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends import backends
from triton.backends.compiler import GPUTarget

from triton_ascend_backend.compiler import AscendBackend, AscendOptions
from triton_ascend_backend.driver import NPUDriver


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_triton_discovery_imports_the_backend():
    assert backends["ascend_a5"].compiler is AscendBackend
    assert backends["ascend_a5"].driver is NPUDriver


def test_the_compiler_lowers_a_kernel_to_ttir():
    backend = AscendBackend(GPUTarget("npu", "Ascend950PR_9579", 32))
    options = AscendOptions(arch=backend.target.arch, num_warps=4)
    stages = {}
    backend.add_stages(stages, options, "ttir")
    assert list(stages) == ["ttir", "npubin"]

    source = triton.compiler.ASTSource(
        fn=add_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
                   "n": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 128},
    )
    context = ir.context()
    backend.load_dialects(context)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options),
        backend.get_module_map(), context,
    )
    ttir = str(stages["ttir"](module, {}))
    assert "tt.func public @add_kernel" in ttir
    assert "tt.load" in ttir and "arith.addf" in ttir and "tt.store" in ttir
