# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""Ascend environment settings and cache-key integration."""
import os
import sys
import triton.knobs as _knobs


class ascend_knobs(_knobs.base_knobs):
    """Knobs for the Ascend backend.

    Field names are Python-friendly; every field keeps its original
    environment variable name so existing scripts keep working.
    """

    # Compilation-affecting switches.
    enable_sanitizer: bool = _knobs.env_bool("TRITON_ENABLE_SANITIZER", False)
    print_ub_bits: bool = _knobs.env_bool("ENABLE_PRINT_UB_BITS", False)
    memory_display: bool = _knobs.env_bool("TRITON_MEMORY_DISPLAY", False)
    npu_compiler_path: str = _knobs.env_opt_str("TRITON_NPU_COMPILER_PATH")

    # Autotuning switches.
    enable_ubtuner: str = _knobs.env_str("TRITON_ENABLE_UBTUNER", "")
    autotune_parallel_compile: bool = _knobs.env_bool(
        "TRITON_AUTOTUNE_PARALLEL_COMPILE", True)
    bench_method: str = _knobs.env_str("TRITON_BENCH_METHOD", "default")

    # Launcher / runtime switches.
    compile_only: bool = _knobs.env_bool("TRITON_COMPILE_ONLY", False)
    enable_taskqueue: bool = _knobs.env_bool("TRITON_ENABLE_TASKQUEUE", True)
    device_print: bool = _knobs.env_bool("TRITON_DEVICE_PRINT", False)
    grid_warn_print: bool = _knobs.env_bool("TRITON_GRID_WARN_PRINT", False)


ascend = ascend_knobs()


# Include effective knob values, including overrides that do not update os.environ.
_COMPILE_ENV_VARS = (
    "TRITON_ENABLE_SANITIZER",
    "ENABLE_PRINT_UB_BITS",
    "TRITON_MEMORY_DISPLAY",
    "TRITON_NPU_COMPILER_PATH",
    "NPU_DEVICE_LIMIT",
    "TRITON_ENABLE_UBTUNER",
)
_AUTOTUNE_ENV_VARS = ("TRITON_ENABLE_UBTUNER", "TRITON_BENCH_METHOD")


def _extend_cache_invalidating_env_vars(module):
    original = getattr(module, "get_cache_invalidating_env_vars", None)
    if original is None or getattr(original, "_ascend_extended", False):
        return

    def _patched():
        env_vars = original()
        fields = {knob.key: name for name, knob in ascend.knob_descriptors.items()}
        for key in dict.fromkeys(_COMPILE_ENV_VARS + _AUTOTUNE_ENV_VARS):
            env_vars.pop(key, None)
            field = fields.get(key)
            if field is None:
                value = os.environ.get(key)
                if value is not None:
                    env_vars[key] = value
                continue
            knob = ascend.knob_descriptors[field]
            current = getattr(ascend, field)
            if current == getattr(knob, "default", None):
                continue
            token = _knobs.toenv(current)
            if token is not None and token[0] is not None:
                env_vars[key] = token[0]
        return env_vars

    _patched._ascend_extended = True
    module.get_cache_invalidating_env_vars = _patched


def install_ascend_knobs():
    # Discovery precedes JIT/autotuner imports in upstream Triton. Patch their
    # source too; the union conservatively invalidates both kinds of cache.
    for name in ("triton._C.libtriton", "triton.runtime.jit",
                 "triton.compiler.compiler", "triton.runtime.autotuner"):
        module = sys.modules.get(name)
        if module is not None:
            _extend_cache_invalidating_env_vars(module)


def __getattr__(name):
    return getattr(_knobs, name)
