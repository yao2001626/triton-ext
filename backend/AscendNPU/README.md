# Ascend NPU Backend for Triton

Experimental out-of-tree Ascend A5 pure-SIMT backend for upstream Triton 3.7.0.
It compiles TTIR through CANN's `bishengir-compile` and launches kernels through
CANN and torch-npu. The compiler and runtime are ported from
[Triton-Ascend](https://github.com/triton-lang/triton-ascend).

## Build and test

Use Python 3.10 or later, Triton 3.7.0, pybind11 3.0.1, CMake, Ninja, and
scikit-build-core. The native attribute binding requires LLVM/MLIR headers from
Triton 3.7.0's LLVM revision `ac5dc54d509169d387fcfd495d71853d81c46484`.

```bash
export LLVM_INSTALL_DIR=/path/to/matching/llvm
make -C backend/AscendNPU build
make -C backend/AscendNPU install
make -C backend/AscendNPU test
```

The two smoke tests cover backend discovery and vector-add lowering to TTIR.
They require the installed wheel but do not require CANN or an NPU.

## Run the examples

Install compatible PyTorch and torch-npu packages, then source the CANN
environment. The backend preserves the device's reported target architecture.

```bash
source /path/to/cann/set_env.sh
export TRITON_DEFAULT_BACKEND=ascend_a5
python backend/AscendNPU/examples/01-vector-add.py
python backend/AscendNPU/examples/02-fused-softmax.py
```

Both examples passed correctness checks on Ascend 950 with Triton 3.7.0 and CANN
9.3.0's BiShengIR 2.0.0 compiler at revision `cba6779b8e1e`. Compiler
capabilities vary between CANN releases; this does not establish compatibility
with every CANN 9.3.0 build.

