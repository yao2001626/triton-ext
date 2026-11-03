# Ascend NPU Backend for Triton

Experimental out-of-tree backend scaffold, following the package structure and
`triton.backends` entry point used by [Apple GPU](../AppleGPU/README.md).

This patch establishes discovery and packaging only. It cannot compile or launch
kernels. The driver always reports inactive, including on Ascend hosts, and
explicit compilation raises `NotImplementedError` before AST lowering.

## Layout

```text
backend/AscendNPU/
  CMakeLists.txt
  Makefile
  pyproject.toml
  python/triton_ascend_backend/
    __init__.py
    compiler.py
    driver.py
  test/test_ascend_backend.py
```

## Build and test

Use the Python environment described in the
[repository README](../../README.md). From the repository root:

```bash
make -C backend/AscendNPU build
make -C backend/AscendNPU install
make -C backend/AscendNPU test
```

The root `make build` and `make test` discover this extension automatically
through `[tool.triton-ext]`. Tests require the built wheel to be installed,
matching the Apple GPU workflow. These discovery tests require Triton but no
CANN or hardware. There are no native passes to build in this initial patch.

## Completing the backend

The existing [triton-ascend implementation][ascend] is the reference for the
port. Its compiler imports Ascend-specific bindings from its customized
`libtriton`; copying its Python files or depending on its wheel does not provide
an independent backend for upstream Triton.

Before treating this as an operational backend:

- Select and pin compatible Triton, LLVM, triton-ascend and CANN revisions, plus
  the initial supported Ascend device family.
- Port the necessary Ascend dialects and lowering passes into a native plugin,
  registered through `tritonGetPluginInfo`, as Apple GPU does. Load that library
  from this package and register its dialects on compilation contexts.
- Implement TTIR lowering and BiSheng binary generation, including validated
  compiler options, metadata and toolchain-sensitive cache keys. Do not reuse
  Metal's TTGIR layouts or SIMD width as Ascend hardware defaults.
- Implement CANN binary loading, argument packing, streams and kernel launch,
  with a compatible torch-npu integration.
- Add compile tests and hardware tests for masked vector addition, including
  tail elements and multiple launches. Enable driver activation only after these
  paths work together.
- Establish maintainers and an Ascend CI/deployment plan before upstream
  submission, as required by the repository's contributing guidelines.

Installing this scaffold alongside another backend with an `ascend` entry point
is unsupported: Triton's discovery dictionary uses entry point names as keys.

[ascend]: https://github.com/triton-lang/triton-ascend
