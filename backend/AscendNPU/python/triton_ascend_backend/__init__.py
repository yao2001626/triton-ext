"""Ascend NPU backend scaffold; compilation and execution are not yet ported.

Import has no CANN, torch-npu or native library side effects, so Triton can
discover this package on hosts without Ascend hardware.
"""

UNAVAILABLE = (
    "The triton-ext Ascend NPU backend is a scaffold. "
    "Ascend lowering, binary generation and the CANN launcher are not "
    "implemented. Use a matching triton-ascend distribution for execution.")
