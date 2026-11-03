"""Exercise discovery with real Triton; no Ascend hardware is required."""

import inspect
from importlib.metadata import entry_points

import pytest
from triton.backends import backends
from triton.backends.compiler import GPUTarget

from triton_ascend_backend.compiler import AscendBackend
from triton_ascend_backend.driver import AscendDriver


def test_backend_discovery():
    entries = entry_points().select(group="triton.backends", name="ascend")
    assert any(ep.value == "triton_ascend_backend" for ep in entries)
    assert backends["ascend"].compiler is AscendBackend
    assert backends["ascend"].driver is AscendDriver
    assert not inspect.isabstract(AscendBackend)
    assert not inspect.isabstract(AscendDriver)


def test_target_selection():
    assert AscendBackend.supports_target(GPUTarget("ascend", "test", 1))
    assert not AscendBackend.supports_target(GPUTarget("cuda", 90, 32))
    assert not AscendBackend.supports_target(GPUTarget("hip", "gfx942", 64))


def test_scaffold_does_not_activate():
    assert not AscendDriver.is_active()
    with pytest.raises(NotImplementedError, match="CANN launcher"):
        AscendDriver()


def test_compilation_reports_missing_implementation():
    backend = AscendBackend(GPUTarget("ascend", "test", 1))
    with pytest.raises(NotImplementedError, match="Ascend lowering"):
        backend.parse_options({})
    stages = {}
    with pytest.raises(NotImplementedError, match="binary generation"):
        backend.add_stages(stages, None, None)
    assert stages == {}
