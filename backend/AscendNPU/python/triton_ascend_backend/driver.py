"""Discovery interface for the future CANN runtime integration."""

from triton.backends.driver import DriverBase

from . import UNAVAILABLE


class AscendDriver(DriverBase):

    @classmethod
    def is_active(cls):
        # Do not claim an Ascend device until both compilation and launch work.
        # Merely having torch-npu installed is insufficient.
        return False

    def __init__(self):
        raise NotImplementedError(UNAVAILABLE)

    def map_python_to_cpp_type(self, ty):
        raise NotImplementedError(UNAVAILABLE)

    def get_current_target(self):
        raise NotImplementedError(UNAVAILABLE)

    def get_active_torch_device(self):
        raise NotImplementedError(UNAVAILABLE)

    def get_benchmarker(self):
        raise NotImplementedError(UNAVAILABLE)
