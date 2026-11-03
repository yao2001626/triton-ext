"""Triton compiler interface reserved for the out-of-tree Ascend port."""

from triton.backends.compiler import BaseBackend, GPUTarget

from . import UNAVAILABLE


class AscendBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "ascend"

    def hash(self):
        return "ascend-npu-scaffold-0.1.0"

    def parse_options(self, options):
        # Fail before AST lowering: no architecture or tuning defaults have
        # been validated against an Ascend toolchain yet.
        raise NotImplementedError(UNAVAILABLE)

    def add_stages(self, stages, options, language):
        raise NotImplementedError(UNAVAILABLE)

    def load_dialects(self, context):
        raise NotImplementedError(UNAVAILABLE)

    def get_module_map(self):
        return {}
