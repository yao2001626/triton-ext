"""Load MLIR bindings only when compilation needs them."""
from importlib import import_module


def __getattr__(name):
    if name == 'ir':
        import triton
        if triton.__version__ != '3.7.0':
            raise RuntimeError(
                f'Ascend native bindings require Triton 3.7.0; found {triton.__version__}. '
                'Install the backend wheel with dependencies enabled.')
        return import_module('._ascend_ir', __package__)
    raise AttributeError(name)
