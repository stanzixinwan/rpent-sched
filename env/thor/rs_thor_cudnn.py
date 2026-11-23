"""Turn cuDNN off as soon as PyTorch loads it, when RS_DISABLE_CUDNN=1.

Installed into the Thor venvs by env/thor/setup.sh together with a one-line .pth
that imports this module. A .pth runs before the container's site-packages (where
torch lives) are on sys.path, and the system's sitecustomize shadows a venv one,
so instead of importing torch here we hook the import of torch.backends.cudnn.
"""
import importlib.abc
import importlib.machinery
import os
import sys


class _CudnnOff(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "torch.backends.cudnn":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            # the module replaces itself in sys.modules with a CudnnModule
            setattr(sys.modules.get(name, module), "enabled", False)

        spec.loader.exec_module = exec_module
        sys.meta_path.remove(self)
        return spec


if os.environ.get("RS_DISABLE_CUDNN") == "1":
    sys.meta_path.insert(0, _CudnnOff())
