###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Choosing where backprojection and forward projection run.

'cpu' is the numpy reference and always works. 'gpu' uses WebGPU through
wgpu-py (Vulkan, Metal or DX12, so any vendor) and raises if no usable
adapter is found. That includes software adapters such as Mesa's lavapipe,
which run on the CPU but are much faster than numpy. 'auto' tries WebGPU
and falls back to numpy, recording why.

Environment variables (optional):
    TOMVIZ_WGPU_ADAPTER: pick an adapter whose name contains this text
        (case-insensitive), or 'cpu' / 'discrete' / 'integrated' for the
        adapter type. Default: discrete, then integrated, then software.
    TOMVIZ_CONEBEAM_THREADS: CPU threads for the CPU backend.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from . import backproject_cpu, forward_cpu


@dataclass
class Backend:
    name: str  # 'cpu' or 'gpu'
    description: str
    backproject: Callable
    forward_project: Callable
    # Optional: the whole SIRT loop on the device (see webgpu.sirt_loop).
    sirt_loop: Optional[Callable] = None
    # A software adapter (lavapipe, WARP): GPU code running on the CPU.
    software: bool = False


def cpu_backend(reason=None):
    desc = f'CPU, {backproject_cpu.default_threads()} threads'
    if reason:
        desc += f' ({reason})'
    return Backend('cpu', desc, backproject_cpu.backproject,
                   forward_cpu.forward_project)


def get_backend(kind='auto'):
    kind = (kind or 'auto').lower()
    if kind not in ('auto', 'cpu', 'gpu'):
        raise ValueError(f"Unknown backend {kind!r}; use 'auto', 'cpu' "
                         "or 'gpu'")
    if kind == 'cpu':
        return cpu_backend()
    try:
        from . import webgpu
        return webgpu.make_backend()
    except Exception as e:
        if kind == 'gpu':
            raise RuntimeError(f'GPU backend unavailable: {e}') from e
        return cpu_backend(f'GPU unavailable: {e}')
