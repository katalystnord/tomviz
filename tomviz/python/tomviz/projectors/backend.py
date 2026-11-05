###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Choosing where backprojection runs.

'cpu' always works. 'gpu' uses WebGPU through wgpu-py (Vulkan, Metal or
DX12, so any vendor) and raises if no usable adapter is found. 'auto' tries
the GPU and falls back to the CPU, recording why.

Environment variables (optional):
    TOMVIZ_WGPU_ADAPTER: pick an adapter whose name contains this text
        (case-insensitive), or 'cpu' / 'discrete' / 'integrated' for the
        adapter type. Default: the high-performance adapter.
    TOMVIZ_CONEBEAM_THREADS: CPU threads for the CPU backend.
"""

from dataclasses import dataclass
from typing import Callable

from . import backproject_cpu


@dataclass
class Backend:
    name: str  # 'cpu' or 'gpu'
    description: str
    backproject: Callable


def cpu_backend(reason=None):
    desc = f'CPU, {backproject_cpu.default_threads()} threads'
    if reason:
        desc += f' ({reason})'
    return Backend('cpu', desc, backproject_cpu.backproject)


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
