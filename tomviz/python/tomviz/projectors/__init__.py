###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tomography geometry and projectors, shared by the reconstruction operators.

Cone-beam geometry and voxel-driven backprojection, on the CPU (numpy) or on
any GPU through WebGPU (optional wgpu).

Headless: depends only on numpy, never on the tomviz application, so it
can be used and tested on its own.
"""

from .geometry import ConeGeometry, VolumeGeometry

__all__ = ['ConeGeometry', 'VolumeGeometry']
