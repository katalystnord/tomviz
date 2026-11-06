###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tomography geometry and projectors, shared by the reconstruction operators.

Cone- and parallel-beam geometry, voxel-driven backprojection and ray-driven
forward projection (Joseph's method), on the CPU (numpy) or on any GPU through
WebGPU (optional wgpu).

Headless: depends only on numpy, never on the tomviz application, so it
can be used and tested on its own.
"""

from .geometry import ConeGeometry, ParallelGeometry, VolumeGeometry

__all__ = ['ConeGeometry', 'ParallelGeometry', 'VolumeGeometry']
