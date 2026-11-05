###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Cone-beam X-ray CT: FDK reconstruction and X-ray preprocessing.

Builds on tomviz.projectors for geometry and projectors. Headless: depends
only on numpy (plus optional pyfftw), never on the tomviz application.
"""

from .fdk import fdk
from .preprocess import flat_dark_correct

__all__ = ['fdk', 'flat_dark_correct']
