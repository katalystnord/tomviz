# -*- coding: utf-8 -*-

###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
from abc import ABCMeta, abstractmethod


class FileType(object):
    """
    Container class for file type information.
    """
    def __init__(self, display_name=None, extensions=None):
        self.display_name = display_name
        self.extensions = extensions

    def __str__(self):
        return "%s (%s)" % (self.display_name,
                            " ".join(["*."+ext for ext in self.extensions]))


class IOBase(object, metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def file_type():
        """
        :returns An instance of the FileFormat class. This is used to associate
        a file type with a reader.
        :rtype tomviz.io.FileType
        """


class Reader(IOBase, metaclass=ABCMeta):
    """
    The base reader class from which readers should be derived.
    """

    """
    Set to True if reader supports loading image stacks.
    """
    supports_stacks = False

    @abstractmethod
    def read(self, file):
        """
        :returns Return a vtkDataObject containing the scalars read.
        :param file: the path or file object to read from
        :rtype vtk.vtkDataObject
        """


class Writer(IOBase, metaclass=ABCMeta):
    """
    The base reader class from which writers should be derived.
    """

    @abstractmethod
    def write(self, file, data):
        """
        :param file: the path or file object to write to
        :param data: The data to write to the file.
        :type data: The vtkDataObject instance.
        """


# -- Helpers for readers ------------------------------------------------------
#
# A reader's read() may return a vtkImageData, or a dict for projection data
# that comes with reference frames:
#
#     {'data': <vtkImageData>, 'flat': <vtkImageData or None>,
#      'dark': <vtkImageData or None>}
#
# Tomviz then opens the flat and dark frames as datasets of their own, ready
# to connect to Flat/Dark Field Correction. Build the images with the
# functions below so tilt angles and metadata (for example the scanner
# geometry) reach the operators.


def image_from_array(array, spacing=(1.0, 1.0, 1.0), tilt_angles=None,
                     metadata=None):
    """A vtkImageData holding a 3D array in Tomviz's (x, y, z) order.

    tilt_angles: one angle in degrees per z slice; makes it a tilt series.
    metadata: a JSON-serialisable dict, available to operators as
        dataset.metadata.
    """
    import numpy as np
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkCommonDataModel import vtkImageData

    from tomviz import _boundary

    a = np.asarray(array)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f'Expected a 2D or 3D array, got shape {a.shape}')
    image = vtkImageData()
    image.SetOrigin(0, 0, 0)
    image.SetSpacing(*[float(v) for v in spacing])
    image.SetExtent(0, a.shape[0] - 1, 0, a.shape[1] - 1, 0, a.shape[2] - 1)
    # VTK runs x fastest, which is Fortran order for (x, y, z).
    flat = np.ravel(a, order='F')
    if flat.dtype.kind not in 'iuf' or flat.dtype.itemsize > 8:
        flat = flat.astype(np.float32)
    scalars = numpy_support.numpy_to_vtk(flat, deep=1)
    scalars.SetName('scalars')
    image.GetPointData().SetScalars(scalars)
    if tilt_angles is not None:
        angles = np.asarray(tilt_angles, dtype=np.float64).ravel()
        if len(angles) != a.shape[2]:
            raise ValueError(f'{len(angles)} tilt angles for {a.shape[2]} '
                             'projections')
        vtk_angles = numpy_support.numpy_to_vtk(angles, deep=1)
        vtk_angles.SetName('tilt_angles')
        image.GetFieldData().AddArray(vtk_angles)
    if metadata:
        _boundary.set_metadata(image, metadata)
    return image


def image_from_frames(frames, pixel_size=(1.0, 1.0), angles=None,
                      metadata=None):
    """A vtkImageData from detector frames (n_frames, n_rows, n_columns),
    in the layout Tomviz's reconstruction operators expect: x along the
    rows (the rotation axis), y along the columns, z the frame index.

    pixel_size: (row pitch, column pitch). angles: degrees per frame, for
    projections (makes it a tilt series); None for flat or dark frames.
    """
    import numpy as np
    f = np.asarray(frames)
    if f.ndim == 2:
        f = f[None]
    if f.ndim != 3:
        raise ValueError(f'Expected frames (n, rows, columns), got {f.shape}')
    return image_from_array(np.moveaxis(f, 0, 2),
                            spacing=(pixel_size[0], pixel_size[1], 1.0),
                            tilt_angles=angles, metadata=metadata)
