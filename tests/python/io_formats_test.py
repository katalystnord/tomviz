###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tests for DICOM, MRC and DigitalMicrograph I/O format implementations."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

from vtk import vtkImageData
import vtk.numpy_interface.dataset_adapter as dsa
import vtk.util.numpy_support as np_s
from vtkmodules.util.vtkConstants import VTK_DOUBLE


# -- VTK helpers that replicate what tomviz.internal_utils does, without
#    requiring the in-application guard.

def _vtk_set_array(
    dataobject: vtkImageData,
    newarray: npt.NDArray[np.generic],
    name: str = 'Scalars',
) -> None:
    arr = np.asfortranarray(newarray)
    flat = arr.reshape(-1, order='F')
    if flat.dtype not in (np.int8, np.int16, np.uint16, np.float32,
                          np.float64, np.uint8):
        flat = flat.astype(np.float32)
    vtkshape = newarray.shape
    extent = [0, vtkshape[0] - 1, 0, vtkshape[1] - 1, 0, vtkshape[2] - 1]
    dataobject.SetExtent(extent)
    do = dsa.WrapDataObject(dataobject)
    do.PointData.append(flat, name)
    do.PointData.SetActiveScalars(name)


def _vtk_get_array(
    dataobject: vtkImageData,
    name: str | None = None,
    order: str = 'F',
) -> npt.NDArray[np.generic]:
    do = dsa.WrapDataObject(dataobject)
    if name is not None:
        rawarray = do.PointData.GetAbstractArray(name)
    else:
        rawarray = do.PointData.GetScalars()
    scalars = dsa.vtkDataArrayToVTKArray(rawarray, do)
    return np.reshape(scalars, dataobject.GetDimensions(), order=order)


def _vtk_set_tilt_angles(
    dataobject: vtkImageData,
    newarray: npt.NDArray[np.float64],
) -> None:
    vtkarray = np_s.numpy_to_vtk(newarray, deep=1, array_type=VTK_DOUBLE)
    vtkarray.SetName('tilt_angles')
    do = dsa.WrapDataObject(dataobject)
    do.FieldData.RemoveArray('tilt_angles')
    do.FieldData.AddArray(vtkarray)


def _vtk_get_tilt_angles(
    dataobject: vtkImageData,
) -> npt.NDArray[np.float64] | None:
    do = dsa.WrapDataObject(dataobject)
    rawarray = do.FieldData.GetArray('tilt_angles')
    if isinstance(rawarray, dsa.VTKNoneArray):
        return None
    return dsa.vtkDataArrayToVTKArray(rawarray, do)


def _make_image_data(
    arr: npt.NDArray[np.generic],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> vtkImageData:
    img = vtkImageData()
    img.SetSpacing(*spacing)
    img.SetOrigin(*origin)
    _vtk_set_array(img, arr)
    return img


# Patch tomviz.internal_utils so that the MRC/DICOM writers can call
# get_array, set_array, etc. without needing the full application.
_iu_patch = mock.patch.dict(
    'tomviz.internal_utils.__dict__',
    {
        'get_array': _vtk_get_array,
        'set_array': _vtk_set_array,
        'get_tilt_angles': _vtk_get_tilt_angles,
        'set_tilt_angles': _vtk_set_tilt_angles,
    },
)


@pytest.fixture(autouse=True)
def _patch_internal_utils() -> Generator[None, None, None]:
    with _iu_patch:
        yield


# ---------------------------------------------------------------------------
# MRC writer
# ---------------------------------------------------------------------------

from tomviz.io.formats.mrc import MrcWriter, HEADER_DTYPE


def test_mrc_writer_float32_round_trip(tmp_path: Path) -> None:
    arr = np.random.default_rng(42).random((4, 5, 6)).astype(np.float32)
    path = tmp_path / 'test.mrc'
    MrcWriter().write(str(path), _make_image_data(arr))

    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(1024), dtype=HEADER_DTYPE)[0]
        data = np.frombuffer(f.read(), dtype=np.float32)

    assert header['nx'] == 4
    assert header['ny'] == 5
    assert header['nz'] == 6
    assert header['mode'] == 2  # float32
    assert header['map'] == b'MAP '
    data = data.reshape((6, 5, 4))
    np.testing.assert_allclose(data, np.ascontiguousarray(np.transpose(arr)),
                               atol=1e-7)


def test_mrc_writer_preserves_spacing(tmp_path: Path) -> None:
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    path = tmp_path / 'test.mrc'
    MrcWriter().write(str(path), _make_image_data(arr, spacing=(2.0, 3.0, 4.0)))

    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(1024), dtype=HEADER_DTYPE)[0]

    assert float(header['cella']['x']) == pytest.approx(3 * 2.0)
    assert float(header['cella']['y']) == pytest.approx(4 * 3.0)
    assert float(header['cella']['z']) == pytest.approx(5 * 4.0)


def test_mrc_writer_int16_uses_mode_1(tmp_path: Path) -> None:
    arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    path = tmp_path / 'test.mrc'
    MrcWriter().write(str(path), _make_image_data(arr))

    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(1024), dtype=HEADER_DTYPE)[0]

    assert header['mode'] == 1  # int16


def test_mrc_writer_uint16_uses_mode_6(tmp_path: Path) -> None:
    arr = np.arange(24, dtype=np.uint16).reshape((2, 3, 4))
    path = tmp_path / 'test.mrc'
    MrcWriter().write(str(path), _make_image_data(arr))

    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(1024), dtype=HEADER_DTYPE)[0]

    assert header['mode'] == 6  # uint16


def test_mrc_writer_converts_unsupported_dtype_to_float32(tmp_path: Path) -> None:
    arr = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    path = tmp_path / 'test.mrc'
    MrcWriter().write(str(path), _make_image_data(arr))

    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(1024), dtype=HEADER_DTYPE)[0]
        data = np.frombuffer(f.read(), dtype=np.float32)

    assert header['mode'] == 2  # float32
    data = data.reshape((4, 3, 2))
    np.testing.assert_allclose(data,
                               np.ascontiguousarray(
                                   np.transpose(arr.astype(np.float32))),
                               atol=1e-7)


def test_mrc_writer_statistics(tmp_path: Path) -> None:
    arr = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    path = tmp_path / 'test.mrc'
    MrcWriter().write(str(path), _make_image_data(arr))

    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(1024), dtype=HEADER_DTYPE)[0]

    assert float(header['dmin']) == pytest.approx(1.0)
    assert float(header['dmax']) == pytest.approx(5.0)
    assert float(header['dmean']) == pytest.approx(3.0)
    assert float(header['rms']) == pytest.approx(float(arr.std()), rel=1e-5)


# ---------------------------------------------------------------------------
# DICOM reader/writer
# ---------------------------------------------------------------------------

try:
    import itk
    _HAS_ITK = True
except ImportError:
    _HAS_ITK = False


@pytest.mark.skipif(not _HAS_ITK, reason='ITK not available')
class TestDicom:

    def test_dicom_writer_reader_uint16_round_trip(self, tmp_path: Path) -> None:
        from tomviz.io.formats.dicom import DicomReader, DicomWriter

        arr = np.random.default_rng(7).integers(0, 65535, (4, 5, 6),
                                                 dtype=np.uint16)
        img = _make_image_data(arr, spacing=(1.5, 2.0, 2.5))
        path = str(tmp_path / 'test.dcm')
        DicomWriter().write(path, img)

        result = DicomReader().read(path)
        result_arr = _vtk_get_array(result)

        assert result_arr.shape == arr.shape
        np.testing.assert_array_equal(result_arr, arr)

        result_spacing = result.GetSpacing()
        assert result_spacing[0] == pytest.approx(1.5)
        assert result_spacing[1] == pytest.approx(2.0)
        assert result_spacing[2] == pytest.approx(2.5)

    def test_dicom_writer_reader_uint8_round_trip(self, tmp_path: Path) -> None:
        from tomviz.io.formats.dicom import DicomReader, DicomWriter

        arr = np.random.default_rng(8).integers(0, 255, (3, 4, 5),
                                                 dtype=np.uint8)
        img = _make_image_data(arr)
        path = str(tmp_path / 'test.dcm')
        DicomWriter().write(path, img)

        result = DicomReader().read(path)
        result_arr = _vtk_get_array(result)
        assert result_arr.shape == arr.shape
        np.testing.assert_array_equal(result_arr, arr)

    def test_dicom_writer_converts_float_to_uint16(self, tmp_path: Path) -> None:
        from tomviz.io.formats.dicom import DicomReader, DicomWriter

        arr = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        img = _make_image_data(arr)
        path = str(tmp_path / 'test.dcm')
        DicomWriter().write(path, img)

        result = DicomReader().read(path)
        result_arr = _vtk_get_array(result)
        assert result_arr.dtype == np.uint16
        # 0.0 -> 0, 0.5 -> ~32767, 1.0 -> 65535
        assert result_arr.flat[0] == 0
        assert result_arr.flat[2] == 65535
        assert abs(int(result_arr.flat[1]) - 32767) < 2

    def test_dicom_preserves_tilt_angles(self, tmp_path: Path) -> None:
        from tomviz.io.formats.dicom import DicomReader, DicomWriter

        arr = np.zeros((3, 3, 3), dtype=np.uint16)
        img = _make_image_data(arr)
        angles = np.array([-60.0, 0.0, 60.0])
        _vtk_set_tilt_angles(img, angles)

        path = str(tmp_path / 'test.dcm')
        DicomWriter().write(path, img)

        result = DicomReader().read(path)
        result_angles = _vtk_get_tilt_angles(result)
        assert result_angles is not None
        np.testing.assert_allclose(result_angles, angles)

    def test_dicom_reader_handles_no_tilt_angles(self, tmp_path: Path) -> None:
        from tomviz.io.formats.dicom import DicomReader, DicomWriter

        arr = np.zeros((3, 3, 3), dtype=np.uint16)
        img = _make_image_data(arr)

        path = str(tmp_path / 'test.dcm')
        DicomWriter().write(path, img)

        result = DicomReader().read(path)
        result_angles = _vtk_get_tilt_angles(result)
        assert result_angles is None


# ---------------------------------------------------------------------------
# Gatan DigitalMicrograph reader
# ---------------------------------------------------------------------------

class _FakeFileDM:
    """Stands in for tomviz.io.dm.FileDM, returning a dataset as its
    getDataset() does: the data in C order with one calibration per
    axis."""

    dataset: dict[str, object] | Exception = {}
    numObjects = 2
    thumbnail = True

    def __init__(self, path: str, on_memory: bool = True) -> None:
        pass

    def __enter__(self) -> _FakeFileDM:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def getDataset(self, index: int) -> dict[str, object]:
        if isinstance(self.dataset, Exception):
            raise self.dataset
        return self.dataset


def _units(image_data: vtkImageData) -> list[str]:
    units = image_data.GetFieldData().GetAbstractArray('units')
    return [units.GetValue(i) for i in range(units.GetNumberOfValues())]


def _read_dm(dataset: dict[str, object] | Exception,
             num_objects: int = 2) -> vtkImageData:
    from tomviz.io.formats.digitalmicrograph import DigitalMicrographReader

    with mock.patch.object(_FakeFileDM, 'dataset', dataset), \
            mock.patch.object(_FakeFileDM, 'numObjects', num_objects), \
            mock.patch('tomviz.io.dm.FileDM', _FakeFileDM):
        return DigitalMicrographReader().read('a.dm4')


def test_dm_image_reads_with_calibration() -> None:
    image = np.arange(12, dtype=np.uint16).reshape((3, 4))
    result = _read_dm({
        'data': image,
        'pixelSize': [0.5, 0.25],
        'pixelOrigin': [0.0, 0.0],
        'pixelUnit': ['nm', 'nm'],
    })

    # x is the image column, y the image row (as VTK reads a TIFF).
    result_arr = _vtk_get_array(result)
    assert result_arr.dtype == np.uint16
    np.testing.assert_array_equal(result_arr[:, :, 0], image.T)

    assert result.GetSpacing() == pytest.approx((0.25, 0.5, 0.25))
    assert _units(result) == ['nm', 'nm', 'nm']
    assert _vtk_get_tilt_angles(result) is None


def test_dm_stack_in_degrees_reads_as_tilt_series() -> None:
    # As Gatan's tomography acquisition writes it: -90 to +90 degrees.
    stack = np.arange(91 * 3 * 4, dtype=np.float32).reshape((91, 3, 4))
    result = _read_dm({
        'data': stack,
        'pixelSize': [2.0, 0.018, 0.018],
        'pixelOrigin': [45.0, 0.0, 0.0],
        'pixelUnit': ['deg', '\u00b5m', '\u00b5m'],
    })

    np.testing.assert_array_equal(_vtk_get_array(result),
                                  np.transpose(stack, (2, 1, 0)))

    result_angles = _vtk_get_tilt_angles(result)
    assert result_angles is not None
    np.testing.assert_allclose(result_angles, np.arange(-90.0, 91.0, 2.0))

    assert result.GetSpacing() == pytest.approx((0.018, 0.018, 0.018))
    assert _units(result) == ['um', 'um', 'um']


def test_dm_stack_calibrated_in_its_images_only() -> None:
    # Stacks can carry a calibration for the images and none for the
    # stack axis.
    stack = np.zeros((3, 2, 4), dtype=np.uint8)
    result = _read_dm({
        'data': stack,
        'pixelSize': [0.06, 0.06],
        'pixelOrigin': [0.0, 0.0],
        'pixelUnit': ['\u00b5m', '\u00b5m'],
    })

    assert result.GetDimensions() == (4, 2, 3)
    assert result.GetSpacing() == pytest.approx((0.06, 0.06, 0.06))
    assert _vtk_get_tilt_angles(result) is None


def test_dm_uncalibrated_image_has_no_units() -> None:
    result = _read_dm({
        'data': np.zeros((3, 4), dtype=np.int16),
        'pixelSize': [1.0, 1.0],
        'pixelOrigin': [0.0, 0.0],
        'pixelUnit': ['', ''],
    })

    assert result.GetSpacing() == (1.0, 1.0, 1.0)
    assert result.GetFieldData().GetAbstractArray('units') is None


def test_dm_complex_data_reads_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _read_dm({
        'data': np.zeros((3, 4), dtype=np.complex64),
        'pixelSize': [1.0, 1.0],
        'pixelOrigin': [0.0, 0.0],
        'pixelUnit': ['', ''],
    })

    assert result.GetNumberOfPoints() == 0
    assert 'complex64' in capsys.readouterr().out


def test_dm_rgb_data_reads_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As tomviz.io.dm raises it for DM data type 23.
    error = OSError('Unsupported binary data type during conversion to '
                    'numpy dtype. DM dataType == 23')
    result = _read_dm(error)

    assert result.GetNumberOfPoints() == 0
    assert 'RGB or binary' in capsys.readouterr().out


def test_dm_file_with_several_images_reads_the_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A thumbnail and three images.
    result = _read_dm({
        'data': np.zeros((3, 4), dtype=np.uint8),
        'pixelSize': [1.0, 1.0],
        'pixelOrigin': [0.0, 0.0],
        'pixelUnit': ['', ''],
    }, num_objects=4)

    assert result.GetDimensions() == (4, 3, 1)
    assert 'has 3 images; opening the first' in capsys.readouterr().out
