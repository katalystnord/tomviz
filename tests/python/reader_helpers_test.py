"""Helpers for file readers: images with tilt angles and metadata, and
metadata carried through to operators and back."""

import numpy as np
import pytest

from tomviz import _boundary
from tomviz.io import image_from_array, image_from_frames


def test_frames_become_a_tilt_series():
    frames = np.arange(4 * 3 * 5, dtype=np.uint16).reshape(4, 3, 5)
    image = image_from_frames(frames, (0.25, 0.5), angles=[0, 1, 2, 3])
    ds = _boundary.wrap_vtk_image(image, legacy=True)
    assert ds.active_scalars.shape == (3, 5, 4)
    np.testing.assert_array_equal(ds.active_scalars[:, :, 2], frames[2])
    np.testing.assert_array_equal(ds.tilt_angles, [0, 1, 2, 3])
    assert tuple(ds.spacing) == (0.25, 0.5, 1.0)
    assert ds.active_scalars.dtype == np.uint16


def test_wrong_number_of_angles_is_refused():
    with pytest.raises(ValueError, match='3 tilt angles for 4'):
        image_from_frames(np.zeros((4, 2, 2)), angles=[0, 1, 2])


def test_metadata_reaches_operators_and_comes_back():
    image = image_from_array(np.zeros((2, 2, 2), np.float32),
                             metadata={'geometry': {'sod': 66.0}})
    ds = _boundary.wrap_vtk_image(image)
    assert ds.metadata == {'geometry': {'sod': 66.0}}
    ds.metadata['geometry']['sdd'] = np.float64(199.0)
    ds.metadata['angles'] = np.array([0.0, 1.5])
    _boundary.flush_dataset(ds, image)
    assert _boundary.get_metadata(image) == {
        'geometry': {'sod': 66.0, 'sdd': 199.0}, 'angles': [0.0, 1.5]}


def test_unreadable_metadata_is_ignored():
    image = image_from_array(np.zeros((2, 2, 2), np.float32))
    assert _boundary.get_metadata(image) == {}
    _boundary.set_metadata(image, {'x': 1})
    image.GetFieldData().GetAbstractArray('tomviz_metadata').SetValue(
        0, 'not json')
    assert _boundary.get_metadata(image) == {}
