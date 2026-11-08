"""Reconstructing scans on disk into files: row sources for TIFF and HDF5
frames, line integrals computed in the stream, and the command line."""

import numpy as np
import pytest

from tomviz.conebeam import fdk
from tomviz.conebeam.reconstruct_file import main, read_preview
from tomviz.conebeam.scan import open_scan
from tomviz.conebeam.sources import (FrameFiles, HDF5Frames, LineIntegrals,
                                     open_beam_levels)
from tomviz.projectors import ConeGeometry
from tomviz.projectors.phantom import SHEPP_LOGAN_3D, EllipsoidPhantom

I0 = 50000.0
GEOMETRY = ['--sod', '60', '--sdd', '120', '--pixel-size', '0.6',
            '--offset-u', '1.3', '--offset-v', '-0.7', '--backend', 'cpu',
            '--memory-gb', '0.0004']


@pytest.fixture(scope='module')
def scan():
    phantom = EllipsoidPhantom(SHEPP_LOGAN_3D, radius=9.0)
    g = ConeGeometry.circular(np.arange(0, 360, 3.0), 60.0, 120.0, 0.6, 0.6,
                              48, 40, offset_u=1.3, offset_v=-0.7)
    proj = phantom.project(g).astype(np.float32)
    # What a detector records: counts, rounded to integers.
    counts = np.round(I0 * np.exp(-proj)).astype(np.uint16)
    # The line integrals those counts hold, which FDK should see.
    seen = -np.log(counts.astype(np.float32) / np.float32(I0))
    return g, counts, seen


def reference(g, seen):
    from tomviz.conebeam.operator_support import build_volume
    return fdk(seen, g, build_volume(g), backend='cpu')


def emd_volume(path):
    """The volume back in (z, y, x), as fdk() returns it."""
    import h5py
    with h5py.File(path, 'r') as f:
        return f['data/tomography/data'][()].transpose(0, 2, 1)


def close(a, b, tol=1e-4):
    np.testing.assert_allclose(a, b, atol=tol * np.abs(b).max())


def test_tiff_folder_with_an_open_beam_level(scan, tmp_path):
    import tifffile
    g, counts, seen = scan
    folder = tmp_path / 'proj'
    folder.mkdir()
    for k, frame in enumerate(counts):
        # Some frames big-endian, as some scanners write them.
        tifffile.imwrite(folder / f'p{k:04d}.tif', frame,
                         byteorder='>' if k % 3 == 0 else '<')
    src = FrameFiles(sorted(folder.iterdir()))
    assert src.shape == counts.shape
    np.testing.assert_array_equal(src.read_rows(5, 17), counts[:, 5:17])
    out = tmp_path / 'vol.emd'
    assert main([str(folder), str(out), '--i0', str(I0)] + GEOMETRY) == 0
    close(emd_volume(out), reference(g, seen))


def test_compressed_multipage_tiff(scan, tmp_path):
    import tifffile
    g, counts, seen = scan
    path = tmp_path / 'stack.tif'
    tifffile.imwrite(path, counts, compression='zlib')
    src = FrameFiles([path])
    assert src.shape == counts.shape
    np.testing.assert_array_equal(src.read_rows(0, 40), counts)
    out = tmp_path / 'vol.raw'
    assert main([str(path), str(out), '--i0', str(I0)] + GEOMETRY) == 0
    vol = np.fromfile(out, np.float32).reshape(reference(g, seen).shape)
    close(vol, reference(g, seen))


def test_data_exchange_with_flats_and_darks(scan, tmp_path):
    import h5py
    g, counts, _ = scan
    rng = np.random.default_rng(1)
    gain = rng.uniform(0.8, 1.2, counts.shape[1:]).astype(np.float32)
    dark = np.float32(100)
    raw = counts * gain + dark
    path = tmp_path / 'scan.h5'
    with h5py.File(path, 'w') as f:
        f['exchange/data'] = raw
        f['exchange/data_white'] = np.stack([I0 * gain + dark] * 3)
        f['exchange/data_dark'] = np.full((2,) + counts.shape[1:], dark,
                                          np.float32)
    s = open_scan(path)
    assert s.kind == 'Data Exchange' and s.flat is not None
    out = tmp_path / 'vol.emd'
    assert main([str(path), str(out), '--beam', 'cone'] + GEOMETRY) == 0
    expected = -np.log(((raw - dark) / (I0 * gain)).astype(np.float32))
    close(emd_volume(out), reference(g, expected))


def test_nxtomo_projections_among_flats_and_darks(scan, tmp_path):
    pytest.importorskip('tomviz.io.formats.nxtomo')
    import h5py
    g, counts, seen = scan
    n = len(counts)
    frames = np.concatenate([np.full((2,) + counts.shape[1:], 7, np.uint16),
                             np.full((3,) + counts.shape[1:], I0 + 7,
                                     np.uint16),
                             counts + 7])
    keys = np.array([2, 2, 1, 1, 1] + [0] * n)
    path = tmp_path / 'scan.nxs'
    with h5py.File(path, 'w') as f:
        e = f.create_group('entry')
        e.attrs['NX_class'] = 'NXentry'
        e['definition'] = 'NXtomo'
        det = e.create_group('instrument/detector')
        det['data'] = frames
        det['image_key'] = keys
        a = e.create_group('sample').create_dataset(
            'rotation_angle', data=np.r_[np.zeros(5), np.arange(n) * 3.0])
        a.attrs['units'] = 'degree'
    s = open_scan(path)
    assert s.kind == 'NXtomo' and s.shape == counts.shape
    np.testing.assert_allclose(s.angles, np.arange(n) * 3.0)
    out = tmp_path / 'vol.emd'
    assert main([str(path), str(out), '--beam', 'cone'] + GEOMETRY) == 0
    close(emd_volume(out), reference(g, seen))


def test_open_beam_estimated_from_the_brightest_pixels(scan):
    _, counts, _ = scan

    class Memory:
        shape = counts.shape

        def read_rows(self, r0, r1):
            return counts[:, r0:r1].astype(np.float32)

    levels = open_beam_levels(Memory())
    # The detector's edges see the open beam.
    np.testing.assert_allclose(levels, I0, rtol=1e-3)
    t = LineIntegrals(Memory(), levels).read_rows(0, 40)
    assert np.all(np.isfinite(t)) and abs(np.median(t[:, :, :3])) < 1e-3


def test_hdf5_rows_of_selected_frames(tmp_path):
    import h5py
    data = np.arange(6 * 5 * 4, dtype=np.uint16).reshape(6, 5, 4)
    with h5py.File(tmp_path / 'f.h5', 'w') as f:
        f['d'] = data
    for index in ([1, 2, 3], [0, 2, 5]):
        src = HDF5Frames(tmp_path / 'f.h5', 'd', index)
        np.testing.assert_array_equal(src.read_rows(1, 4),
                                      data[index, 1:4])
        src.close()


def test_preview_is_strided_with_scaled_spacing(scan, tmp_path):
    from tomviz.conebeam.stream import create_emd
    from tomviz.projectors import VolumeGeometry
    vg = VolumeGeometry((20, 12, 10), (0.5, 0.25, 0.2))
    out = create_emd(str(tmp_path / 'v.emd'), vg)
    vol = np.random.default_rng(0).random(vg.shape, np.float32)
    out.write_slab(0, vol)
    out.close()
    data, spacing, stride = read_preview(tmp_path / 'v.emd',
                                         max_voxels=400)
    assert stride == 2   # 2400 voxels to at most 400: (2400 / 400)^(1/3)
    # Tomviz's (x, y, z) = our (z, x, y)
    np.testing.assert_array_equal(data, vol.transpose(0, 2, 1)[::2, ::2,
                                                               ::2])
    np.testing.assert_allclose(spacing, [1.0, 0.4, 0.5])


def test_tomviz_source_node_writes_the_file_and_shows_a_preview(scan,
                                                                tmp_path):
    import importlib.util
    import json
    import tifffile
    from tomviz_pipeline._internal import OperatorWrapper
    from utils import OPERATOR_PATH
    name = 'ReconstructLargeScanToFile'
    spec = importlib.util.spec_from_file_location(
        name, OPERATOR_PATH / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = json.loads((OPERATOR_PATH / f'{name}.json').read_text())
    node = getattr(module, name)()
    node._operator_wrapper = OperatorWrapper()
    node._parameter_spec = {p['name']: p for p in description['parameters']}
    params = {}
    for p in description['parameters']:
        value = p.get('default')
        if p['type'] == 'enumeration':   # the runtime passes the value
            value = list(p['options'][value].values())[0]
        params[p['name']] = value
    assert params['intensity'] == 'auto' and params['backend'] == 'auto'

    g, counts, seen = scan
    folder = tmp_path / 'proj'
    folder.mkdir()
    for k, frame in enumerate(counts):
        tifffile.imwrite(folder / f'p{k:04d}.tif', frame)
    params.update(scan_file=str(folder / 'p0000.tif'), whole_folder=True,
                  output_file=str(tmp_path / 'vol'), i0=I0, sod=60.0,
                  sdd=120.0, pixel_size=0.6, offset_u=1.3, offset_v=-0.7,
                  backend='cpu', memory_gb=0.0004, preview_voxels=16)
    result = node.produce(**params)
    full = emd_volume(tmp_path / 'vol.emd')
    close(full, reference(g, seen))
    shown = result['volume']
    s = shown.metadata['preview_stride']
    assert s > 1 and max(shown.active_scalars.shape) <= 16 * 1.5
    np.testing.assert_allclose(shown.active_scalars,
                               full.transpose(0, 2, 1)[::s, ::s, ::s])


@pytest.mark.parametrize('method', ['fbp', 'sirt'])
def test_parallel_beam_data_exchange_with_the_axis_found(tmp_path, method):
    """A synchrotron-like scan: parallel beam over 180 degrees (both ends),
    the rotation axis off centre, flats and darks; reconstructed with the
    axis found from opposite projections."""
    import h5py
    from tomviz.projectors import ParallelGeometry, VolumeGeometry
    from tomviz.projectors.backend import get_backend
    from tomviz.projectors.sirt import sirt
    phantom = EllipsoidPhantom.shepp_logan(radius=5.0)
    angles = np.arange(0, 180.1, 2.0)
    offset = 2.6
    g = ParallelGeometry.circular(angles, 0.25, 0.25, 64, 40,
                                  offset_u=offset)
    proj = phantom.project(g)
    gain = np.random.default_rng(2).uniform(0.9, 1.1, (40, 64))
    raw = (I0 * gain * np.exp(-proj) + 100).astype(np.float32)
    path = tmp_path / 'scan.h5'
    with h5py.File(path, 'w') as f:
        f['exchange/data'] = raw
        f['exchange/data_white'] = (I0 * gain + 100)[None].astype(
            np.float32)
        f['exchange/data_dark'] = np.full((1, 40, 64), 100, np.float32)
        f['exchange/theta'] = angles
    out = tmp_path / 'vol.emd'
    printed = []
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: printed.append(' '.join(map(str, a)))
    try:
        assert main([str(path), str(out), '--find-axis', '--pixel-size',
                     '0.25', '--method', method, '--iterations', '20',
                     '--memory-gb', '0.0002']) == 0
    finally:
        builtins.print = real_print
    assert any('parallel beam' in line for line in printed)
    found = [line for line in printed if 'rotation axis' in line][0]
    assert abs(float(found.split()[2]) - offset) < 0.1, found
    vol = emd_volume(out)
    seen = -np.log(((raw - 100) / (I0 * gain)).astype(np.float32))
    vg = VolumeGeometry(vol.shape, 0.25)
    if method == 'fbp':
        expected = fdk(seen, g, vg)
    else:
        expected = sirt(seen, g, vg, iterations=20,
                        backend=get_backend('auto'))
    # The axis is found to a few hundredths of a pixel, not exactly.
    rel = np.sqrt(np.mean((vol - expected) ** 2)) / np.abs(expected).max()
    assert rel < 0.01, rel


def test_integer_counts_in_npy_and_raw(scan, tmp_path):
    # RawStack returns the file's own dtype: uint16 counts must not be
    # corrected in place (that raised a casting error).
    g, counts, seen = scan
    np.save(tmp_path / 'counts.npy', counts)
    counts.tofile(tmp_path / 'counts.raw')
    ref = reference(g, seen)
    out = tmp_path / 'a.emd'
    assert main([str(tmp_path / 'counts.npy'), str(out), '--i0', str(I0)] +
                GEOMETRY) == 0
    close(emd_volume(out), ref)
    out = tmp_path / 'b.emd'
    assert main([str(tmp_path / 'counts.raw'), str(out), '--raw-shape',
                 *map(str, counts.shape), '--raw-dtype', 'uint16'] +
                GEOMETRY) == 0   # open beam estimated from the counts
    vol = emd_volume(out)
    assert np.isfinite(vol).all()
    assert np.corrcoef(vol.ravel(), ref.ravel())[0, 1] > 0.99


@pytest.mark.parametrize('angles,offset', [
    (np.arange(0, 180.1, 1.0), 3.3),     # both ends: exact pairs
    (np.arange(0, 180, 1.0), 3.3),       # one step short of 180
    (np.arange(0, 180, 0.6), -7.6),
    (np.arange(0, 360, 1.0), 5.2),
])
def test_axis_found_with_or_without_exact_opposite_views(angles, offset):
    from tomviz.conebeam.scan import axis_offset_from_opposite
    from tomviz.projectors import ParallelGeometry
    phantom = EllipsoidPhantom.shepp_logan(radius=10.0)
    g = ParallelGeometry.circular(angles, 0.25, 0.25, 128, 48,
                                  offset_u=offset)
    proj = phantom.project(g).astype(np.float32)

    class Memory:
        shape = proj.shape

        def read_rows(self, r0, r1):
            return proj[:, r0:r1]

    assert axis_offset_from_opposite(Memory(), angles) == \
        pytest.approx(offset, abs=0.1)


def test_axis_needs_a_view_near_180_degrees_on():
    from tomviz.conebeam.scan import axis_offset_from_opposite

    class Memory:
        shape = (140, 8, 32)

        def read_rows(self, r0, r1):
            return np.ones((140, r1 - r0, 32), np.float32)

    with pytest.raises(RuntimeError, match='180 degrees apart'):
        axis_offset_from_opposite(Memory(), np.arange(10, 150, 1.0))


def test_given_offset_used_when_the_axis_cannot_be_found(tmp_path):
    import builtins
    import h5py
    from tomviz.projectors import ParallelGeometry
    phantom = EllipsoidPhantom.shepp_logan(radius=5.0)
    angles = np.arange(10, 150, 2.0)      # no view near 180 degrees on
    g = ParallelGeometry.circular(angles, 0.25, 0.25, 64, 24,
                                  offset_u=1.5)
    path = tmp_path / 'scan.h5'
    with h5py.File(path, 'w') as f:
        f['exchange/data'] = phantom.project(g).astype(np.float32)
        f['exchange/theta'] = angles
    printed = []
    real = builtins.print
    builtins.print = lambda *a, **k: printed.append(' '.join(map(str, a)))
    try:
        assert main([str(path), str(tmp_path / 'v.emd'), '--find-axis',
                     '--offset-u', '1.5', '--intensity', 'none',
                     '--pixel-size', '0.25', '--memory-gb', '0.0002']) == 0
    finally:
        builtins.print = real
    assert any('not found' in line and '+1.50' in line for line in printed)
