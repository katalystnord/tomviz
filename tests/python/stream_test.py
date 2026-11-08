"""FDK streamed from and to files, a slab at a time, against the
in-memory FDK."""

import numpy as np
import pytest

from tomviz.conebeam import fdk
from tomviz.conebeam.stream import (RawStack, crop_rows, fdk_stream,
                                    plan_slabs)
from tomviz.projectors import ConeGeometry, VolumeGeometry
from tomviz.projectors.phantom import SHEPP_LOGAN_3D, EllipsoidPhantom


@pytest.fixture(scope='module')
def scan():
    phantom = EllipsoidPhantom(SHEPP_LOGAN_3D, radius=9.0)
    g = ConeGeometry.circular(np.arange(0, 360, 3.0), 60.0, 120.0, 0.6, 0.6,
                              48, 40, offset_u=1.3, offset_v=-0.7)
    return g, VolumeGeometry((30, 32, 32), 0.6), phantom.project(g)


def test_cropped_geometry_maps_like_the_full_one(scan):
    g, _, _ = scan
    c = crop_rows(g, 7, 29)
    pts = np.random.default_rng(0).uniform(-8, 8, (20, 3))
    h = np.concatenate([pts, np.ones((20, 1))], axis=1)
    full = np.einsum('kij,cj->kci', g.projection_matrices(), h)
    part = np.einsum('kij,cj->kci', c.projection_matrices(), h)
    np.testing.assert_allclose(part[..., 0] / part[..., 2],
                               full[..., 0] / full[..., 2], atol=1e-9)
    np.testing.assert_allclose(part[..., 1] / part[..., 2],
                               full[..., 1] / full[..., 2] - 7, atol=1e-9)


@pytest.mark.parametrize('angles', ['full', 'short'])
def test_streamed_equals_in_memory(scan, tmp_path, angles):
    g, vg, proj = scan
    if angles == 'short':  # 180 degrees plus the fan: Parker weights
        keep = np.flatnonzero(np.arange(0, 360, 3.0) <= 228)
        g = ConeGeometry.from_astra_vectors(np.concatenate(
            [g.source, g.det_center, g.det_u, g.det_v], axis=1)[keep],
            g.n_u, g.n_v)
        proj = proj[keep]
    np.save(tmp_path / 'proj.npy', np.ascontiguousarray(proj, np.float32))
    src = RawStack.npy(str(tmp_path / 'proj.npy'))
    out = RawStack(str(tmp_path / 'vol.raw'), vg.shape, mode='w')
    # A tiny memory budget: many slabs.
    slabs = plan_slabs(g, vg, memory_bytes=400_000)
    assert len(slabs) >= 4
    fdk_stream(src, g, vg, out, backend='cpu', memory_bytes=400_000)
    src.close()
    out.close()
    streamed = np.fromfile(tmp_path / 'vol.raw', np.float32).reshape(
        vg.shape)
    reference = fdk(proj, g, vg, backend='cpu')
    np.testing.assert_allclose(streamed, reference,
                               atol=1e-5 * np.abs(reference).max())


def test_streamed_on_the_gpu(scan, tmp_path):
    pytest.importorskip('wgpu')
    from tomviz.projectors.backend import get_backend
    try:
        be = get_backend('gpu')
    except RuntimeError as e:
        pytest.skip(str(e))
    g, vg, proj = scan
    np.save(tmp_path / 'proj.npy', np.ascontiguousarray(proj, np.float32))
    src = RawStack.npy(str(tmp_path / 'proj.npy'))
    out = RawStack(str(tmp_path / 'vol.raw'), vg.shape, mode='w')
    fdk_stream(src, g, vg, out, backend=be, memory_bytes=400_000)
    src.close()
    out.close()
    streamed = np.fromfile(tmp_path / 'vol.raw', np.float32).reshape(
        vg.shape)
    reference = fdk(proj, g, vg, backend='cpu')
    np.testing.assert_allclose(streamed, reference,
                               atol=1e-4 * np.abs(reference).max())


def test_streams_into_an_emd_file_tomviz_opens(scan, tmp_path):
    from tomviz.conebeam.operator_support import set_volume
    from tomviz.conebeam.stream import create_emd
    from tomviz_pipeline.dataset import Dataset
    from tomviz_pipeline.io import load_dataset
    g, vg, proj = scan
    np.save(tmp_path / 'proj.npy', np.ascontiguousarray(proj, np.float32))
    src = RawStack.npy(str(tmp_path / 'proj.npy'))
    out = create_emd(str(tmp_path / 'vol.emd'), vg)
    fdk_stream(src, g, vg, out, backend='cpu', memory_bytes=400_000)
    src.close()
    out.close()
    opened = load_dataset(tmp_path / 'vol.emd')
    # Laid out as the FDK operator gives the volume to Tomviz.
    expected = Dataset({'FDK': np.zeros((1, 1, 1), np.float32)}, 'FDK')
    set_volume(expected, fdk(proj, g, vg, backend='cpu'), vg)
    np.testing.assert_allclose(opened.active_scalars,
                               expected.active_scalars,
                               atol=1e-5 * np.abs(expected.active_scalars)
                               .max())
    np.testing.assert_allclose(opened.spacing, expected.spacing, rtol=1e-6)


@pytest.mark.parametrize('platform', ['preadv', 'pread', 'seek'])
def test_positioned_io_on_every_platform(scan, tmp_path, monkeypatch,
                                         platform):
    # Linux has preadv, other Unix and macOS only pread and pwrite,
    # Windows neither: the same reconstruction through each.
    import os
    if platform != 'preadv':
        monkeypatch.delattr(os, 'preadv', raising=False)
    if platform == 'seek':
        monkeypatch.delattr(os, 'pread', raising=False)
        monkeypatch.delattr(os, 'pwrite', raising=False)
    g, vg, proj = scan
    np.save(tmp_path / 'proj.npy', np.ascontiguousarray(proj, np.float32))
    src = RawStack.npy(str(tmp_path / 'proj.npy'))
    out = RawStack(str(tmp_path / 'vol.raw'), vg.shape, mode='w')
    fdk_stream(src, g, vg, out, backend='cpu', memory_bytes=400_000)
    src.close()
    out.close()
    streamed = np.fromfile(tmp_path / 'vol.raw', np.float32).reshape(
        vg.shape)
    reference = fdk(proj, g, vg, backend='cpu')
    np.testing.assert_allclose(streamed, reference,
                               atol=1e-5 * np.abs(reference).max())


def test_partial_reads_are_continued(tmp_path, monkeypatch):
    import os
    from tomviz.conebeam.stream import read_at
    data = np.arange(1000, dtype=np.float32)
    data.tofile(tmp_path / 'd.raw')
    real = os.preadv

    def stingy(fd, buffers, offset):   # at most 100 bytes a call
        return real(fd, [buffers[0][:100]], offset)

    monkeypatch.setattr(os, 'preadv', stingy, raising=False)
    out = np.empty(1000, np.float32)
    fd = os.open(tmp_path / 'd.raw', os.O_RDONLY)
    try:
        read_at(fd, memoryview(out).cast('B'), 0)
        with pytest.raises(IOError, match='end of file'):
            read_at(fd, memoryview(np.empty(10, np.float32)).cast('B'),
                    3990)
    finally:
        os.close(fd)
    np.testing.assert_array_equal(out, data)
