"""Parallel-beam projection with a sparse matrix per slice (the CPU path
of SIRT without a GPU), and the shared Siddon matrix of the ART, SIRT and
TV minimisation operators."""

import numpy as np
import pytest

from utils import load_operator_class, load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset

from tomviz.projectors import ConeGeometry, ParallelGeometry, VolumeGeometry
from tomviz.projectors import sparse
from tomviz.projectors.forward_cpu import forward_project
from tomviz.projectors.phantom import EllipsoidPhantom
from tomviz.projectors.sirt import sirt

RNG = np.random.default_rng(7)


def level(angles=np.arange(0, 180, 6.0), n_u=40, n_v=12, pitch=0.5,
          **kw):
    return ParallelGeometry.circular(angles, pitch, pitch, n_u, n_v, **kw)


CASES = {
    # rows on the slice grid
    'aligned': (level(), VolumeGeometry((12, 30, 30), 0.5)),
    # rows between slices, non-square slices, an off-centre volume and
    # axis, the other rotation direction
    'general': (level(offset_u=2.3, offset_v=0.4, clockwise=True),
                VolumeGeometry((9, 26, 33), (0.61, 0.5, 0.45),
                               center=(0.2, -0.4, 0.3))),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_joseph_matrix_projects_as_forward_cpu(name):
    g, vg = CASES[name]
    x = RNG.random(vg.shape, dtype=np.float32)
    got = sparse.SliceProjector(g, vg, threads=3).forward(x)
    ref = forward_project(x, g, vg, threads=2)
    np.testing.assert_allclose(got, ref, atol=1e-5 * abs(ref).max())


@pytest.mark.parametrize('matrix', sparse.MATRICES)
def test_back_is_the_exact_transpose(matrix):
    g = level(n_v=10)
    vg = VolumeGeometry((8, 30, 30), 0.5)
    P = sparse.SliceProjector(g, vg, matrix, threads=4)
    x = RNG.random(vg.shape)
    p = RNG.random((g.n_angles, g.n_v, g.n_u))
    lhs = float(np.vdot(P.forward(x), p))
    rhs = float(np.vdot(x, P.back(p)))
    assert lhs == pytest.approx(rhs, rel=1e-5)


def test_siddon_and_joseph_agree_on_a_smooth_volume():
    # Away from the axes. Within a few degrees of them the Siddon matrix
    # reads the nearest pixel row instead of interpolating: 3 % from the
    # true line integrals at 2 degrees here, against 0.2 % for Joseph.
    angles = np.arange(-90, 90, 4.0)
    angles = angles[np.abs(np.mod(angles + 45, 90) - 45) > 5]
    g = level(angles, n_u=48, n_v=6, pitch=1.0)
    vg = VolumeGeometry((6, 48, 48), 1.0)
    z, y, x = np.mgrid[0:6, 0:48, 0:48]
    vol = (np.exp(-(((x - 30) / 6.0) ** 2 + ((y - 18) / 9.0) ** 2)) +
           0.5 * np.exp(-(((x - 14) / 4.0) ** 2 + ((y - 31) / 5.0) ** 2))
           ) * (1 + 0.1 * z)
    pj = sparse.SliceProjector(g, vg, 'joseph').forward(vol)
    ps = sparse.SliceProjector(g, vg, 'siddon').forward(vol)
    # Path lengths against interpolation: close, not equal.
    assert np.abs(ps - pj).max() < 0.005 * np.abs(pj).max()


def test_what_cannot_go_slice_by_slice():
    vg = VolumeGeometry((8, 16, 16), 1.0)
    cone = ConeGeometry.circular(np.arange(0, 360, 30.0), 50, 100, 1, 1,
                                 16, 8)
    assert 'parallel' in sparse.unsuitable(cone, vg)
    tilted = level(n_u=16, n_v=8, pitch=1.0)
    tilted.ray = tilted.ray + [0, 0, 0.01]
    assert 'perpendicular' in sparse.unsuitable(tilted, vg)
    off = level(n_u=16, n_v=8, pitch=1.0, offset_u=1.5)
    assert sparse.unsuitable(off, vg) is None
    assert 'detector centre' in sparse.unsuitable(off, vg, 'siddon')
    assert 'GB' in sparse.unsuitable(level(n_u=16, n_v=8, pitch=1.0), vg,
                                     memory_bytes=1000)


@pytest.fixture(scope='module')
def scan():
    ph = EllipsoidPhantom.shepp_logan(radius=7.0)
    g = level(np.arange(0, 180, 4.0), n_u=40, n_v=16, pitch=0.4)
    vg = VolumeGeometry((16, 36, 36), 0.4)
    return g, vg, ph.project(g).astype(np.float32), ph.voxelize(
        vg, supersample=2)


@pytest.mark.parametrize('matrix', sparse.MATRICES)
def test_sirt_without_a_gpu_goes_slice_by_slice(scan, matrix):
    g, vg, proj, truth = scan
    notes = []
    x = sirt(proj, g, vg, iterations=30, backend='cpu', cpu_matrix=matrix,
             log=notes.append)
    assert any(f'{matrix} matrix per slice' in n for n in notes)
    # At least as good as SIRT with the numpy projectors (measured: RMSE
    # 0.052 Joseph, 0.050 Siddon, 0.054 numpy), and close to it (that is
    # an unmatched pair; the sparse matrix and its transpose are matched).
    ref = sirt(proj, g, vg, iterations=30, backend='cpu', cpu_matrix='none')

    def rmse(a):
        return float(np.sqrt(np.mean((a - truth) ** 2)))

    assert rmse(x) <= rmse(ref)
    assert float(np.sqrt(np.mean((x - ref) ** 2))) < 0.03


def test_sirt_falls_back_when_the_matrix_does_not_fit(scan):
    g, vg, proj, _ = scan
    notes = []
    sirt(proj, g, vg, iterations=1, backend='cpu', cpu_memory_bytes=1000,
         log=notes.append)
    assert any('No sparse matrix per slice' in n for n in notes)
    with pytest.raises(ValueError, match='cpu_matrix'):
        sirt(proj, g, vg, iterations=1, backend='cpu', cpu_matrix='astra')


@pytest.mark.parametrize('name,kw', [
    ('Recon_SIRT', dict(Niter=2)),
    ('Recon_ART', dict(Niter=1)),
    ('Recon_TV_minimization', dict(Niter=1)),
])
def test_operators_use_the_shared_siddon_matrix(name, kw):
    # They import parallelRay from tomviz.projectors.siddon now.
    angles = np.arange(-60, 61, 15.0)
    vol = np.zeros((3, 16, 16), np.float32)
    vol[:, 5:11, 6:10] = 1
    tilt = np.zeros((3, 16, len(angles)), np.float32, order='F')
    for k, a in enumerate(angles):
        import scipy.ndimage
        tilt[:, :, k] = scipy.ndimage.rotate(vol, a, axes=(1, 2),
                                             reshape=False,
                                             order=1).sum(axis=2)
    ds = Dataset({'scalars': tilt})
    ds.tilt_angles = angles
    op = load_operator_class(load_operator_module(name))
    out = op.transform(ds, **kw)
    recon = out['reconstruction'].active_scalars
    assert recon.shape == (3, 16, 16) and np.all(np.isfinite(recon))
    assert recon.max() > 0
