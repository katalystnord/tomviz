###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Cone-beam backprojection on any GPU through WebGPU (wgpu-py).

WebGPU runs on Vulkan, Metal and DX12, so this works on NVIDIA, AMD, Intel
and Apple GPUs alike, and on software adapters such as Mesa's lavapipe.
It computes exactly what backproject_cpu does, in float32.

The volume is processed in slabs of z slices. For each slab, projections
are uploaded in batches containing only the detector rows that the slab
projects onto, so buffers stay under the device's limits and large volumes
stream through. Each dispatch is kept short, since some platforms (Windows
in particular) reset the GPU if one runs for more than about two seconds.
"""

import os

import numpy as np

SHADER = """
struct Params {
    nx: u32, ny: u32, nz: u32, n_proj: u32,
    n_u: u32, n_rows: u32, pad0: u32, pad1: u32,
}

@group(0) @binding(0) var<storage, read> proj: array<f32>;
// Three rows per projection, mapping (ix, iy, iz, 1) of this slab's voxel
// indices to homogeneous detector coordinates (u, v - first row, depth).
@group(0) @binding(1) var<storage, read> mats: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vol: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn sample(k: u32, u: f32, v: f32) -> f32 {
    let fu = floor(u);
    let fv = floor(v);
    let au = u - fu;
    let av = v - fv;
    let u0 = i32(fu);
    let v0 = i32(fv);
    let nu = i32(params.n_u);
    let nr = i32(params.n_rows);
    let base = k * params.n_u * params.n_rows;
    var acc = 0.0;
    for (var dv = 0; dv < 2; dv++) {
        let vi = v0 + dv;
        if (vi < 0 || vi >= nr) { continue; }
        let wv = select(1.0 - av, av, dv == 1);
        for (var du = 0; du < 2; du++) {
            let ui = u0 + du;
            if (ui < 0 || ui >= nu) { continue; }
            let wu = select(1.0 - au, au, du == 1);
            acc += proj[base + u32(vi) * params.n_u + u32(ui)] * wu * wv;
        }
    }
    return acc;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.nx || gid.y >= params.ny || gid.z >= params.nz) {
        return;
    }
    let p = vec4<f32>(f32(gid.x), f32(gid.y), f32(gid.z), 1.0);
    var acc = 0.0;
    for (var k = 0u; k < params.n_proj; k++) {
        let h_u = dot(mats[3u * k], p);
        let h_v = dot(mats[3u * k + 1u], p);
        let h_w = dot(mats[3u * k + 2u], p);
        let inv = 1.0 / h_w;
        acc += sample(k, h_u * inv, h_v * inv) * inv * inv;
    }
    let idx = (gid.z * params.ny + gid.y) * params.nx + gid.x;
    vol[idx] += acc;
}
"""

# Rough number of voxel updates one dispatch may do, to keep each dispatch
# well under the ~2 s watchdog of some drivers even on slow GPUs.
MAX_UPDATES_PER_DISPATCH = 1 << 29
NON_COMPUTE_BACKENDS = ('OpenGL',)


def _restrict_backends():
    """Never use the OpenGL backend for compute. This must run before the
    wgpu instance is created; if something already created it, keep going
    and filter adapters instead."""
    try:
        from wgpu.backends.wgpu_native.extras import set_instance_extras
        set_instance_extras(backends=['Vulkan', 'Metal', 'DX12'])
    except Exception:
        pass


def _choose_adapter(wgpu):
    """The adapter to use, following TOMVIZ_WGPU_ADAPTER if set. Otherwise
    a discrete GPU, then an integrated one, then a software adapter (such
    as Mesa's lavapipe or Windows' WARP): these compile the shader to
    vectorised multi-threaded CPU code and measured about ten times faster
    than the numpy backend."""
    want = os.environ.get('TOMVIZ_WGPU_ADAPTER', '').strip().lower()
    adapters = [a for a in wgpu.gpu.enumerate_adapters_sync()
                if a.info['backend_type'] not in NON_COMPUTE_BACKENDS]
    if want:
        types = {'cpu': 'CPU', 'discrete': 'DiscreteGPU',
                 'integrated': 'IntegratedGPU'}
        for a in adapters:
            if (types.get(want) == a.info['adapter_type'] or
                    want in a.info['device'].lower()):
                return a
        names = ', '.join(a.info['device'] for a in adapters) or 'none'
        raise RuntimeError(f'No WebGPU adapter matches '
                           f'TOMVIZ_WGPU_ADAPTER={want!r} (found: {names})')
    if not adapters:
        raise RuntimeError('no WebGPU adapter found')
    rank = {'DiscreteGPU': 0, 'IntegratedGPU': 1, 'CPU': 3}
    return min(adapters, key=lambda a: rank.get(a.info['adapter_type'], 2))


class WgpuBackprojector:
    def __init__(self, adapter):
        import wgpu
        self.wgpu = wgpu
        self.adapter = adapter
        alim = adapter.limits
        want = {
            'max-storage-buffer-binding-size':
                alim['max-storage-buffer-binding-size'],
            'max-buffer-size': alim['max-buffer-size'],
        }
        self.device = adapter.request_device_sync(required_limits=want)
        lim = self.device.limits
        # Stay well below the limits; some drivers struggle near them.
        self.max_binding = int(min(lim['max-storage-buffer-binding-size'],
                                   lim['max-buffer-size'], 1 << 30)) // 2
        self.max_groups = int(lim['max-compute-workgroups-per-dimension'])
        module = self.device.create_shader_module(code=SHADER)
        self.pipeline = self.device.create_compute_pipeline(
            layout='auto', compute={'module': module, 'entry_point': 'main'})
        # Reading this tiny buffer back waits for all submitted work. (The
        # queue's on_submitted_work_done is broken in some wgpu-py
        # releases, and this works everywhere.)
        self._fence = self.device.create_buffer(
            size=4, usage=wgpu.BufferUsage.COPY_SRC)
        info = adapter.info
        kind = ('software, on the CPU' if info['adapter_type'] == 'CPU'
                else info['backend_type'])
        self.description = f"WebGPU: {info['device']} ({kind})"

    def _wait(self):
        self.device.queue.read_buffer(self._fence)

    def _buffer(self, data=None, size=None, usage=0):
        wgpu = self.wgpu
        if data is not None:
            return self.device.create_buffer_with_data(
                data=data, usage=usage | wgpu.BufferUsage.STORAGE)
        return self.device.create_buffer(
            size=size, usage=usage | wgpu.BufferUsage.STORAGE)

    def backproject(self, filtered, geom, vol_geom, threads=None,
                    progress=None, should_stop=None, slab=None):
        wgpu = self.wgpu
        should_stop = should_stop or (lambda: False)
        filtered = np.ascontiguousarray(filtered, dtype=np.float32)
        n_angles, n_v, n_u = filtered.shape
        nz, ny, nx = vol_geom.shape
        P = geom.projection_matrices()

        plane = ny * nx
        if slab is None:
            slab = max(1, min(nz, self.max_binding // (4 * plane)))
        if slab * plane * 4 > self.max_binding:
            raise RuntimeError(
                f'A single {ny}x{nx} slice is too large for this GPU '
                f'({self.max_binding} bytes per buffer)')

        dz, dy, dx = vol_geom.voxel_size
        x0 = vol_geom.x_coords()[0]
        y0 = vol_geom.y_coords()[0]
        z_all = vol_geom.z_coords()

        vol = np.zeros(vol_geom.shape, dtype=np.float32)
        done = 0
        for s0 in range(0, nz, slab):
            s1 = min(nz, s0 + slab)
            nzs = s1 - s0
            # Voxel (ix, iy, iz) of this slab -> world (x, y, z, 1).
            G = np.array([[dx, 0, 0, x0],
                          [0, dy, 0, y0],
                          [0, 0, dz, z_all[s0]],
                          [0, 0, 0, 1.0]])
            M = P @ G  # (n_angles, 3, 4), in float64 until upload
            corners = np.concatenate(
                [vol_geom.slab_corners(s0, s1), np.ones((8, 1))], axis=1)
            h = np.einsum('kij,cj->kci', P, corners)
            if np.any(h[:, :, 2] <= 0):
                raise RuntimeError('Part of the volume is behind the '
                                   'source; make the volume smaller')
            vcorner = h[:, :, 1] / h[:, :, 2]
            v_lo = np.clip(np.floor(vcorner.min(1)) - 1, 0, n_v).astype(int)
            v_hi = np.clip(np.ceil(vcorner.max(1)) + 2, 0, n_v).astype(int)

            vol_buf = self._buffer(
                size=nzs * plane * 4,
                usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
            vol_buf_zero = np.zeros(nzs * plane, dtype=np.float32)
            self.device.queue.write_buffer(vol_buf, 0, vol_buf_zero)
            del vol_buf_zero

            k = 0
            while k < n_angles:
                if should_stop():
                    return None
                # Grow the batch while its row window fits in a buffer
                # and the dispatch stays short.
                r0, r1 = v_lo[k], v_hi[k]
                k1 = k + 1
                max_batch = max(1, MAX_UPDATES_PER_DISPATCH //
                                (nzs * plane))
                while k1 < n_angles and k1 - k < max_batch:
                    nr0, nr1 = min(r0, v_lo[k1]), max(r1, v_hi[k1])
                    if (k1 + 1 - k) * (nr1 - nr0) * n_u * 4 > \
                            self.max_binding:
                        break
                    r0, r1 = nr0, nr1
                    k1 += 1
                if r1 > r0:
                    self._dispatch(filtered[k:k1, r0:r1], M[k:k1], r0,
                                   vol_buf, nx, ny, nzs, n_u)
                k = k1
                # Wait so cancel stays responsive and work cannot pile up.
                self._wait()

            data = self.device.queue.read_buffer(vol_buf)
            vol[s0:s1] = np.frombuffer(data, dtype=np.float32).reshape(
                nzs, ny, nx)
            vol_buf.destroy()
            done += nzs
            if progress is not None:
                progress(done, nz)
        if should_stop():
            return None
        return vol

    def _dispatch(self, rows, M, r0, vol_buf, nx, ny, nzs, n_u):
        wgpu = self.wgpu
        n_proj, n_rows, _ = rows.shape
        mats = M.copy()
        # Shift v so it indexes the uploaded rows: v' = v - r0.
        mats[:, 1, :] -= r0 * mats[:, 2, :]
        mats = np.ascontiguousarray(mats, dtype=np.float32)
        params = np.array([nx, ny, nzs, n_proj, n_u, n_rows, 0, 0],
                          dtype=np.uint32)
        proj_buf = self._buffer(np.ascontiguousarray(rows))
        mat_buf = self._buffer(mats)
        par_buf = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM)
        bind = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {'binding': 0, 'resource': {'buffer': proj_buf}},
                {'binding': 1, 'resource': {'buffer': mat_buf}},
                {'binding': 2, 'resource': {'buffer': vol_buf}},
                {'binding': 3, 'resource': {'buffer': par_buf}},
            ])
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.pipeline)
        cp.set_bind_group(0, bind)
        cp.dispatch_workgroups(-(-nx // 8), -(-ny // 8), -(-nzs // 4))
        cp.end()
        self.device.queue.submit([enc.finish()])
        proj_buf.destroy()
        mat_buf.destroy()
        par_buf.destroy()

    def self_test(self):
        """Backproject a tiny problem and compare with the CPU path."""
        from . import backproject_cpu
        from .geometry import ConeGeometry, VolumeGeometry
        g = ConeGeometry.circular(np.arange(0, 360, 30), 50, 100, 1.0, 1.0,
                                  12, 10, offset_u=0.7)
        vg = VolumeGeometry((6, 7, 8), 0.4)
        rng = np.random.default_rng(0)
        f = rng.random((g.n_angles, 10, 12), dtype=np.float32)
        ref = backproject_cpu.backproject(f, g, vg, threads=1)
        got = self.backproject(f, g, vg, slab=4)
        if not np.allclose(got, ref, rtol=1e-4, atol=1e-4 * abs(ref).max()):
            raise RuntimeError('GPU self-test gave wrong results '
                               f'(max error {abs(got - ref).max():.3g})')


_cached = None


def make_backend():
    """Create the WebGPU backend once per process, or raise with a reason
    if it cannot be used."""
    global _cached
    if _cached is not None:
        return _cached
    try:
        import wgpu
    except ImportError:
        raise RuntimeError('the wgpu package is not installed '
                           '(pip install wgpu)')
    _restrict_backends()
    from .backend import Backend
    bp = WgpuBackprojector(_choose_adapter(wgpu))
    bp.self_test()
    _cached = Backend('gpu', bp.description, bp.backproject)
    return _cached
