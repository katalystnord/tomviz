###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Backprojection and forward projection on any GPU through WebGPU
(wgpu-py), for cone and parallel beams.

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
        // Parallel beams land exactly on rows: skip the unused one.
        if (wv == 0.0) { continue; }
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

FORWARD_SHADER = """
struct Params {
    n_u: u32, n_v: u32, n_proj: u32, nx: u32,
    ny: u32, nz: u32, z_first: i32, nzb: i32,
    own0: i32, own1: i32, pad0: u32, pad1: u32,
    sx: f32, sy: f32, sz: f32, pad2: f32,
}

@group(0) @binding(0) var<storage, read> vol: array<f32>;
// Six rows per projection, in voxel index coordinates: ray origin o0, o_u,
// o_v and direction d0, d_u, d_v; origin = o0 + iu o_u + iv o_v, etc.
@group(0) @binding(1) var<storage, read> rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// One axis of the uploaded slab: flat-index stride, number of slices in
// the buffer, and the global index of its first slice.
struct Axis { stride: i32, size: i32, first: i32, }

fn axis(i: u32) -> Axis {
    let nx = i32(params.nx);
    if (i == 0u) { return Axis(1, nx, 0); }
    if (i == 1u) { return Axis(nx, i32(params.ny), 0); }
    return Axis(nx * i32(params.ny), params.nzb, params.z_first);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n_u || gid.y >= params.n_v ||
        gid.z >= params.n_proj) {
        return;
    }
    let fu = f32(gid.x);
    let fv = f32(gid.y);
    let r = 6u * gid.z;
    let o = (rays[r] + fu * rays[r + 1u] + fv * rays[r + 2u]).xyz;
    let d = (rays[r + 3u] + fu * rays[r + 4u] + fv * rays[r + 5u]).xyz;
    let ad = abs(d);
    // Dominant axis a and the other two axes b < c. When a is x or y, c
    // is z.
    var a = 0u; var b = 1u; var c = 2u;
    var oa = o.x; var ob = o.y; var oc = o.z;
    var da = d.x; var db = d.y; var dc = d.z;
    var na = i32(params.nx);
    if (ad.y > ad.x && ad.y >= ad.z) {
        a = 1u; b = 0u;
        oa = o.y; ob = o.x; da = d.y; db = d.x; na = i32(params.ny);
    } else if (ad.z > ad.x && ad.z > ad.y) {
        a = 2u; b = 0u; c = 1u;
        oa = o.z; ob = o.x; oc = o.y; da = d.z; db = d.x; dc = d.y;
        na = i32(params.nz);
    }
    let A = axis(a);
    let B = axis(b);
    let C = axis(c);
    var p_lo = 0;
    var p_hi = na - 1;
    if (a == 2u) {
        p_lo = max(p_lo, params.own0);
        p_hi = min(p_hi, params.own1 - 1);
    } else {
        // Only planes where the ray's z lies in this slab's own range.
        let rz = dc / da;
        if (abs(rz) > 1e-9) {
            let pa = oa + (f32(params.own0) - oc) / rz;
            let pb = oa + (f32(params.own1) - oc) / rz;
            // Clamp in float before converting: for a ray almost parallel
            // to the slab, pa and pb can exceed the i32 range, and drivers
            // differ in how they convert that (RADV saturates, so the + 1
            // wrapped and the ray was dropped). The per-sample ownership
            // test below keeps the result exact either way.
            let lim = f32(na);
            p_lo = max(p_lo, i32(clamp(floor(min(pa, pb)) - 1.0, -1.0, lim)));
            p_hi = min(p_hi, i32(clamp(ceil(max(pa, pb)) + 1.0, -1.0, lim)));
        }
    }
    var acc = 0.0;
    for (var p = p_lo; p <= p_hi; p++) {
        let t = (f32(p) - oa) / da;
        let qb = ob + t * db;
        let qc = oc + t * dc;
        let fb = floor(qb);
        let fc = floor(qc);
        let c0 = i32(fc);
        if (a != 2u) {
            // Each sample belongs to the slab holding floor(z).
            if (c0 < params.own0 || c0 >= params.own1) { continue; }
        }
        let wb1 = qb - fb;
        let wc1 = qc - fc;
        // Indices into the uploaded slab; p is always inside it.
        let b0 = i32(fb) - B.first;
        let cl = c0 - C.first;
        let base = (p - A.first) * A.stride;
        for (var i = 0; i < 2; i++) {
            let wb = select(1.0 - wb1, wb1, i == 1);
            let bi = b0 + i;
            if (wb == 0.0 || bi < 0 || bi >= B.size) { continue; }
            for (var j = 0; j < 2; j++) {
                let wc = select(1.0 - wc1, wc1, j == 1);
                let ci = cl + j;
                if (wc == 0.0 || ci < 0 || ci >= C.size) { continue; }
                acc += vol[base + bi * B.stride + ci * C.stride] * wb * wc;
            }
        }
    }
    let scale = vec3<f32>(params.sx, params.sy, params.sz);
    let step = length(d * scale) / abs(da);
    out[(gid.z * params.n_v + gid.y) * params.n_u + gid.x] = acc * step;
}
"""

# Element-wise steps of SIRT, so its whole loop can stay on the GPU.
# residual: a = w * (c - a), with a = A x, w = R, c = b.
# update: a = a + w * c (clipped at 0 if asked), with a = x, w = C, c = the
# backprojected residual.
ELEMENTWISE_SHADER = """
struct Params { n: u32, nonneg: u32, pad0: u32, pad1: u32, }

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 256u;
}

@compute @workgroup_size(256)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= params.n) { return; }
    a[i] = w[i] * (c[i] - a[i]);
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= params.n) { return; }
    var v = a[i] + w[i] * c[i];
    if (params.nonneg == 1u) { v = max(v, 0.0); }
    a[i] = v;
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
        module = self.device.create_shader_module(code=FORWARD_SHADER)
        self.forward_pipeline = self.device.create_compute_pipeline(
            layout='auto', compute={'module': module, 'entry_point': 'main'})
        module = self.device.create_shader_module(code=ELEMENTWISE_SHADER)
        self.residual_pipeline, self.update_pipeline = (
            self.device.create_compute_pipeline(
                layout='auto', compute={'module': module,
                                        'entry_point': name})
            for name in ('residual', 'update'))
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
        from . import backproject_cpu
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
            corners = vol_geom.slab_corners(s0, s1)
            backproject_cpu.check_in_front_of_source(P, corners)
            h = np.einsum('kij,cj->kci', P, np.concatenate(
                [corners, np.ones((8, 1))], axis=1))
            vcorner = h[:, :, 1] / h[:, :, 2]
            v_lo = np.clip(np.floor(vcorner.min(1)) - 1, 0, n_v).astype(int)
            v_hi = np.clip(np.ceil(vcorner.max(1)) + 2, 0, n_v).astype(int)

            # WebGPU guarantees that new buffers are zero-filled.
            vol_buf = self._buffer(
                size=nzs * plane * 4, usage=wgpu.BufferUsage.COPY_SRC)

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

    def forward_project(self, volume, geom, vol_geom, threads=None,
                        progress=None, should_stop=None, slab=None):
        """Same as forward_cpu.forward_project, on the GPU. The volume
        streams through in slabs of z slices (plus one slice of overlap
        for interpolation); each sample is counted in exactly one slab."""
        from .forward_cpu import index_space_rays
        wgpu = self.wgpu
        should_stop = should_stop or (lambda: False)
        vol = np.ascontiguousarray(volume, dtype=np.float32)
        if vol.shape != vol_geom.shape:
            raise ValueError(f'Volume shape {vol.shape} does not match '
                             f'{vol_geom.shape}')
        nz, ny, nx = vol.shape
        n, n_v, n_u = geom.n_angles, geom.n_v, geom.n_u
        rays, scale = index_space_rays(geom, vol_geom)
        rays = np.stack(rays, axis=1)  # (n, 6, 3)
        rays = np.concatenate([rays, np.zeros((n, 6, 1))], axis=2)
        rays = rays.astype(np.float32)

        plane = ny * nx
        if slab is None:
            slab = max(1, self.max_binding // (4 * plane) - 1)
        slab = min(slab, nz)
        if (slab + 1) * plane * 4 > self.max_binding and slab > 1:
            slab = max(1, self.max_binding // (4 * plane) - 1)
        if 2 * plane * 4 > self.max_binding:
            raise RuntimeError(
                f'A single {ny}x{nx} slice is too large for this GPU '
                f'({self.max_binding} bytes per buffer)')
        planes = max(nx, ny, nz)
        per_proj = n_u * n_v
        batch = max(1, min(n, MAX_UPDATES_PER_DISPATCH //
                           max(1, per_proj * planes),
                           self.max_binding // (4 * per_proj),
                           self.max_groups))
        if per_proj * 4 > self.max_binding:
            raise RuntimeError('A single projection is too large for this '
                               'GPU')

        out = np.zeros((n, n_v, n_u), dtype=np.float32)
        slabs = list(range(0, nz, slab))
        total = len(slabs) * n
        done = 0
        for s0 in slabs:
            s1 = min(nz, s0 + slab)
            b1 = min(nz, s1 + 1)
            vol_buf = self._buffer(np.ascontiguousarray(vol[s0:b1]))
            own0 = -1 if s0 == 0 else s0
            for k0 in range(0, n, batch):
                if should_stop():
                    vol_buf.destroy()
                    return None
                k1 = min(n, k0 + batch)
                params = np.zeros(16, dtype=np.uint32)
                params[:6] = [n_u, n_v, k1 - k0, nx, ny, nz]
                params[6:10] = np.array([s0, b1 - s0, own0, s1],
                                        dtype=np.int32).view(np.uint32)
                params[12:15] = np.array(scale, dtype=np.float32).view(
                    np.uint32)
                out_buf = self._buffer(
                    size=(k1 - k0) * per_proj * 4,
                    usage=wgpu.BufferUsage.COPY_SRC)
                ray_buf = self._buffer(np.ascontiguousarray(rays[k0:k1]))
                par_buf = self.device.create_buffer_with_data(
                    data=params, usage=wgpu.BufferUsage.UNIFORM)
                bind = self.device.create_bind_group(
                    layout=self.forward_pipeline.get_bind_group_layout(0),
                    entries=[
                        {'binding': 0, 'resource': {'buffer': vol_buf}},
                        {'binding': 1, 'resource': {'buffer': ray_buf}},
                        {'binding': 2, 'resource': {'buffer': out_buf}},
                        {'binding': 3, 'resource': {'buffer': par_buf}},
                    ])
                enc = self.device.create_command_encoder()
                cp = enc.begin_compute_pass()
                cp.set_pipeline(self.forward_pipeline)
                cp.set_bind_group(0, bind)
                cp.dispatch_workgroups(-(-n_u // 8), -(-n_v // 8), k1 - k0)
                cp.end()
                self.device.queue.submit([enc.finish()])
                data = self.device.queue.read_buffer(out_buf)
                out[k0:k1] += np.frombuffer(data, dtype=np.float32).reshape(
                    k1 - k0, n_v, n_u)
                for b in (out_buf, ray_buf, par_buf):
                    b.destroy()
                done += k1 - k0
                if progress is not None:
                    progress(done, total)
            vol_buf.destroy()
        return out

    def _elementwise(self, pipeline, a, w, c, n, nonneg=False):
        """Bind group, workgroup counts and parameter buffer for an
        element-wise SIRT step on n values."""
        params = np.array([n, int(nonneg), 0, 0], dtype=np.uint32)
        par_buf = self.device.create_buffer_with_data(
            data=params, usage=self.wgpu.BufferUsage.UNIFORM)
        bind = self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{'binding': i, 'resource': {'buffer': b}}
                     for i, b in enumerate((a, w, c, par_buf))])
        groups = -(-n // 256)
        gx = min(groups, self.max_groups)
        return bind, (gx, -(-groups // gx), 1), par_buf

    def sirt_loop(self, b, R, C, x, geom, vol_geom, iterations,
                  nonnegative=True, should_stop=None, progress=None,
                  callback=None, volume_needed=None, log=None):
        """Run SIRT iterations with everything on the GPU:
        x <- x + C * A^T (R * (b - A x)), clipped at 0 if nonnegative.

        b, R: (n_angles, n_v, n_u) projections and row weights.
        C: column weights, already scaled by the relaxation. x: start.
        progress(iteration) after each one; callback(it, x) as for
        sirt(), with x None when volume_needed(it) says it is not used.

        Only the volume comes back. Returns the volume, None if stopped,
        or NotImplemented if the problem does not fit on this GPU (then
        the caller runs the general loop)."""
        log = log or (lambda msg: None)
        should_stop = should_stop or (lambda: False)
        nz, ny, nx = vol_geom.shape
        n, n_v, n_u = geom.n_angles, geom.n_v, geom.n_u
        plane = ny * nx
        n_vox = nz * plane
        per_proj = n_u * n_v
        if n_vox * 4 > self.max_binding or per_proj * 4 > self.max_binding:
            log('SIRT on the GPU streams the data (the volume is larger '
                'than one GPU buffer)')
            return NotImplemented
        planes = max(nx, ny, nz)
        batch = max(1, min(n,
                           MAX_UPDATES_PER_DISPATCH //
                           max(1, per_proj * planes),
                           MAX_UPDATES_PER_DISPATCH // max(1, n_vox),
                           self.max_binding // (4 * per_proj),
                           self.max_groups))
        try:
            plan = self._sirt_resident(b, R, C, x, geom, vol_geom, batch)
        except Exception as e:
            log(f'SIRT on the GPU streams the data ({e})')
            return NotImplemented
        try:
            return self._sirt_iterate(plan, iterations, nonnegative,
                                      should_stop, progress, callback,
                                      volume_needed, vol_geom.shape)
        finally:
            for buf in plan['buffers']:
                buf.destroy()

    def _sirt_resident(self, b, R, C, x, geom, vol_geom, batch):
        """Upload everything SIRT needs and record one iteration's work:
        per projection batch, a forward projection, the residual and a
        backprojection, as bind groups to reuse every iteration."""
        from . import backproject_cpu
        from .forward_cpu import index_space_rays
        wgpu = self.wgpu
        nz, ny, nx = vol_geom.shape
        n, n_v, n_u = geom.n_angles, geom.n_v, geom.n_u
        per_proj = n_u * n_v
        n_vox = nz * ny * nx
        buffers = []

        def upload(a, usage=0):
            buf = self._buffer(np.ascontiguousarray(a, dtype=np.float32),
                               usage=usage)
            buffers.append(buf)
            return buf

        def uniform(values, dtype):
            buf = self.device.create_buffer_with_data(
                data=np.asarray(values, dtype=dtype),
                usage=wgpu.BufferUsage.UNIFORM)
            buffers.append(buf)
            return buf

        try:
            x_buf = upload(x, wgpu.BufferUsage.COPY_SRC)
            c_buf = upload(C)
            u_buf = self._buffer(size=n_vox * 4,
                                 usage=wgpu.BufferUsage.COPY_DST)
            buffers.append(u_buf)
            r_buf = self._buffer(size=batch * per_proj * 4)
            buffers.append(r_buf)

            # Backprojection matrices for the whole volume as one slab.
            P = geom.projection_matrices()
            backproject_cpu.check_in_front_of_source(
                P, vol_geom.slab_corners(0, nz))
            dz, dy, dx = vol_geom.voxel_size
            G = np.array([[dx, 0, 0, vol_geom.x_coords()[0]],
                          [0, dy, 0, vol_geom.y_coords()[0]],
                          [0, 0, dz, vol_geom.z_coords()[0]],
                          [0, 0, 0, 1.0]])
            mats = (P @ G).astype(np.float32)
            rays, scale = index_space_rays(geom, vol_geom)
            rays = np.stack(rays, axis=1)
            rays = np.concatenate([rays, np.zeros((n, 6, 1))], axis=2)

            steps = []
            for k0 in range(0, n, batch):
                k1 = min(n, k0 + batch)
                nk = k1 - k0
                fwd = np.zeros(16, dtype=np.uint32)
                fwd[:6] = [n_u, n_v, nk, nx, ny, nz]
                # The whole volume is one slab that owns every sample.
                fwd[6:10] = np.array([0, nz, -1, nz],
                                     dtype=np.int32).view(np.uint32)
                fwd[12:15] = np.array(scale, dtype=np.float32).view(
                    np.uint32)
                fwd_bind = self.device.create_bind_group(
                    layout=self.forward_pipeline.get_bind_group_layout(0),
                    entries=[
                        {'binding': 0, 'resource': {'buffer': x_buf}},
                        {'binding': 1,
                         'resource': {'buffer': upload(rays[k0:k1])}},
                        {'binding': 2, 'resource': {'buffer': r_buf}},
                        {'binding': 3,
                         'resource': {'buffer': uniform(fwd, np.uint32)}},
                    ])
                res = self._elementwise(self.residual_pipeline, r_buf,
                                        upload(R[k0:k1]), upload(b[k0:k1]),
                                        nk * per_proj)
                buffers.append(res[2])
                bp_bind = self.device.create_bind_group(
                    layout=self.pipeline.get_bind_group_layout(0),
                    entries=[
                        {'binding': 0, 'resource': {'buffer': r_buf}},
                        {'binding': 1,
                         'resource': {'buffer': upload(mats[k0:k1])}},
                        {'binding': 2, 'resource': {'buffer': u_buf}},
                        {'binding': 3, 'resource': {'buffer': uniform(
                            [nx, ny, nz, nk, n_u, n_v, 0, 0], np.uint32)}},
                    ])
                steps.append((fwd_bind, nk, res, bp_bind))
        except Exception:
            for buf in buffers:
                buf.destroy()
            raise
        return dict(buffers=buffers, x=x_buf, c=c_buf, u=u_buf,
                    steps=steps, n_vox=n_vox, n_u=n_u, n_v=n_v,
                    nx=nx, ny=ny, nz=nz)

    def _sirt_iterate(self, plan, iterations, nonnegative, should_stop,
                      progress, callback, volume_needed, shape):
        n_u, n_v = plan['n_u'], plan['n_v']
        nx, ny, nz = plan['nx'], plan['ny'], plan['nz']
        upd_bind, upd_groups, upd_par = self._elementwise(
            self.update_pipeline, plan['x'], plan['c'], plan['u'],
            plan['n_vox'], nonneg=nonnegative)
        plan['buffers'].append(upd_par)

        def read_x():
            data = self.device.queue.read_buffer(plan['x'])
            return np.frombuffer(data, dtype=np.float32).reshape(
                shape).copy()

        def run(pipeline, bind, groups):
            enc = self.device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipeline)
            cp.set_bind_group(0, bind)
            cp.dispatch_workgroups(*groups)
            cp.end()
            # One submission per dispatch keeps each one short for the
            # driver watchdog; the queue runs them in order.
            self.device.queue.submit([enc.finish()])

        for it in range(iterations):
            if should_stop():
                return None
            enc = self.device.create_command_encoder()
            enc.clear_buffer(plan['u'])
            self.device.queue.submit([enc.finish()])
            for fwd_bind, nk, res, bp_bind in plan['steps']:
                run(self.forward_pipeline, fwd_bind,
                    (-(-n_u // 8), -(-n_v // 8), nk))
                run(self.residual_pipeline, res[0], res[1])
                run(self.pipeline, bp_bind,
                    (-(-nx // 8), -(-ny // 8), -(-nz // 4)))
            run(self.update_pipeline, upd_bind, upd_groups)
            self._wait()
            if progress is not None:
                progress(it)
            if callback is not None:
                want = volume_needed is None or volume_needed(it)
                if callback(it, read_x() if want else None):
                    break
        return read_x()

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
        from . import forward_cpu
        vol = rng.random(vg.shape, dtype=np.float32)
        ref = forward_cpu.forward_project(vol, g, vg, threads=1)
        got = self.forward_project(vol, g, vg, slab=2)
        if not np.allclose(got, ref, rtol=1e-4, atol=1e-4 * abs(ref).max()):
            raise RuntimeError('GPU forward projection self-test gave wrong '
                               f'results (max error '
                               f'{abs(got - ref).max():.3g})')


_cached = None
_failure = None


def make_backend():
    """Create the WebGPU backend once per process, or raise with a reason
    if it cannot be used. A failure is remembered too, so later runs fall
    back at once instead of repeating device setup and the self-test."""
    global _cached, _failure
    if _cached is not None:
        return _cached
    if _failure is not None:
        raise RuntimeError(_failure)
    try:
        try:
            import wgpu
        except ImportError:
            raise RuntimeError('the wgpu package is not installed '
                               '(pip install wgpu)')
        _restrict_backends()
        from .backend import Backend
        bp = WgpuBackprojector(_choose_adapter(wgpu))
        bp.self_test()
    except Exception as e:
        _failure = str(e) or type(e).__name__
        raise
    _cached = Backend('gpu', bp.description, bp.backproject,
                      bp.forward_project, bp.sirt_loop)
    return _cached
