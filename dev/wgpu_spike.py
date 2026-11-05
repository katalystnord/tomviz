"""WebGPU feasibility check for tomviz GPU compute.

Lists every adapter wgpu can see, then on each one runs a small compute
shader (y = a * x + b over a million floats) and checks the result against
numpy. Prints one line per adapter and exits non-zero if any adapter that
was found failed the check.

Usage:
    python dev/wgpu_spike.py            # every adapter
    python dev/wgpu_spike.py --json     # machine-readable, for a CI matrix

Needs only numpy and wgpu (pip install wgpu). Runs headless; no window or
display is required.
"""

import argparse
import json
import platform
import sys
import time

import numpy as np

SHADER = """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

struct Params { a: f32, b: f32, n: u32, pad: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        y[i] = params.a * x[i] + params.b;
    }
}
"""


def run_on_adapter(adapter, n=1 << 20):
    import wgpu

    device = adapter.request_device_sync()
    x = np.random.default_rng(0).random(n, dtype=np.float32)
    a, b = np.float32(2.5), np.float32(-1.0)

    buf_x = device.create_buffer_with_data(
        data=x, usage=wgpu.BufferUsage.STORAGE)
    buf_y = device.create_buffer(
        size=x.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    params = np.zeros(4, dtype=np.uint32)
    params[0:2] = np.array([a, b], dtype=np.float32).view(np.uint32)
    params[2] = n
    buf_p = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM)

    module = device.create_shader_module(code=SHADER)
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"})
    bind = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": buf_x}},
            {"binding": 1, "resource": {"buffer": buf_y}},
            {"binding": 2, "resource": {"buffer": buf_p}},
        ])

    t0 = time.perf_counter()
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bind)
    cp.dispatch_workgroups((n + 255) // 256)
    cp.end()
    device.queue.submit([enc.finish()])
    y = np.frombuffer(device.queue.read_buffer(buf_y), dtype=np.float32)
    elapsed = time.perf_counter() - t0

    ok = bool(np.allclose(y, a * x + b, rtol=1e-6, atol=1e-6))
    limits = device.limits
    return {
        "ok": ok,
        "seconds": round(elapsed, 4),
        "max_storage_buffer_binding_size":
            limits["max-storage-buffer-binding-size"],
        "max_buffer_size": limits["max-buffer-size"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "adapters": [],
    }
    try:
        import wgpu
    except ImportError as e:
        report["error"] = f"wgpu not importable: {e}"
        print(json.dumps(report, indent=2) if args.json else report["error"])
        return 1
    report["wgpu"] = wgpu.__version__

    failed = 0
    for adapter in wgpu.gpu.enumerate_adapters_sync():
        info = adapter.info
        entry = {
            "device": info["device"],
            "adapter_type": info["adapter_type"],
            "backend": info["backend_type"],
        }
        # The OpenGL backend has no compute shaders worth relying on and is
        # not a target; record it but don't count it as a failure.
        if info["backend_type"] == "OpenGL":
            entry["skipped"] = "OpenGL backend is not a compute target"
        else:
            try:
                entry.update(run_on_adapter(adapter))
            except Exception as e:
                entry["ok"] = False
                entry["error"] = f"{type(e).__name__}: {e}"
            if not entry["ok"]:
                failed += 1
        report["adapters"].append(entry)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{report['platform']}  python {report['python']}  "
              f"wgpu {report['wgpu']}")
        if not report["adapters"]:
            print("  no adapters found")
        for e in report["adapters"]:
            status = ("skip" if "skipped" in e
                      else "PASS" if e["ok"] else "FAIL")
            extra = e.get("error") or e.get("skipped") or (
                f"{e['seconds']} s, max storage binding "
                f"{e['max_storage_buffer_binding_size'] // 2**20} MiB")
            print(f"  [{status}] {e['backend']:7} {e['adapter_type']:12} "
                  f"{e['device']}: {extra}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
