# Against what Tomviz users run today

Measured 2026-11-07 with `dev/upstream_compare.py`, which runs upstream's
`Recon_SIRT.py` and `GenerateTiltSeries.py` exactly as on
`upstream/master`. Problem: a Shepp-Logan phantom on 256^3 unit voxels,
parallel beam, 180 projections over 180 degrees (tilt angles -90 to 90),
exact projections. Machine: AMD Radeon RX 9070, 24 CPU threads; lavapipe
is Mesa's software Vulkan on the same CPU (what a machine without a GPU
gets when `mesa-vulkan-drivers` is installed). TomoPy was not installed
and is not compared.

## SIRT

| | Time per iteration | Against upstream |
|---|---|---|
| Upstream SIRT (sparse matrix per slice, one core) | 13.3 s (+3.5 s to build the matrix) | 1x |
| Ours, RX 9070 | 0.031 s | about 430x faster |
| Ours, lavapipe | 2.77 s | 4.8x faster |
| Ours, numpy reference | 154 s | **12x slower** |

Error against the phantom (RMSE; 0.171 for an all-zero volume):

| Iterations | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| Ours | 0.141 | 0.130 | 0.112 | 0.096 | 0.077 | 0.049 |
| Upstream, step 1/‖A‖² = 2.3e-5 | 0.139 | 0.130 | 0.113 | 0.097 | 0.078 | 0.050 |
| Upstream, default step 1e-4 | 0.298 | 0.171 | 0.298 | 0.170 | 0.170 | 0.170 |

* With a good step, upstream's Landweber iteration converges exactly as
  fast per iteration as our SIRT: the gain is the time per iteration,
  not better iterations.
* Upstream's default step diverges at this size: Landweber needs a step
  below 2/‖A‖² = 4.5e-5 here, and ‖A‖² grows with the slice size and the
  number of projections. The positivity clamp keeps the result bounded,
  so it looks like an empty volume rather than failing. (At 64^3 from 60
  projections the default is stable but slower: RMSE 0.063 after 50
  iterations against 0.037 with a tuned step.) Our SIRT normalises by
  the row and column sums and needs no step size.
* Our numpy backend is the slow, obvious reference the GPU code is tested
  against, and is far slower than upstream's sparse matrix. Machines with
  no Vulkan driver at all should fall back to faster CPU code, not to it.

## Generate Tilt Series

| | 256^3, 180 tilts | Against upstream |
|---|---|---|
| Upstream (scipy.ndimage.rotate, then a sum) | 476 s | 1x |
| Ours, RX 9070 | 1.2 s | about 400x faster |
| Ours, lavapipe | 4.7 s | about 100x faster |
| Ours, no GPU adapter | 473 s | upstream's code, unchanged |

Same accuracy: both are within 3.8 % on average of the exact projections
of the phantom (at 96^3; the limit is voxelising the phantom's sharp
edges), and within 6.5 % of the peak of each other (bilinear rotation
against Joseph interpolation). Same orientation and angle convention.
