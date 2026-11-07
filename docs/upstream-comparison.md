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
| Ours, lavapipe (its own projectors; parallel beams now take (a)) | 2.77 s | 4.8x faster |
| Ours, numpy reference | 154 s | **12x slower** |
| Ours, CPU, sparse Joseph matrix per slice (a) | 0.45 s (+2.3 s setup) | 29x faster |
| Ours, CPU, upstream's Siddon matrix per slice (b) | 0.24 s (+4.7 s setup) | 55x faster |

Error against the phantom (RMSE; 0.171 for an all-zero volume):

| Iterations | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| Ours, RX 9070 | 0.141 | 0.130 | 0.112 | 0.096 | 0.077 | 0.049 |
| Upstream, step 1/‖A‖² = 2.3e-5 | 0.139 | 0.130 | 0.113 | 0.097 | 0.078 | 0.050 |
| Upstream, default step 1e-4 | 0.298 | 0.171 | 0.298 | 0.170 | 0.170 | 0.170 |
| Ours, CPU, Joseph matrix (a) | 0.141 | 0.130 | 0.112 | 0.096 | 0.077 | 0.049 |
| Ours, CPU, Siddon matrix (b) | 0.141 | 0.130 | 0.112 | 0.096 | 0.077 | 0.049 |

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

## Without a GPU: two options for upstream to choose from

For parallel beams with rays perpendicular to the axis (which covers
Tomviz's tilt series), SIRT without a hardware GPU now projects with a
sparse matrix per slice, applied to all slices at once and threaded
(`tomviz.projectors.sparse`; `sirt(..., cpu_matrix=...)`). Upstream's SIRT
applies the same kind of matrix one slice at a time on one core, which is
where its time goes. Both matrices are implemented:

| | (a) `joseph` | (b) `siddon` |
|---|---|---|
| Matrix | new: Joseph's interpolation, as the GPU projector | upstream's `parallelRay`, unchanged |
| SIRT per iteration, 256^3 from 180, 24 threads | 0.45 s | 0.24 s (fewer non-zeros) |
| Building the matrix | 1.1 s | 3.5 s (Python loop over rays) |
| Rays within a few degrees of an axis | 0.2 % from the true line integrals | 3 % at 2 degrees (flat pixels: reads the nearest row) |
| Geometries | any level parallel beam: off-centre axis, non-square slices | square slices, axis at the detector centre |
| Same as the GPU result | yes, to the matched/unmatched pair | close |
| New code to maintain | the matrix builder (about 50 lines) | none |

Same reconstruction quality either way (0.049 after 50 iterations, as on
the GPU). For (b), `parallelRay` moved unchanged into
`tomviz/projectors/siddon.py`: Recon_ART, Recon_SIRT and
Recon_TV_minimization each carried a copy (differing only in formatting
and a mask's dtype) and now import it; the shared one builds bit-identical
matrices to all three originals. The default here is (a); switching is one
argument. Both keep a memory guard: the matrix grows with slice size,
detector width and projections (about 4.5 GB for 512^2 slices from 720
projections), and bigger problems go to the GPU path.

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
