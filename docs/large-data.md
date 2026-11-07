# Reconstructing scans larger than memory

Tomviz keeps every dataset in memory as a vtkImageData. Synchrotron and
high-resolution lab scans no longer fit: 2048^3 voxels are 34 GB in
float32, 4096^3 are 275 GB. This branch adds the first piece, FDK that
streams from and to files a slab at a time, and measures it.

## How it works (tomviz/conebeam/stream.py)

The ideas come from our kvstream work (llama.cpp with a key-value cache
larger than GPU memory): keep the full data in a larger, slower store,
keep a working set close to the processor, stream pages in while the
processor works, and merge partial results exactly.

* **Pages.** The volume is processed in slabs of slices. Each slab needs
  only the detector rows it projects onto (found from the geometry), so
  only those rows are read from the projection file.
* **Exact pieces.** FDK's pre-weighting and ramp filter work row by row,
  and backprojection of a slab through a *row-cropped geometry* (the same
  source and pixel steps, the detector centre moved to the middle of the
  rows) is exactly the in-memory result for those slices. Tested:
  identical to `fdk()` to 1e-5 of the peak, full and short scans, CPU and
  GPU (tests/python/stream_test.py).
* **Resident set.** Neighbouring slabs need overlapping rows. The previous
  slab's filtered rows are kept, so every detector row is read and
  filtered exactly once.
* **Overlap.** A reader thread fetches the next slab's new rows (with
  positioned reads that release the GIL) while the current slab is
  filtered and backprojected; a writer thread stores finished slabs.
* **Bounded memory.** The slab size follows a memory budget (rows read
  ahead, filtered and kept; the volume slab, its GPU read-back and the
  previous slab being written).

## Measured (RX 9070, KC3000 NVMe, 24 threads, 60 GB RAM)

Storage on this machine, read with the page cache bypassed:

| Store | Read |
|---|---|
| HDD (`~/store`, where data lives today) | 0.28 GB/s |
| NVMe (system drive) | 6.6 GB/s, 6.9 with 4 streams; 4.3 GB/s write |

FDK, 720 projections of 1024^2 into 1024^3 voxels (3.0 GB in, 4.3 GB
out), projections read from the NVMe:

| | Total | Filter | Backproject | Waiting for reads | Peak memory |
|---|---|---|---|---|---|
| Streamed, first version | 94.2 s | 79.1 s | 14.2 s | 0.8 s | 4.1 GB |
| Streamed, filter 6x faster, rows kept | 21.5 s | 7.6 s | 13.0 s | 0.7 s | 5.5 GB |
| In memory, same filter | 22.3 s (1.9 read + 20.4) | | | | 7.8 GB |
| Streamed, filter reused weights, filtered one slab ahead | **14.0 s** | 4.7 s (hidden) | 12.7 s | 0.5 s | **3.6 GB** |
| In memory, same filter | 16.8 s (1.9 read + 14.9) | | | | 7.8 GB |

The first measurement showed the CPU filter, not the disk, as the
bottleneck. Profiling it gave a 6x faster filter for all FDK (a separable
cosine weight instead of (v, u, 3) float64 temporaries, and scipy.fft in
float32 on all cores instead of pyfftw's numpy interface), and keeping
filtered rows between slabs removed the re-filtering. Streaming is now as
fast as reconstructing in memory, with the disk waits under 1 s: on this
NVMe, reading is not the limit; the GPU backprojection is.

Larger than memory: 720 projections of 2048^2 into 2048^3 voxels (12.1 GB
in, 34.4 GB out; about 38 GB of RAM was free):

| Memory budget | Slabs | Total | Filter | Backproject | Waiting for reads | Peak memory |
|---|---|---|---|---|---|---|
| 4 GiB | 1142 | 1408 s | 377 s | 1030 s | 1.4 s | 6.4 GB |
| 20 GiB | 8 | 193 s | 44 s | 146 s | 2.5 s | 18.8 GB |
| 20 GiB, filter reused weights, filtered one slab ahead | 10 | **118 s** | 20 s (hidden) | 114 s | 1.4 s | 15.3 GB |

With a cone beam even one slice near the top or bottom of the volume
projects onto a tall band of detector rows, and keeping that band's
filtered rows for all 720 projections costs 18 MB per row. Within 4 GiB
only about 230 rows fit, which forced slabs of one or two slices; the GPU
then worked on slivers (half of every 4-slice work group idle) with
thousands of synchronisations. Within 20 GiB, 8 slabs, and the
backprojection (146 s) is close to what the 1024^3 run predicts for 8x the
voxels (about 105 s). The budget now defaults to half the memory available
at the time. The disk was never the limit: 2.5 s of waiting in 193 s.

With the filter running on the CPU one slab ahead of the GPU (reader,
filter, GPU and writer as four stages), the GPU waits 3.7 s for filtered
rows in 118 s: the reconstruction now takes about as long as its
backprojection alone, and streaming is faster than in memory.

## Using it

From Tomviz: *Tomography > Reconstruction > Reconstruct Large Scan to
File*. It is a data source, not an operator on loaded
data: choose the scan on disk and an output .emd file; Tomviz shows the
volume subsampled to about 256^3 and the file opens at full resolution.

From a shell:

    python -m tomviz.conebeam.reconstruct_file SCAN OUTPUT.emd [options]

FDK for cone beams; for parallel beams (the default for NXtomo and Data
Exchange files, or `--beam parallel`) FBP or `--method sirt`, with
`--find-axis` to locate the rotation axis.

Scans it opens without loading (tomviz/conebeam/scan.py, sources.py):

* scanner projects with their geometry: Nikon .xtekct, Waygate .pca,
  Bruker SkyScan .log (through Tomviz's readers for them, when present);
* a folder of TIFF files or a multi-page TIFF. Uncompressed pages, which
  scanners write, are read a row band at a time with positioned reads
  from eight threads; compressed pages are decoded whole per band;
* HDF5: NXtomo (projections picked out from among flats and darks by
  image_key) and Data Exchange, both with their flat and dark fields, or
  any 3D dataset by name;
* .npy and raw frames.

Intensities become line integrals as each band is read: from the scan's
flat and dark fields, or an open-beam level (given, stored by the
scanner, or estimated per projection from the brightest pixels of 32
sampled rows).

720 uint16 TIFF files of 2048^2 (6 GB) on the NVMe into a 1024^3 EMD
file, through the command line: 30 s, of which reading 3.1 s.

## Parallel beams (synchrotrons)

The same code reconstructs parallel-beam scans: with no fan, FDK's filter
is FBP's (no cosine weight, the detector pitch as the sample spacing,
Parker weights with a zero fan angle), and a slab needs only its own
detector rows.

A synchrotron-sized scan, 1800 projections of 2048^2 over 180 degrees
(30.2 GB, float32 on the NVMe), into 2048^3 voxels (34.4 GB) within
20 GiB: 176 s in total (backprojection 168 s, the GPU waiting 6.9 s for
filtered rows, reading 4.2 s), peak memory 16.3 GB.

**SIRT** streams for parallel beams whose rays are perpendicular to the
axis: each ray stays in its slice, so slabs are separate problems, solved
one after another with all iterations on the GPU. Interpolation between
detector rows and slices couples neighbours slightly; each slab is solved
with 8 extra slices on each side, which are dropped. Measured against
SIRT of the whole volume (largest difference, relative to the peak):

| Halo slices | 0 | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|---|
| 10 iterations | 22 | 1.0 | 0.19 | 5e-3 | 1.3e-5 | |
| 50 iterations | | | | | 2.9e-5 | 4.4e-7 |
| 200 iterations | | | | | 4.0e-5 | 7.1e-7 |

With slices on the detector row grid (voxel size equal to the row pitch,
the default) the difference is exactly zero.

**Rotation axis.** Synchrotron files rarely say where the axis projects;
*Find the Rotation Axis* takes it from projections 180 degrees apart (the
method of Tilt Axis Shift Alignment's opposite projections), on 16
detector rows read from the file: within 0.03 pixels on the phantom.

## Tried and measured

* **Sharing slabs between the GPU and the CPU** (the kvstream
  co-processor idea; backprojection is a sum, so it would be exact). On
  this machine lavapipe, the same WebGPU kernel on the 24 CPU cores,
  backprojects 37x slower than the RX 9070 (34.6 s against 0.94 s for
  the same slab); giving it even 1/6 of the slices made the whole six
  times slower. Not worth it beside a discrete GPU; possibly on machines
  with only an integrated one. (An earlier estimate of 28 % came from
  whole FDK runs dominated by the then slow CPU filter.)
* **A GPU filter** would save at most what the pipeline already hides.

## Next steps

1. **Cone-beam SIRT.** Every cone-beam ray crosses many slices, so
   slabs are not independent: each iteration needs the forward projection
   of the whole volume, summed slab by slab, before any slab can be
   updated. Streamed, that is two passes over the volume file and one
   over the projections per iteration; at 2048^3 the GPU work alone
   would be hours for 50 iterations.
3. **Faster SIRT.** Its ray-driven forward projector costs about three
   times a backprojection; SIRT of 512^3 from 900 projections takes 9 s
   per iteration on the RX 9070, streamed or not.
4. **Where the data lives.** Data on the HDD (0.28 GB/s) would make reads
   the bottleneck again (12 GB of projections take 43 s just to read).
   Copying a scan to the NVMe first, or recording straight to it, keeps
   streaming free.
