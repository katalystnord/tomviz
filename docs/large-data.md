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

1. **SIRT and parallel beams.** Parallel-beam slices are independent
   (each needs one detector row), so streaming them is simpler still.
   SIRT needs every projection per iteration; streaming it means one pass
   over the projection file per iteration, which at 6.6 GB/s is a few
   seconds for tens of GB.
2. **Bring it into Tomviz.** Tomviz's pipeline passes whole images
   between operators. The least intrusive route:
   * a *Reconstruct to File* option on the cone-beam operators: when the
     projections came from a file (or are saved to the NVMe first), FDK
     writes the volume to a file there instead of returning it;
   * open the result with Tomviz's existing subsampled loading (it can
     already read strided sub-volumes of HDF5), and full resolution where
     needed;
   * write the volume as HDF5 (chunked) rather than raw, so it opens like
     any other file and carries its spacing.
3. **Where the data lives.** Data on the HDD (0.28 GB/s) would make reads
   the bottleneck again (12 GB of projections take 43 s just to read).
   Copying a scan to the NVMe first, or recording straight to it, keeps
   streaming free.
