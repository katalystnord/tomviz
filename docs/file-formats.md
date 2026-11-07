# Tomography file formats in Tomviz

What the major tomography instruments write, what Tomviz can open, and
how. Compiled 2026-11-06/07. Every geometry
convention below was checked on real public scans; where something could
not be checked, it says so.

## What opening a scan gives you now

For the formats marked **projections + geometry**, opening the file gives:

* the projections as a tilt series with their angles;
* flat and dark frames, when the file has them, as datasets of their own
  (no displays), ready to connect to **Flat/Dark Field Correction**;
* the scanner geometry in the dataset's metadata
  (`dataset.metadata['geometry']`: distances in mm, detector pixel pitch,
  centre-of-rotation offsets in pixels, rotation direction, or per-view
  vectors in the ASTRA `cone_vec` layout);
* the open-beam level, where the scanner records one
  (`dataset.metadata['white_level']`).

Operators can use these directly. The cone-beam reconstruction work
(a separate branch) does: its FDK and SIRT take the geometry from the
data, and its intensity conversion uses the stored open-beam level, so
a lab CT scan goes open, convert to line integrals, reconstruct, with
nothing typed in.

## Summary

| Format | Instruments | Tomviz now | Checked against |
|---|---|---|---|
| Nikon `.xtekct` + TIFF | Nikon XT H, X-TEK | projections + geometry, centre from `*_recon.xtekct` | Nikon's own reconstruction, correlation 0.84 to 0.87 |
| Waygate `.pca` + TIFF + `.pcp` | phoenix v\|tome\|x, datos\|x | projections + geometry | Waygate's own reconstruction, 0.9935 |
| ZEISS `.txrm` | Xradia Versa, Ultra | projections + flat + geometry with per-view shifts | ZEISS's own reconstruction, 0.968 (object region) |
| ZEISS `.txm`, `.xrm` | Xradia | volume / image | file structure |
| Bruker `.log` + TIFF/BMP | SkyScan micro-CT | projections + geometry (no centre in the log) | sharpness only; no vendor reconstruction available |
| NeXus NXtomo `.nxs` `.nx` `.h5` | ESRF, Diamond, others | projections + flats + darks + angles | Diamond's `tomo_standard.nxs` |
| Data Exchange `.h5` | APS, TomoBank, DanMAX | projections + flats + darks + angles (flats and darks are now datasets) | existing reader |
| MRC `.mrc .mrcs .st .ali .preali .rec` | SerialEM, Thermo Fisher Tomography, IMOD | stacks with tilt angles from headers or sidecar files | ETSpy samples (legacy FEI, SerialEM) |
| Gatan `.dm3` `.dm4` | DigitalMicrograph, Gatan tomography | images and stacks; tilt series from a degree-calibrated z axis | ETSpy `HAADF.dm3` |
| EMD (Berkeley) | | as before, with tilt angles | |
| Velox EMD | Thermo Fisher Velox | images and stacks; tilt series from per-frame alpha tilt | a Velox stack (rosettasciio test data) |
| YXLON, Diondo, METROTOM | | not yet (no public sample scans) | |
| Scanco `.isq` `.aim` | Scanco micro-CT | not yet | |
| Volume Graphics `.vgi` + `.vol` | VGSTUDIO, Nikon exports | not yet | |
| Avizo/Amira `.am` | Thermo Fisher | not yet | |
| EDF | ESRF legacy | not yet | |
| NRRD, NIfTI, OME-Zarr | volume exchange | not yet (MetaImage, VTI, OME-TIFF, DICOM already open) | |

## Lab and industrial CT

None of these vendors publishes a specification. The readers follow the
key names that the open readers use (CIL, Apache-2.0; TIGRE,
BSD-3-Clause; dxchange, Argonne BSD-style), and the conventions that
code cannot settle (rotation direction, offset signs, pixel pitch) were
set by reconstructing public scans and comparing with each vendor's own
reconstruction of the same scan (a script that repeats those
comparisons lives with the cone-beam reconstruction work, since it needs
a cone-beam reconstruction).

### Nikon Metrology XT H / X-TEK (`.xtekct`)

INI file, section `[XTekCT]`, next to one TIFF per projection
(`<Name>_0001.tif` onwards; one more file than `Projections`, the
360 degree repeat).

* Distances `SrcToObject`, `SrcToDetector`; detector `DetectorPixelsX/Y`,
  `DetectorPixelSizeX/Y`, `DetectorOffsetX/Y` (mm).
* Angles `InitialAngle` + i x `AngularStep`. Like CIL, the `.ang` and
  `_ctdata.txt` files are not used.
* Centre of rotation: `ObjectOffsetX` (at the object, so magnified), or
  the mean of `CentreOfRotationTop/Bottom` (at the detector). A scan's own
  file often has 0 and `AutomaticCentreOfRotation=1`; Nikon's software
  stores the centre it found in `<Name>_recon.xtekct`, which the reader
  uses when present.
* `WhiteLevel`: the open-beam intensity.
* Checked: Korn (Zenodo 6874123) and crystals in clay (Zenodo 4912635),
  CC-BY-4.0. The wrong direction smears every edge; our slices match
  Nikon's reconstruction at 0.84 to 0.87 with the slice order reversed and
  slices transposed, a proper rotation, not a mirror image (if Nikon's
  `.vol` axes are right-handed, which Volume Graphics does not state).
* Not applied yet: `ObjectRoll`, `ObjectTilt` (noted in the metadata when
  set).

### Waygate Technologies / GE phoenix (`.pca`)

INI file next to `<name>NNNNN.tif` and a `.pcp` table of angles.

* `[Geometry]` `FDD`, `FOD`, `cx`, `cy`, `VoxelSizeX/Y`, `CalibValue`;
  `[CT]` `NumberImages`; `[Image]` `DimX/Y`, `FreeRay` (open beam).
* Pixel pitch = `VoxelSizeX` x `FDD`/`FOD`. The pitch computed from the
  detector region (`Left..Right` x `PixelsizeX` / `DimX`) was twice too
  large on the sample, and the reconstruction showed it at once; the
  reconstruction file's `VoxelSizeRec` confirms the formula.
* `CalibValue` (repeated as `CorrectionValue` in the `.pcr`) is a pixel
  offset: as millimetres it doubles every edge; as pixels the best match
  to Waygate's reconstruction lies between 0 and its value.
* Checked: plug scan (Zenodo 14993754, CC-BY-4.0), 0.9935 against
  Waygate's own reconstruction, a 90 degree rotation, not mirrored.
* The flat and offset images named in `[CalibImages]` were applied at
  acquisition; the samples do not include them.

### ZEISS Xradia (`.txrm`, `.txm`, `.xrm`)

OLE2 compound files (needs `olefile`, BSD).

* `ImageInfo/NoOfImages`, `ImageWidth`, `ImageHeight`, `DataType`
  (5 uint16, 10 float32); images in `ImageData{ceil(i/100)}/Image{i}`.
* `ImageInfo/PixelSize`: the voxel size at the object (um); detector pitch
  = voxel x (SOD + ODD) / SOD.
* `ImageInfo/Angles` (degrees), `StoRADistance`, `DtoRADistance` (mm,
  signed).
* `alignment/x-shifts`, `y-shifts`: per-view shifts the ZEISS software
  applies. CIL rolls the images by their whole-pixel part; Tomviz puts
  them into the geometry as per-view detector offsets (ASTRA-style
  vectors), keeping the sub-pixel part and the recorded images untouched.
* `ReferenceData/Image`: the flat field, opened as its own dataset.
* Checked: USB stick scan (Zenodo 4822516, CC-BY-4.0) against ZEISS's own
  reconstruction in the same dataset: 0.968 over the circuit board with
  the shifts applied; 0.70 without them.

### Bruker micro-CT / SkyScan (`.log`)

INI-style log next to `<prefix>NNNN.tif` (or `.bmp`).

* `[System] Camera Pixel Size (um)`, `CameraXYRatio`; `[Acquisition]`
  `Number of Files` (one more than the projections), `Camera binning`,
  `Object to Source (mm)`, `Camera to Source (mm)`, `Optical Axis (line)`,
  `Rotation Step (deg)`, `Rotation Direction`, `Flat Field Correction`.
* Projections are usually flat-field corrected already.
* The horizontal centre of rotation is not in the acquisition log (NRecon
  finds it when reconstructing); it starts at the detector centre.
* Checked: grains scan (Zenodo 12744181, CC-BY-4.0): the direction by
  sharpness. The sign of the optical-axis offset could not be checked
  (it moves slices up or down without blurring them) and there is no
  Bruker reconstruction to compare with; TIGRE's convention is used.

### Not implemented yet

* **YXLON** (`ScanParameter.xml` + raw) and **Diondo** (XML + raw): TIGRE
  has BSD-3 loaders, but no public sample scans were found to check
  conventions against.
* **ZEISS METROTOM** (`UncorrectedSingle.uint16` + `0_reco_dll.log`): one
  young reader (xscat, BSD-3) with hard-coded frame counts; no samples.
* **North Star Imaging** `.nsihdr/.nsidat` (volumes): the newer binary
  header is only readable through NSI's Windows DLL.
* **Rigaku, Thermo Fisher HeliScan, TESCAN, Shimadzu**: nothing public;
  these export TIFF stacks or exchange formats.
* **Scanco** `.isq/.aim/.rsq`: ITK's IOScanco module (Apache-2.0) reads
  them; Tomviz already uses ITK, so this is mostly wiring.
* **Volume Graphics** `.vgi` + `.vol`: a small text header plus raw data
  (the Nikon crystals sample has one); the only open reader (simplnx) is
  AGPL, so write our own from the sample.
* **Avizo/Amira** `.am`: ahds (Apache-2.0) reads it.

## Synchrotron and standard formats

### NeXus NXtomo

HDF5 with a group whose `definition` is `NXtomo` (the NXtomo application
definition). Tomviz now recognises it before the generic HDF5 reader:

* `instrument/detector/data` split by `instrument/detector/image_key`
  (0 projection, 1 flat, 2 dark, 3 invalid);
* `sample/rotation_angle` with its units;
* `x_pixel_size`/`y_pixel_size` (or ESRF's `real_x_pixel_size`) for the
  spacing; the sample-detector `distance` kept in the metadata.

Facilities differ in where the group sits (ESRF `/entry0000`, Diamond
`/entry1/tomo_entry`); the first NXtomo group found is used, as httomo
does. Checked on Diamond's `tomo_standard.nxs` (httomo test data).

Not handled yet: HDF5 external links to detector data in sibling files
(Diamond DIAD, MAX IV DanMAX); whether the HDF5 path follows them needs a
test with such a file. ESRF BLISS raw files are not NXtomo: convert them
with `nxtomomill h52nx` (MIT).

### Data Exchange (APS)

Already read: `/exchange/data`, `data_white`, `data_dark`, `theta`. The
flat and dark frames used to be dropped (kept as an unused pointer); they
now open as datasets of their own. Not read yet: pixel size and energy
(the spec path and the path APS 2-BM actually writes differ, and their
units were not verified), and the spec's default angles when `theta` is
missing (the spec does not say whether 180 degrees is included).

### Other synchrotrons and neutron facilities

Elettra SYRMEP, SLS TOMCAT (legacy), ALS 8.3.2 (legacy), PETRA III P05,
the Australian Synchrotron, ISIS IMAT, ORNL CG-1D/VENUS and PSI write TIFF
or FITS stacks with naming conventions and log files for flats, darks and
angles (for example IMAT's `Projection: N angle: ...` log lines, ORNL's
angle in the file name or TIFF tag 65039). A generic *projection folder*
importer with per-facility presets would cover all of them with tifffile
and astropy (both permissive). Not built yet.

### Volume exchange formats

MetaImage, VTI, OME-TIFF and DICOM already open. NRRD (pynrrd, MIT) and
NIfTI (nibabel, MIT) would be small readers; OME-Zarr needs care across
its 0.4, 0.5 (Zarr v3) and 0.6 metadata.

## Electron tomography

### MRC stacks with tilt angles

MRC has no tilt-angle field. Tomviz now looks, in this order (the first
list with one angle per image wins):

1. for an aligned stack (`.ali`), IMOD's refined `<name>.tlt`;
2. the extended header: FEI1/FEI2 (Thermo Fisher Tomography: "Alpha
   tilt", float64 at byte 100 per the MRC2014 spec PN 306687, with IMOD's
   caveat that old files hold radians), SERI (SerialEM: int16 x 100 when
   bit 1 of `nreal` is set), and the legacy 128-byte FEI header;
3. the autodoc `<file>.mdoc` or `<name>.mdoc` (`TiltAngle` per
   `[ZValue = n]`);
4. IMOD's `<name>.rawtlt`, then `<name>.tlt`.

Checked on ETSpy test files: the legacy header of `HAADF.mrc` gives the
same 77 angles as its `.rawtlt`; SerialEM headers agree with their `.mdoc`
to 0.01 degree. A `.rec` (a reconstruction) never looks for angles.

All MRC extensions now open through one path in nanometres. Before, `.mrc`
kept Angstrom while `.st/.ali/.rec` were converted to nm.

### Gatan DigitalMicrograph (`.dm3`, `.dm4`)

Tomviz had a parser but no reader. Stacks whose z axis is calibrated in
degrees (Gatan tomography acquisition) open as tilt series. Not read yet:
the measured per-image angles in the Tomography tracking tags, and series
of one file per tilt (`Stage Alpha` in each).

### Thermo Fisher Velox EMD

HDF5 with `/Version` JSON `"format": "Velox"`, recognised before the
Berkeley EMD reader. Images are `/Data/Image/<uuid>/Data` (y, x,
frames); each frame's JSON metadata gives `Stage.AlphaTilt` (radians)
and `BinaryResult.PixelSize` (metres). A stack whose frames' tilts differ
opens as a tilt series. With several detectors in one file the first
image is read (the detector is named in the metadata). rosettasciio
reads only the first frame's tilt.

### Not implemented yet

* **TIA `.ser`/`.emi`**: Tomviz has `ser.py`; per-file tilt is
  `Stage_A_deg` in the `.emi` (as rosettasciio names it).
* **JEOL TEMography, Hitachi**: formats and angle storage not public.

## Licences

Readers here are written from specifications, real files and the key
names of permissive readers (CIL Apache-2.0, TIGRE BSD-3, dxchange and
olefile BSD-style, mrcfile BSD-3). GPL or AGPL code (rosettasciio,
simplnx, OMEGA, tomojax) was used only to learn file layouts, never
copied. Sample scans used for checking are CC-BY-4.0 (Zenodo) or public
test data (ETSpy, NIST; httomo); none is committed to the repository.

## Next steps, by value

1. **A projection-folder importer** with presets for the TIFF/FITS
   facilities (and neutron imaging).
2. **Data Exchange metadata** (pixel size, energy) once checked on an APS
   file, and HDF5 external links (DIAD, DanMAX).
3. **Scanco** via ITK, **Volume Graphics** `.vgi/.vol`, **Avizo** `.am`.
4. **YXLON, Diondo** when sample scans can be found.
5. **Nikon ObjectRoll/ObjectTilt** and Waygate `DetectorRot`/`Tilt` as
   per-view vectors (the geometry supports it already).
