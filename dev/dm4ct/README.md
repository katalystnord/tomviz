# DM4CT F3_1: our parallel-beam path against the authors' reconstruction

A real synchrotron scan with its authors' own reconstruction:
[DM4CT on Zenodo](https://zenodo.org/records/15420527) (CC BY 4.0), sample
F3_1: 1200 projections over [0, 180), 20 flats, 20 darks, 679 x 1653
pixels. The authors reconstruct with ASTRA's FBP at a rotation-axis shift
of -61 pixels ([their script](https://github.com/DM4CT/DM4CT/blob/main/reconstruct_rocks_fbp.py)).

    # raw scan (1.9 GB) and a few reference slices of the 6.9 GB
    # reconstruction (the zip allows HTTP range requests)
    python -I prepare.py F3_1_mono.zip dm4ct_f3_1.h5
    python -m tomviz.conebeam.reconstruct_file dm4ct_f3_1.h5 ours.emd \
        --find-axis --pixel-size 1
    python -I compare.py ours.emd reference_slices/ compare.png

Measured 2026-11-08 (RX 9070): the whole 679 x 1653 x 1653 volume in 35 s
from the file. At the authors' axis, our slices correlate 0.993 to 0.999
with theirs inside the sample, after a 90-degree rotation (the ASTRA
geometry predicts a rotation, not a mirror).

Our axis finder gives +62.28 pixels over the whole height; per band of
rows it gives +61.75, +62.25 and +62.72 (rows 100, 339, 578), and two
sharpness measures of our reconstructions agree to a quarter or half
pixel. The rotation axis is tilted by about 1 pixel over 478 rows; the
authors' single value of 61 is below every estimate.
