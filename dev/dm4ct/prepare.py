"""DM4CT F3_1: zip of TIFFs -> one Data Exchange file on the NVMe.
    python -I prepare.py ZIP OUT.h5
Reads members straight from the zip (no extraction to disk), so member
names never become paths."""
import io
import re
import sys
import zipfile

import h5py
import numpy as np
import tifffile

zpath, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(zpath)
names = {re.sub(r'.*/', '', n): n for n in z.namelist() if n.endswith('.tif')}
kinds = {k: sorted(n for n in names if n.startswith(k + '_'))
         for k in ('tomo', 'flat', 'dark')}
print({k: len(v) for k, v in kinds.items()})
first = tifffile.imread(io.BytesIO(z.read(names[kinds['tomo'][0]])))
print('frame', first.shape, first.dtype)
with h5py.File(out, 'w') as f:
    for key, ds in (('tomo', 'data'), ('flat', 'data_white'),
                    ('dark', 'data_dark')):
        d = f.create_dataset(f'exchange/{ds}',
                             (len(kinds[key]),) + first.shape, first.dtype)
        for i, n in enumerate(kinds[key]):
            d[i] = tifffile.imread(io.BytesIO(z.read(names[n])))
    n = len(kinds['tomo'])
    # As the authors reconstruct: linspace(0, pi, n, endpoint=False).
    f['exchange/theta'] = np.arange(n) * (180.0 / n)
    f['exchange/theta'].attrs['units'] = 'deg'
print('wrote', out)
