import os

import tomviz.nodes


class ReconstructLargeScanToFile(tomviz.nodes.SourceNode):
    """Cone-beam FDK of a scan on disk into an EMD file, a slab at a time,
    for scans and volumes larger than memory. Shows a subsampled view of
    the result; File > Open the .emd for full resolution."""

    def produce(self, scan_file='', whole_folder=False, output_file='',
                intensity='auto', i0=0.0, dark_level=0.0,
                geometry_source='auto', sod=100.0, sdd=200.0,
                pixel_size=0.1, offset_u=0.0, offset_v=0.0, clockwise=False,
                geometry_file='', arc=360.0, filter_name='ram-lak',
                short_scan='auto', volume_size_xy=0, volume_size_axial=0,
                voxel_size=0.0, backend='auto', memory_gb=0.0,
                preview_voxels=256):
        from tomviz.conebeam.reconstruct_file import (read_preview,
                                                      reconstruct_to_file)
        if not scan_file:
            raise RuntimeError('Choose the scan to reconstruct')
        if not output_file:
            raise RuntimeError('Choose where to write the volume (.emd)')
        if not output_file.lower().endswith('.emd'):
            output_file += '.emd'
        path = os.path.dirname(scan_file) if whole_folder else scan_file

        self.progress.maximum = 100
        self.progress.value = 0
        notes = []

        def log(msg):
            notes.append(msg)
            self.progress.message = '; '.join(notes[-3:])

        def progress(fraction):
            self.progress.value = int(round(100 * fraction))

        result = reconstruct_to_file(
            path, output_file, geometry_source=geometry_source, sod=sod,
            sdd=sdd, pixel_size=pixel_size, offset_u=offset_u,
            offset_v=offset_v, clockwise=clockwise,
            geometry_file=geometry_file or None, arc=arc,
            intensity=intensity, i0=i0, dark_level=dark_level,
            filter_name=filter_name, short_scan=short_scan,
            volume_size_xy=volume_size_xy,
            volume_size_axial=volume_size_axial, voxel_size=voxel_size,
            backend=backend, memory_gb=memory_gb, progress=progress,
            should_stop=lambda: self.canceled, log=log)
        if self.canceled:
            return None
        _, timings = result

        data, spacing, stride = read_preview(output_file,
                                             max(preview_voxels, 16) ** 3)
        volume = self.create_dataset()
        volume.set_scalars('FDK', data)
        volume.spacing = spacing
        volume.metadata = {'reconstruction_file': output_file,
                           'preview_stride': stride}
        log(f'written to {output_file} in {timings["total"]:.0f} s'
            + (f'; showing one voxel in {stride} along each axis'
               if stride > 1 else ''))
        return {'volume': volume}
