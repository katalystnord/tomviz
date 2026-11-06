import numpy as np

import tomviz.operators
from tomviz.conebeam.rings import remove_rings


class RemoveRingArtifactsOperator(tomviz.operators.CancelableOperator):

    def transform(self, dataset, size=0, sigma=3.0, dead_pixels=True):
        """Remove ring artifacts from a tilt series of line integrals."""
        tilt_series = dataset.active_scalars
        if tilt_series is None:
            raise RuntimeError('No scalars found!')
        if tilt_series.ndim != 3:
            raise RuntimeError('Expected a stack of projections')
        # Tomviz (x along the axis, y across it, angle) -> (angle, v, u).
        frames = np.moveaxis(tilt_series, 2, 0)

        self.progress.maximum = 100
        self.progress.value = 0

        def progress(fraction):
            self.progress.value = int(round(100 * fraction))

        try:
            out = remove_rings(frames, size=size or None, sigma=sigma,
                               dead=dead_pixels, progress=progress,
                               should_stop=lambda: self.canceled)
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        if out is None:
            return
        dataset.active_scalars = np.asfortranarray(np.moveaxis(out, 0, 2))
