import numpy as np
import tomviz.operators
from tomviz.projectors.siddon import parallelRay
import time


class ReconARTOperator(tomviz.operators.CompletableOperator):

    def transform(self, dataset, Niter=1, Nupdates=0, beta=1.0):
        """
        3D Reconstruction using Algebraic Reconstruction Technique (ART)
        """
        self.progress.maximum = 1

        # Get Tilt angles
        tiltAngles = dataset.tilt_angles

        # Get Tilt Series
        tiltSeries = dataset.active_scalars
        (Nslice, Nray, Nproj) = tiltSeries.shape

        if tiltSeries is None:
            raise RuntimeError("No scalars found!")

        #Check if there's negative values, shift by minimum if true.
        if np.any(tiltSeries < 0):
            tiltSeries -= np.amin(tiltSeries)

        # Determine the slices for live updates.
        Nupdates = calc_Nupdates(Nupdates, Niter)

        # Generate measurement matrix
        self.progress.message = 'Generating measurement matrix'
        A = parallelRay(Nray, 1.0, tiltAngles, Nray, 1.0) #A is a sparse matrix
        recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F')

        A = A.tocsr()
        (Nslice, Nray, Nproj) = tiltSeries.shape
        (Nrow, Ncol) = A.shape
        rowInnerProduct = np.zeros(Nrow, dtype=np.float32)
        row = np.zeros(Ncol, dtype=np.float32)
        f = np.zeros(Ncol, dtype=np.float32) # Placeholder for 2d image

        # Calculate row inner product
        for j in range(Nrow):
            row[:] = A[j, :].toarray()
            rowInnerProduct[j] = np.dot(row, row)

        self.progress.maximum = Nslice*Niter
        step = 0
        t0 = time.time()
        etcMessage = 'Estimated time to complete: n/a'

        #create child for recon
        child = dataset.create_child_dataset()

        counter = 1
        for i in range(Niter):

            if self.completed:
                break

            for s in range(Nslice):
                if self.canceled or self.completed:
                    return

                self.progress.message = 'Iteration No.%d/%d,Slice No.%d/%d.' % (
                    i + 1, Niter, s + 1, Nslice) + etcMessage

                #Initialize slice as zeros on first iteration,
                #or vectorize the slice for the next iter.
                f[:] = recon[s, :, :].flatten()

                b = tiltSeries[s, :, :].transpose().flatten()

                for j in range(Nrow):
                    row[:] = A[j, :].toarray()
                    a = (b[j] - np.dot(row, f))/rowInnerProduct[j]
                    f = f + row * a * beta

                recon[s, :, :] = f.reshape((Nray, Nray))

                # Give 4 updates for first iteration.
                if Nupdates != 0 and i == 0 and (s + 1) % (Nslice//4) == 0:
                    child.active_scalars = recon
                    self.progress.data = child

                step += 1
                self.progress.value = step

                timeLeft = (time.time() - t0) / counter * \
                    (Nslice * Niter - counter)
                counter += 1
                timeLeftMin, timeLeftSec = divmod(timeLeft, 60)
                timeLeftHour, timeLeftMin = divmod(timeLeftMin, 60)
                etcMessage = 'Estimated time to complete: %02d:%02d:%02d' % (
                    timeLeftHour, timeLeftMin, timeLeftSec)

            recon[recon < 0] = 0 #Positivity constraint

            #Update for XX iterations.
            if Nupdates != 0 and (i + 1) % Nupdates == 0:
                child.active_scalars = recon
                self.progress.data = child

        # One last update of the child data.
        child.active_scalars = recon #add recon to child
        self.progress.data = child

        returnValues = {}
        returnValues["reconstruction"] = child
        return returnValues


def calc_Nupdates(Nupdates, Niter):
    if Nupdates == 0:
        Nupdates = 0
    #If the user selects 100%, update for every slice.
    elif Nupdates == 100:
        Nupdates = 1
    else:
        Nupdates = int(round((Niter*(1 - Nupdates/100))))
    return Nupdates
