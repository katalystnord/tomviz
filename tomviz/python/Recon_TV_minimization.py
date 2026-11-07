import numpy as np
import tomviz.operators
from tomviz.projectors.siddon import parallelRay
import time


class ReconTVOperator(tomviz.operators.CompletableOperator):

    def transform(self, dataset, Niter=10, Nupdates=0):  # noqa: C901
        """3D Reconstruct from a tilt series using simple TV minimization"""

        self.progress.maximum = 1

        # Get Tilt angles
        tiltAngles = dataset.tilt_angles

        #remove zero tilt anlges
        if np.count_nonzero(tiltAngles) < tiltAngles.size:
            tiltAngles = tiltAngles + 0.001

        # Get Tilt Series
        tiltSeries = dataset.active_scalars
        (Nslice, Nray, Nproj) = tiltSeries.shape

        # Determine the slices for live updates.
        Nupdates = calc_Nupdates(Nupdates, Niter)

        # Generate measurement matrix
        A = parallelRay(Nray, 1.0, tiltAngles, Nray, 1.0) #A is a sparse matrix
        recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F')
        A = A.tocsr()

        (Nslice, Nray, Nproj) = tiltSeries.shape
        (Nrow, Ncol) = A.shape
        rowInnerProduct = np.zeros(Nrow, dtype=np.float32)
        row = np.zeros(Ncol, dtype=np.float32)
        f = np.zeros(Ncol, dtype=np.float32) # Placeholder for 2d image

        ng = 5
        beta = 1.0
        r_max = 1.0
        gamma_red = 0.8

        # Calculate row inner product, preparation for ART recon
        for j in range(Nrow):
            row[:] = A[j, :].toarray()
            rowInnerProduct[j] = np.dot(row, row)

        self.progress.maximum = Niter * Nslice
        t0 = time.time()
        counter = 1
        etcMessage = 'Estimated time to complete: n/a'

        #Create child dataset for recon
        child = dataset.create_child_dataset()

        for i in range(Niter): #main loop

            if self.completed:
                break

            recon_temp = recon.copy()

            #ART recon
            for s in range(Nslice): #

                #In case canceled during ART.
                if self.canceled or self.completed:
                    return

                self.progress.message = 'Slice No.%d/%d, Iteration No.%d/%d. '\
                    % (s + 1, Nslice, i + 1, Niter) + etcMessage

                f[:] = recon[s, :, :].flatten()

                b = tiltSeries[s, :, :].transpose().flatten()

                for j in range(Nrow):
                    row[:] = A[j, :].toarray()
                    a = (b[j] - np.dot(row, f)) / rowInnerProduct[j]
                    f = f + row * a * beta
                recon[s, :, :] = f.reshape((Nray, Nray))

                self.progress.value = i*Nslice + s

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

            if i != (Niter - 1):

                self.progress.message = 'Minimizating the Objects TV'

                #calculate tomogram change due to POCS
                dPOCS = np.linalg.norm(recon_temp - recon)

                recon_temp = recon.copy()

                #3D TV minimization
                for j in range(ng):
                    R_0 = tv(recon)
                    v = tv_derivative(recon)
                    recon_prime = recon - dPOCS * v
                    recon_prime[recon_prime < 0] = 0
                    gamma = 1.0
                    R_f = tv(recon_prime)

                    #Projected Line search
                    while R_f > R_0:
                        gamma = gamma * gamma_red
                        recon_prime = recon - gamma * dPOCS * v
                        recon_prime[recon_prime < 0] = 0
                        R_f = tv(recon_prime)
                    recon = recon_prime

                dg = np.linalg.norm(recon - recon_temp)

                if dg > r_max*dPOCS:
                    recon = r_max*dPOCS/dg*(recon - recon_temp) + recon_temp

        # One last update of the child data.
        child.active_scalars = recon #add recon to child
        self.progress.data = child

        returnValues = {}
        returnValues["reconstruction"] = child
        return returnValues


def tv_derivative(recon):
    r = np.pad(recon, ((1, 1), (1, 1), (1, 1)), 'edge')
    v1n = 3 * r - np.roll(r, 1, axis=0) - \
                          np.roll(r, 1, axis=1) - np.roll(r, 1, axis=2) # noqa TODO reformat this
    v1d = np.sqrt(1e-8 + (r - np.roll(r, 1, axis=0))**2 + (r -
                  np.roll(r, 1, axis=1))**2 + (r - np.roll(r, 1, axis=2))**2) # noqa TODO reformat this

    v2n = r - np.roll(r, -1, axis=0)
    v2d = np.sqrt(1e-8 + (np.roll(r, -1, axis=0) - r)**2 +
            (np.roll(r, -1, axis=0) -  # noqa TODO reformat this
             np.roll(np.roll(r, -1, axis=0), 1, axis=1))**2 +
            (np.roll(r, -1, axis=0) - np.roll(np.roll(r, -1, axis=0), 1, axis=2))**2) # noqa TODO reformat this

    v3n = r - np.roll(r, -1, axis=1)
    v3d = np.sqrt(1e-8 + (np.roll(r, -1, axis=1) - np.roll(np.roll(r, -1, axis=1), 1, axis=0))**2 + # noqa TODO reformat this
                  (np.roll(r, -1, axis=1) - r)**2 + # noqa TODO reformat this
                  (np.roll(r, -1, axis=1) - np.roll(np.roll(r, -1, axis=1), 1, axis=2))**2) # noqa TODO reformat this

    v4n = r - np.roll(r, -1, axis=2)
    v4d = np.sqrt(1e-8 + (np.roll(r, -1, axis=2) - np.roll(np.roll(r, -1, axis=2), 1, axis=0))**2 + # noqa TODO reformat this
                  (np.roll(r, -1, axis=2) -  # noqa TODO reformat this
                  np.roll(np.roll(r, -1, axis=1), 1, axis=1))**2 +
                  (np.roll(r, -1, axis=2) - r)**2) # noqa TODO reformat this

    v = v1n / v1d + v2n / v2d + v3n / v3d + v4n / v4d
    v = v[1:-1, 1:-1, 1:-1]
    v = v / np.linalg.norm(v)
    return v


def tv(recon):
    r = np.pad(recon, ((1, 1), (1, 1), (1, 1)), 'edge')
    tv = np.sqrt(1e-8 + (r - np.roll(r, -1, axis=0))**2 +
                 (r - np.roll(r, -1, axis=1))**2 +
                 (r - np.roll(r, -1, axis=2))**2)
    tv = np.sum(tv[1:-1, 1:-1, 1:-1])
    return tv


def calc_Nupdates(Nupdates, Niter):
    if Nupdates == 0:
        Nupdates = 0
    #If the user selects 100%, update for every slice.
    elif Nupdates == 100:
        Nupdates = 1
    else:
        Nupdates = int(round((Niter*(1 - Nupdates/100))))
    return Nupdates
