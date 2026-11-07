import numpy as np
import scipy.sparse as ss
import tomviz.operators
from tomviz.projectors.siddon import parallelRay
import time


class ReconSirtOperator(tomviz.operators.CompletableOperator):

    def transform(self, dataset, Niter=10, stepSize=0.0001,
                  updateMethodIndex=0, Nupdates=0):
        """
        3D Reconstruct from a tilt series using Simultaneous Iterative
        Reconstruction Techniques (SIRT)"""
        self.progress.maximum = 1

        update_methods = ('landweber', 'cimmino', 'component averaging')
        #reference
        """L. Landweber, Amer. J. Math., 73 (1951), pp. 615–624"""
        """G. Cimmino, La Ric. Sci., XVI, Ser. II, Anno IX, 1 (1938),
        pp. 326–333
        """
        """Y. Censor et al, Parallel Comput., 27 (2001), pp. 777–808"""

        # Get Tilt angles
        tiltAngles = dataset.tilt_angles

        #remove zero tilt anlges
        if np.count_nonzero(tiltAngles) < tiltAngles.size:
            tiltAngles = tiltAngles + 0.001

        # Get Tilt Series
        tiltSeries = dataset.active_scalars
        (Nslice, Nray, Nproj) = tiltSeries.shape

        #Check if there's negative values, shift by minimum if true.
        if np.any(tiltSeries < 0):
            tiltSeries -= np.amin(tiltSeries)

        if tiltSeries is None:
            raise RuntimeError("No scalars found!")

        # Determine slice for live updates
        Nupdates = calc_Nupdates(Nupdates, Niter)

        # Generate measurement matrix
        self.progress.message = 'Generating measurement matrix'
        A = parallelRay(Nray, 1.0, tiltAngles, Nray, 1.0) #A is a sparse matrix
        recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F')

        self.progress.maximum = Nslice*Niter + 1
        step = 0

        #create a reconstruction object
        r = SIRT(A, update_methods[updateMethodIndex], Nslice)
        r.initialize()
        step += 1
        self.progress.value = step

        t0 = time.time()
        counter = 1
        etcMessage = 'Estimated time to complete: n/a'

        #create child for recon
        child = dataset.create_child_dataset()

        for i in range(Niter):

            if self.completed:
                break

            for s in range(Nslice):

                if self.canceled or self.completed:
                    return

                self.progress.message = 'Iteration No.%d/%d,Slice No.%d/%d.' % (
                    i + 1, Niter, s + 1, Nslice) + etcMessage

                b = tiltSeries[s, :, :].transpose().flatten()
                recon_slice = recon[s, :, :].flatten()
                recon[s, :, :] = r.recon2(b, recon_slice, stepSize,
                                          s).reshape((Nray, Nray))

                step += 1
                self.progress.value = step

                timeLeft = (time.time() - t0) / counter * (Nslice*Niter -
                                                           counter)
                counter += 1
                timeLeftMin, timeLeftSec = divmod(timeLeft, 60)
                timeLeftHour, timeLeftMin = divmod(timeLeftMin, 60)
                etcMessage = 'Estimated time to complete: %02d:%02d:%02d' % (
                    timeLeftHour, timeLeftMin, timeLeftSec)

                # Give 4 updates for first iteration.
                if Nupdates != 0 and i == 0 and (s + 1) % (Nslice//4) == 0:
                    child.active_scalars = recon
                    self.progress.data = child

            #Positivity constraint.
            recon[recon < 0] = 0

            #Update at the end of each iteration.
            if Nupdates != 0 and (i + 1) % Nupdates == 0:
                child.active_scalars = recon
                self.progress.data = child

        # One last update of the child data.
        child.active_scalars = recon #add recon to child
        self.progress.data = child

        returnValues = {}
        returnValues["reconstruction"] = child
        return returnValues


class SIRT:

    def __init__(self, A, method, Nslice):
        self.A = A
        self.method = method
        (self.Nrow, self.Ncol) = self.A.shape
        #f = Placeholder for 2d image
        self.f = np.zeros(self.Ncol, dtype=np.float32)

    def initialize(self):
        if self.method == 'landweber':
            self.AT = self.A.transpose()
        elif self.method == 'cimmino':
            self.A = self.A.tocsr()
            self.AT = self.A.transpose()
            rowInnerProduct = np.zeros(self.Nrow, dtype=np.float32)
            self.a = np.zeros(self.Ncol, dtype=np.float32)
            # Calculate row inner product, placeholder for matrix rows
            row = np.zeros(self.Ncol, dtype=np.float32)
            for i in range(self.Nrow):
                row[:] = self.A[i, :].toarray()
                rowInnerProduct[i] = np.dot(row, row)
            self.M = ss.diags(1/rowInnerProduct)
        elif self.method == 'component averaging':
            self.A = self.A.tocsr()
            self.AT = self.A.transpose()
            weightedRowProduct = np.zeros(self.Nrow, dtype=np.float32)
            self.a = np.zeros(self.Ncol, dtype=np.float32)

            # Calculate number of non-zero elements in each column
            s = np.zeros(self.Ncol, dtype=np.float32)

            for i in range(self.Ncol):
                s[i] = self.A[:, i].count_nonzero()

            # Calculate weighted row product
            row = np.zeros(self.Ncol) #placeholder for matrix rows
            for i in range(self.Nrow):
                row[:] = self.A[i, :].toarray()
                weightedRowProduct[i] = np.sum(row * row * s)
            self.M = ss.diags(1/weightedRowProduct)
        else:
            print("Invalid update method!")

    def recon2(self, b, x, stepSize, index):
        self.f[:] = x
        if self.method == 'landweber':
            a = self.AT.dot(b - self.A.dot(self.f))
            self.f = self.f + a * stepSize
        elif self.method == 'cimmino':
            self.a[:] = 0
            g = self.M.dot(b - self.A.dot(self.f))
            self.a = self.AT.dot(g)
            self.f = self.f + self.a * stepSize / self.Nrow
        elif self.method == 'component averaging':
            self.a[:] = 0
            g = self.M.dot(b - self.A.dot(self.f))
            self.a = self.AT.dot(g)
            self.f = self.f + self.a * stepSize
        else:
            print("Invalid update method!")
        return self.f


def calc_Nupdates(Nupdates, Niter):
    if Nupdates == 0:
        Nupdates = 0
    #If the user selects 100%, update for every slice.
    elif Nupdates == 100:
        Nupdates = 1
    else:
        Nupdates = int(round((Niter*(1 - Nupdates/100))))
    return Nupdates
