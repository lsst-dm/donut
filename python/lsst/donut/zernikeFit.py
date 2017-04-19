#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import print_function

import lmfit
import galsim
import numpy as np

class ZernikeFit:
    """!Class to fit Zernike aberrations of donut images"""
    def __init__(self, exposure, jmax, bitmask, lam, pupil, **kwargs):
        """
        @param exposure  Exposure object containing image of donut to fit.
        @param zmax      Maximum Zernike order to fit.
        @param bitmask   Bitmask defining bad pixels.
        @param lam       Wavelength to use for model.
        @param pupil     afwCameraGeom.Pupil for model.
        @param **kwargs  Additional kwargs to pass to lmfit.minimize.
        """
        self.exposure = exposure
        self.jmax = jmax
        self.kwargs = kwargs
        self.initParams()
        self.bitmask = bitmask
        self.lam = lam
        self.aper = galsim.Aperture(diam=8.2,
                                    pupil_plane_im=pupil.illuminated.astype(np.int16),
                                    pupil_plane_scale=pupil.scale,
                                    pupil_plane_size=pupil.size)
        self.mask = (np.bitwise_and(self.exposure.getMask().getArray().astype(np.uint16),
                                    self.bitmask) == 0)
        self.image = self.exposure.getImage().getArray()
        self.sigma = np.sqrt(self.exposure.getVariance().getArray())

    def initParams(self):
        """Initialize lmfit Parameters object.
        """
        params = lmfit.Parameters()
        params.add('z4', 13.0, min=9.0, max=18.0)
        for i in range(5, self.jmax+1):
            params.add('z{}'.format(i), 0.0, min=-2.0, max=2.0)
        params.add('r0', 0.2, min=0.1, max=0.4)
        params.add('dx', 0.0, min=-2, max=2)
        params.add('dy', 0.0, min=-2, max=2)
        flux = float(np.sum(self.exposure.getImage().getArray()))
        params.add('flux', flux, min=0.8*flux, max=1.2*flux)
        self.params = params

    def fit(self):
        """Do the fit
        @returns  lmfit result.
        """
        self.result = lmfit.minimize(self.resid, self.params, **self.kwargs)
        return self.result

    def model(self, params):
        """Construct model image from parameters

        @param params  lmfit.Parameters object
        @returns       numpy array image
        """
        v = params.valuesdict()
        aberrations = [0,0,0,0]
        for i in range(4, self.jmax+1):
            aberrations.append(v['z{}'.format(i)])
        optPsf = galsim.OpticalPSF(lam=self.lam,
                                   diam=self.aper.diam,
                                   aper=self.aper,
                                   aberrations=aberrations)
        atmPsf = galsim.Kolmogorov(lam=self.lam, r0=v['r0'])
        psf = (galsim.Convolve(optPsf, atmPsf)
               .shift(v['dx'], v['dy'])
               * v['flux'])
        modelImg = psf.drawImage(nx=73, ny=73, scale=0.168)
        return modelImg.array

    def resid(self, params):
        """Compute 'chi' image.

        @param params  lmfit.Parameters object.
        @returns       Unraveled chi vector.
        """
        modelImg = self.model(params)
        chi = (self.image - modelImg) / self.sigma * self.mask
        return chi.ravel()

    def report(self):
        """Report fit results.
        """
        if not hasattr(self, 'result'):
            self.fit()
        lmfit.report_fit(self.result)
