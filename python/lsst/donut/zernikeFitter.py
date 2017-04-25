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


class ZernikeFitter(object):
    """!Class to fit Zernike aberrations of donut images"""
    def __init__(self, maskedImage, pixelScale, ignoredPixelMask, zmax,
                 wavelength, pupil, diam, **kwargs):
        """
        @param maskedImage    maskedImage of donut to fit.
        @param pixelScale     pixel scale of maskedImage as afwGeom.Angle.
        @param ignoredPixelMask   Names of mask planes to ignore when fitting.
        @param zmax        Maximum Zernike order to fit.
        @param wavelength  Wavelength to use for model.
        @param pupil       afwCameraGeom.Pupil for model.
        @param diam        Pupil diameter.
        @param **kwargs    Additional kwargs to pass to lmfit.minimize.
        """
        self.image = maskedImage.getImage().getArray()
        self.sigma = np.sqrt(maskedImage.getVariance().getArray())
        mask = maskedImage.getMask()
        bitmask = reduce(lambda x, y: x | mask.getPlaneBitMask(y),
                         ignoredPixelMask, 0x0)
        self.good = (np.bitwise_and(mask.getArray().astype(np.uint16), bitmask)
                     == 0)
        self.shape = self.image.shape
        self.pixelScale = pixelScale.asArcseconds()

        self.zmax = zmax
        self.wavelength = wavelength
        self.aper = galsim.Aperture(
                diam=diam,
                pupil_plane_im=pupil.illuminated.astype(np.int16),
                pupil_plane_scale=pupil.scale,
                pupil_plane_size=pupil.size)

        self.kwargs = kwargs

    def initParams(self, z4Init, z4Range, zRange, r0Init, r0Range,
                   centroidRange, fluxRelativeRange):
        """Initialize lmfit Parameters object.
        """
        params = lmfit.Parameters()
        params.add('z4', z4Init, min=z4Range[0], max=z4Range[1])
        for i in range(5, self.zmax+1):
            params.add('z{}'.format(i), 0.0, min=zRange[0], max=zRange[1])
        params.add('r0', r0Init, min=r0Range[0], max=r0Range[1])
        params.add('dx', 0.0, min=centroidRange[0], max=centroidRange[1])
        params.add('dy', 0.0, min=centroidRange[0], max=centroidRange[1])
        flux = float(np.sum(self.image))
        params.add('flux', flux,
                   min=fluxRelativeRange[0]*flux,
                   max=fluxRelativeRange[1]*flux)
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
        for i in range(4, self.zmax+1):
            aberrations.append(v['z{}'.format(i)])
        optPsf = galsim.OpticalPSF(lam=self.wavelength,
                                   diam=self.aper.diam,
                                   aper=self.aper,
                                   aberrations=aberrations)
        atmPsf = galsim.Kolmogorov(lam=self.wavelength, r0=v['r0'])
        psf = (galsim.Convolve(optPsf, atmPsf)
               .shift(v['dx'], v['dy'])
               * v['flux'])
        modelImg = psf.drawImage(nx=self.shape[0], ny=self.shape[1],
                                 scale=self.pixelScale)
        return modelImg.array

    def resid(self, params):
        """Compute 'chi' image: (data - model)/sigma

        @param params  lmfit.Parameters object.
        @returns       Unraveled chi vector.
        """
        modelImg = self.model(params)
        chi = (self.image - modelImg) / self.sigma
        return chi[self.good].ravel()

    def report(self, *args, **kwargs):
        """Return a string with fit results."""
        return lmfit.fit_report(self.result, *args, **kwargs)
