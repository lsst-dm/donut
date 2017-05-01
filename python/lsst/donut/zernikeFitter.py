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
    """!Class to fit Zernike aberrations of donut images

    The model is constructed using GalSim, and consists of the convolution of
    an OpticalPSF and a Kolmogorov.  The OpticalPSF part includes the
    specification of an arbitrary number of zernike wavefront aberrations.  For
    now, the Kolmogorov part is isotropic.  The centroid and flux of the model
    are also free parameters.

    Note that to use the fitter, the parameters must be initialized with the
    `.initParams` method, which sets initial parameter guesses and ranges.

    To fit the model then use the `.fit` method which returns an
    lmfit.MinimizerResult (and also become accessible through the .result)
    attribute.
    """
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

        @param z4Init   Initial Z4 aberration value in waves.
        @param z4Range  2-tuple for allowed range of Z4 aberration in waves.
        @param zRange   2-tuple for allowed range of Zernike aberrations higher
                        than 4 in waves.
        @param r0Init   Initial value for Fried parameter r0 in meters.
        @param r0Range  2-tuple for allowed range of r0 in meters.
        @param centroidRange  2-tuple for allowed range of centroid in pixels.
                              Note this is the same for both x and y.
        @param fluxRelativeRange  2-tuple for the allowed range of flux
                                  relative to the pixel sum of the input image.
        """
        # Note that order of parameters here must be consistent with order of
        # parameters in the fitDonut schema.
        params = lmfit.Parameters()
        params.add('r0', r0Init, min=r0Range[0], max=r0Range[1])
        params.add('dx', 0.0, min=centroidRange[0], max=centroidRange[1])
        params.add('dy', 0.0, min=centroidRange[0], max=centroidRange[1])
        flux = float(np.sum(self.image))
        params.add('flux', flux,
                   min=fluxRelativeRange[0]*flux,
                   max=fluxRelativeRange[1]*flux)
        params.add('z4', z4Init, min=z4Range[0], max=z4Range[1])
        for i in range(5, self.zmax+1):
            params.add('z{}'.format(i), 0.0, min=zRange[0], max=zRange[1])
        self.params = params

    def fit(self):
        """Do the fit
        @returns  result as an lmfit.MinimizerResult.
        """
        if not hasattr(self, 'params'):
            raise ValueError("Must run .initParams() before running .fit()")
        self.result = lmfit.minimize(self._chi, self.params, **self.kwargs)
        return self.result

    def model(self, params=None):
        """Construct model image from parameters

        @param params  lmfit.Parameters object or None to use self.params
        @returns       numpy array image
        """
        if params is None:
            params = self.params
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

    def _chi(self, params):
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
