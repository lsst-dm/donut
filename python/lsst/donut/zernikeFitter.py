#
# LSST Data Management System
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# See COPYRIGHT file at the top of the source tree.
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
from __future__ import absolute_import, division, print_function

import lmfit
import galsim
import numpy as np

from .utilities import _getGoodPupilShape

class ZernikeModeler(object):
    def __init__(self, camera, visitInfo, maxStampSize, point):
        """
        Note maxStampSize is an afwGeom.Angle type
        """
        self.camera = camera
        self.visitInfo = visitInfo
        self.maxStampSize = maxStampSize
        self.point = point

        # Set some defaults
        self.alpha = 1.0
        self.oversampling = 1.0
        self.padFactor = 1.0
        self.centerPupil = True
        self.r0 = np.inf
        self.dx = 0.0
        self.dy = 0.0
        self.flux = 1.0
        self.aberrations = np.array([0,0,0,0])
        self.jacobian = np.eye(2, dtype=float)
        self.wavelength = 775.0
        self.pixelScale = 0.168
        self.stampSize = 57

        self.psfNeedsUpdate = True

    def updatePupil(self, **kwargs):
        # Update pupil and aper, and any kwargs affecting these, but only if necessary.
        alpha = kwargs.pop('alpha', self.alpha)
        wavelength = kwargs.pop('wavelength', self.wavelength)
        oversampling = kwargs.pop('oversampling', self.oversampling)
        padFactor = kwargs.pop('padFactor', self.padFactor)
        centerPupil = kwargs.pop('centerPupil', self.centerPupil)
        if ((alpha != self.alpha
             or wavelength != self.wavelength
             or oversampling != self.oversampling
             or padFactor != self.padFactor
             or centerPupil != self.centerPupil
             or not hasattr(self, 'pupil'))):
            pupilSize, npix = _getGoodPupilShape(
                self.camera.telescopeDiameter,
                wavelength*alpha,
                self.maxStampSize,
                oversampling = oversampling,
                padFactor = padFactor
            )
            self.alpha = alpha
            self.wavelength = wavelength
            self.oversampling = oversampling
            self.padFactor = padFactor
            self.centerPupil = centerPupil

            pupilFactory = self.camera.getPupilFactory(
                self.visitInfo,
                pupilSize,
                npix,
                doCenter = centerPupil
            )
            self.pupil = pupilFactory.getPupil(self.point)
            self.aper = galsim.Aperture(
                diam = self.camera.telescopeDiameter,
                pupil_plane_im = self.pupil.illuminated.astype(np.int16),
                pupil_plane_scale = self.pupil.scale,
                pupil_plane_size = self.pupil.size)
            self.psfNeedsUpdate = True

    def updatePsf(self, **kwargs):
        r0 = kwargs.pop('r0', self.r0)
        dx = kwargs.pop('dx', self.dx)
        dy = kwargs.pop('dy', self.dy)
        flux = kwargs.pop('flux', self.flux)
        aberrations = np.array(kwargs.pop('aberrations', self.aberrations))
        wavelength = kwargs.pop('wavelength', self.wavelength)
        if ((r0 != self.r0
             or dx != self.dx
             or dy != self.dy
             or flux != self.flux
             or np.any(aberrations != self.aberrations)
             or self.psfNeedsUpdate)):
            self.r0 = r0
            self.dx = dx
            self.dy = dy
            self.flux = flux
            self.aberrations = aberrations
            self.wavelength = wavelength
            self.psf = galsim.OpticalPSF(
                lam = self.wavelength*self.alpha,
                diam = self.aper.diam,
                aper = self.aper,
                aberrations = [a/self.alpha for a in aberrations]
            )
            if r0 != np.inf:
                atmPsf = galsim.Kolmogorov(lam=self.wavelength, r0=r0)
                self.psf = galsim.Convolve(self.psf, atmPsf)
            self.psf = self.psf.shift(dx, dy)*flux
            self.psfNeedsUpdate = False

    def updateDrawParams(self, **kwargs):
        self.jacobian = kwargs.pop('jacobian', self.jacobian)
        self.pixelScale = kwargs.pop('pixelScale', self.pixelScale)
        self.stampSize = kwargs.pop('stampSize', self.stampSize)

    def getModel(self, **kwargs):

        self.updatePupil(**kwargs)
        self.updatePsf(**kwargs)
        self.updateDrawParams(**kwargs)

        wcs = galsim.JacobianWCS(*list(self.pixelScale*self.jacobian.ravel()))
        modelImg = self.psf.drawImage(
            nx = self.stampSize,
            ny = self.stampSize,
            wcs = wcs)
        return modelImg.array


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

    This class can also be used without a target image to fit, in order to draw
    model images given parameters.
    """
    def __init__(self, jmax, wavelength, pupil, diam, pixelScale=None,
                 jacobian=None, maskedImage=None, ignoredPixelMask=None,
                 alpha=1.0, **kwargs):
        """
        @param jmax        Maximum Zernike order to fit.
        @param wavelength  Wavelength to use for model.
        @param pupil       afwCameraGeom.Pupil for model.
        @param diam        Pupil diameter.
        @param maskedImage maskedImage of donut to fit.  May be None if simply
                           using this class to draw models without
                           doing a fit.
        @param pixelScale  pixel scale of maskedImage as afwGeom.Angle, or
                           None if using this class to draw models without
                           doing a fit.
        @param jacobian    An optional 2x2 Jacobian distortion matrix to apply
                           to the forward model.  Note that this is relative to
                           the pixelScale above.  Default is the identity
                           matrix.
        @param ignoredPixelMask  Names of mask planes to ignore when fitting.
                                 May be None if simply using this class to draw
                                 models without doing a fit.
        @param alpha       Wavelength multiplication factor.
        @param **kwargs    Additional kwargs to pass to lmfit.minimize.
        """
        if maskedImage is not None:
            if ignoredPixelMask is None:
                raise ValueError("ignoredPixelMask ")
            self.maskedImage = maskedImage
            mask = self.maskedImage.getMask()
            bitmask = 0x0
            for m in ignoredPixelMask:
                bitmask |= mask.getPlaneBitMask(m)
            self.good = (np.bitwise_and(mask.getArray().astype(np.uint16),
                                        bitmask) == 0)
        if pixelScale is not None:
            self.pixelScale = pixelScale.asArcseconds()
        if jacobian is None:
            jacobian = np.eye(2, dtype=np.float64)
        self.jacobian = jacobian
        self.jmax = jmax
        self.wavelength = wavelength
        self.alpha = alpha
        self.aper = galsim.Aperture(
            diam = diam,
            pupil_plane_im = pupil.illuminated.astype(np.int16),
            pupil_plane_scale = pupil.scale,
            pupil_plane_size = pupil.size)
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
        image = self.maskedImage.getImage().getArray()
        flux = float(np.sum(image))
        params.add('flux', flux,
                   min = fluxRelativeRange[0]*flux,
                   max = fluxRelativeRange[1]*flux)
        params.add('z4', z4Init, min=z4Range[0], max=z4Range[1])
        for i in range(5, self.jmax+1):
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

    def _getOptPsf(self, params):
        aberrations = [0, 0, 0, 0]
        for i in range(4, self.jmax + 1):
            aberrations.append(params['z{}'.format(i)])
        return galsim.OpticalPSF(lam = self.wavelength*self.alpha,
                                 diam = self.aper.diam,
                                 aper = self.aper,
                                 aberrations = [a/self.alpha for a in aberrations])

    def constructModelImage(self, params=None, pixelScale=None, jacobian=None,
                            shape=None):
        """Construct model image from parameters

        @param params      lmfit.Parameters object or python dictionary with
                           param values to use, or None to use self.params
        @param pixelScale  pixel scale in arcseconds to use for model image,
                           or None to use self.pixelScale.
        @param jacobian    An optional 2x2 Jacobian distortion matrix to apply
                           to the forward model.  Note that this is relative to
                           the pixelScale above.  Use self.jacobian if this is
                           None.
        @param shape       (nx, ny) shape for model image, or None to use
                           the shape of self.maskedImage
        @returns       numpy array image
        """
        if params is None:
            params = self.params
        if shape is None:
            shape = self.maskedImage.getImage().getArray().shape
        if pixelScale is None:
            pixelScale = self.pixelScale
        if jacobian is None:
            jacobian = self.jacobian
        try:
            v = params.valuesdict()
        except AttributeError:
            v = params

        optPsf = self._getOptPsf(v)
        if 'r0' in v:
            atmPsf = galsim.Kolmogorov(lam=self.wavelength, r0=v['r0'])
            psf = galsim.Convolve(optPsf, atmPsf)
        else:
            psf = optPsf
        psf = psf.shift(v['dx'], v['dy'])*v['flux']

        wcs = galsim.JacobianWCS(*list(pixelScale*jacobian.ravel()))
        modelImg = psf.drawImage(
            nx = shape[0],
            ny = shape[1],
            wcs = wcs)
        return modelImg.array

    def constructWavefrontImage(self, params=None):
        """Construct an image of the wavefront from parameters

        @param params      lmfit.Parameters object or python dictionary with
                           param values to use, or None to use self.params
        @returns       numpy masked array image
        """
        if params is None:
            params = self.params
        aper = galsim.Aperture(
            diam = self.aper.diam,
            pupil_plane_im = self.aper.illuminated.astype(np.int16),
            pupil_plane_scale = self.aper.pupil_plane_scale,
            pupil_plane_size = self.aper.diam)
        try:
            v = params.valuesdict()
        except AttributeError:
            v = params
        optPsf = self._getOptPsf(v)
        out = np.zeros_like(aper.u)
        out[aper.illuminated] = optPsf._psf.screen_list.wavefront(aper)*self.alpha
        mask = np.logical_not(aper.illuminated)
        return np.ma.masked_array(out, mask)

    def _chi(self, params):
        """Compute 'chi' image: (data - model)/sigma

        @param params  lmfit.Parameters object.
        @returns       Unraveled chi vector.
        """
        modelImg = self.constructModelImage(params)
        image = self.maskedImage.getImage().getArray()
        sigma = self.maskedImage.getVariance().getArray()
        chi = (image - modelImg)/sigma
        return chi[self.good].ravel()

    def report(self, *args, **kwargs):
        """Return a string with fit results."""
        return lmfit.fit_report(self.result, *args, **kwargs)
