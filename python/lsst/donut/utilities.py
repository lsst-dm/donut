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

import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom


def cutoutDonut(x, y, icExp, stampSize):
    """!Cut out a postage stamp image of a single donut

    @param x  X-coordinate of postage stamp center
    @param y  Y-coordinate of postage stamp center
    @param icExp  Exposure from which to cutout stamp.
    @param stampSize  Size of cutout.
    @returns  MaskedImage with cutout.
    """
    point = afwGeom.Point2I(int(x), int(y))
    box = afwGeom.Box2I(point, point)
    box.grow(afwGeom.Extent2I(stampSize//2, stampSize//2))

    subMaskedImage = icExp.getMaskedImage().Factory(
        icExp.getMaskedImage(),
        box,
        afwImage.PARENT
    )
    return subMaskedImage


def markGoodDonuts(donutSrc, icExp, stampSize, ignoredPixelMask):
    good = []
    for donut in donutSrc:
        subMaskedImage = cutoutDonut(
            donut.getX(), donut.getY(), icExp, stampSize)
        mask = subMaskedImage.getMask()
        bitmask = reduce(lambda x, y: x | mask.getPlaneBitMask(y),
                         ignoredPixelMask, 0x0)
        badpix = (np.bitwise_and(mask.getArray().astype(np.uint16),
                                bitmask) != 0)
        good.append(badpix.sum() == 0)
    return np.array(good, dtype=np.bool)


def _getGoodPupilShape(diam, wavelength, donutSize):
    """!Estimate an appropriate size and shape for the pupil array.

    @param[in]  diam    Diameter of aperture in meters.
    @param[in]  wavelength  Wavelength of light in nanometers.
    @param[in]  donutSize   Size of donut image as afwGeom.Angle.
    @returns    pupilSize, pupilScale in meters.
    """
    # pupilSize equal to twice the aperture diameter Nyquist samples the
    # focal plane.
    pupilSize = 2*diam
    # Relation between pupil plane size `L` and scale `dL` and focal plane
    # size `theta` and scale `dtheta` is:
    # dL = lambda / theta
    # L = lambda / dtheta
    # So plug in the donut size for theta and return dL for the scale.
    pupilScale = wavelength*1e-9/(donutSize.asRadians())  # meters
    npix = _getGoodFFTSize(pupilSize//pupilScale)
    return pupilSize, npix


def _getGoodFFTSize(n):
    # Return nearest larger power_of_2 or 3*power_of_2
    exp2 = int(np.ceil(np.log(n)/np.log(2)))
    exp3 = int(np.ceil((np.log(n) - np.log(3))/np.log(2)))
    return min(2**exp2, 3*2**exp3)


def _getJacobian(detector, point):
    # Converting from PIXELS to TAN_PIXELS
    pixTransform = detector.getTransform(cameraGeom.PIXELS)
    tanPixTransform = detector.getTransform(cameraGeom.TAN_PIXELS)

    transform = afwGeom.MultiXYTransform([
        afwGeom.InvertedXYTransform(pixTransform),
        tanPixTransform])
    affineTransform = transform.linearizeForwardTransform(point)
    linearTransform = affineTransform.getLinear()
    return linearTransform.getParameterVector().reshape(2, 2)


# Start off with the Zernikes up to j=15
_noll_n = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
_noll_m = [0, 0, 1, -1, 0, -2, 2, -1, 1, -3, 3, 0, 2, -2, 4, -4]
def _noll_to_zern(j):
    """
    Convert linear Noll index to tuple of Zernike indices.
    j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike
    index.
    @param [in] j Zernike mode Noll index
    @return (n, m) tuple of Zernike indices
    @see <https://oeis.org/A176988>.
    """
    while len(_noll_n) <= j:
        n = _noll_n[-1] + 1
        _noll_n.extend( [n] * (n+1) )
        if n % 2 == 0:
            _noll_m.append(0)
            m = 2
        else:
            m = 1
        # pm = +1 if m values go + then - in pairs.
        # pm = -1 if m values go - then + in pairs.
        pm = +1 if (n//2) % 2 == 0 else -1
        while m <= n:
            _noll_m.extend([ pm * m , -pm * m ])
            m += 2

    return _noll_n[j], _noll_m[j]


def zernikeRotMatrix(jmax, theta):
    """!Return matrix transforming Zernike polynomial coefficients in one
    coordinate system to the corresponding coefficients for a coordinate
    system rotated by theta.

    @param jmax     Maximum Noll index of Zernike coefficient array
    @param theta    afwGeom Angle indicating coordinate system rotation
    @returns        a jmax by jmax numpy array holding the requested
                    transformation matrix.
    """
    nmax, _ = _noll_to_zern(jmax)
    if (nmax//2 + jmax) % 2 == 0:
        raise ValueError("Must have jmax + nmax//2 odd")

    # Use formula from Tatulli (2013) arXiv:1302.7106v1
    M = np.zeros((jmax, jmax), dtype=np.float64)
    for i in range(jmax):
        ni, mi = _noll_to_zern(i+1)
        for j in range(jmax):
            nj, mj = _noll_to_zern(j+1)
            if ni != nj:
                continue
            if abs(mi) != abs(mj):
                continue
            if mi == mj:
                M[i, j] = np.cos(mj * theta.asRadians())
            elif mi == -mj:
                M[i, j] = np.sin(mj * theta.asRadians())
    return M
