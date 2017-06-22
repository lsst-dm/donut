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
