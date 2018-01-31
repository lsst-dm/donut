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

import os

import numpy as np
import galsim

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as afwCameraGeom
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
from lsst.daf.persistence import NoResults

from .zernikeFitter import ZernikeFitter

# https://stackoverflow.com/questions/21062781/
# "Shortest way to get first item of OrderedDict in Python 3"
def first(collection):
    return next(iter(collection))


def lodToDol(lod):
    """Convert a list of dicts into a dict of lists.
    """
    d = lod[0]
    keys = list(d.keys())
    out = {}
    for k in keys:
        out[k] = []
    for d in lod:
        for k in keys:
            out[k].append(d[k])
    return out


def getMoments(arr, pixelScale):
    try:
        mom = galsim.hsm.FindAdaptiveMom(galsim.Image(arr))
    except:
        nan = float('nan')
        return {'e1':nan, 'e2':nan, 'rsqr':nan, 'Mxx':nan, 'Myy':nan, 'Mxy':nan}
    e1, e2 = mom.observed_shape.e1, mom.observed_shape.e2
    rsqr = 2*mom.moments_sigma**2*pixelScale**2
    Mxy = 0.5*rsqr*e2
    Mxx = 0.5*rsqr*(1.+e1)
    Myy = 0.5*rsqr*(1.-e1)
    return {'e1':e1, 'e2':e2, 'rsqr':rsqr, 'Mxx':Mxx, 'Myy':Myy, 'Mxy':Mxy}


def getDonutConfig(ref):
    try:
        donutConfig = ref.get("fitDonut_config")
    except NoResults:
        donutConfig = ref.get("donutDriver_config").fitDonut
    return donutConfig


def donutCoords(icSrc, donutSrc):
    """ Get coordinates of donuts from join of icSrc and donutSrc catalogs
    Return value is numpy array with shape [ndonuts, 2]
    """
    out = []
    for record in donutSrc:
        icRecord = icSrc.find(record.getId())
        out.append((icRecord['base_FPPosition_x'], icRecord['base_FPPosition_y']))
    if len(out) == 0:
        return np.empty((0,2), dtype=float)
    return np.vstack(out)


def getCutout(x, y, exposure, stampSize):
    """!Cut out a postage stamp image from an exposure

    @param x  X-coordinate of postage stamp center
    @param y  Y-coordinate of postage stamp center
    @param exposure  Exposure from which to cutout stamp.
    @param stampSize  Size of cutout.
    @returns  MaskedImage with cutout.
    """
    point = afwGeom.Point2I(int(x), int(y))
    box = afwGeom.Box2I(point, point)
    box.grow(afwGeom.Extent2I(stampSize//2, stampSize//2))

    subMaskedImage = exposure.getMaskedImage().Factory(
        exposure.getMaskedImage(),
        box,
        afwImage.PARENT
    )
    return subMaskedImage


def markGoodDonuts(icSrc, icExp, stampSize, ignoredPixelMask):
    good = []
    for donut in icSrc:
        subMaskedImage = getCutout(
            donut.getX(), donut.getY(), icExp, stampSize)
        mask = subMaskedImage.getMask()
        bitmask = 0x0
        for m in ignoredPixelMask:
            bitmask |= mask.getPlaneBitMask(m)
        badpix = (np.bitwise_and(mask.getArray().astype(np.uint16),
                                 bitmask) != 0)
        good.append(badpix.sum() == 0)
    return np.array(good, dtype=np.bool)


def _getGoodPupilShape(diam, wavelength, donutSize,
                       oversampling=1.0, padFactor=1.0):
    """!Estimate an appropriate size and shape for the pupil array.

    @param[in]  diam    Diameter of aperture in meters.
    @param[in]  wavelength  Wavelength of light in nanometers.
    @param[in]  donutSize   Size of donut image as afwGeom.Angle.
    @param[in]  oversampling  Amount by which to additionally oversample the image.
    @param[in]  padFactor    Amount by which to additionally pad the image.
    @returns    pupilSize, pupilScale in meters.
    """
    # pupilSize equal to twice the aperture diameter Nyquist samples the
    # focal plane.
    pupilSize = 2*diam*oversampling
    # Relation between pupil plane size `L` and scale `dL` and focal plane
    # size `theta` and scale `dtheta` is:
    # dL = lambda / theta
    # L = lambda / dtheta
    # So plug in the donut size for theta and return dL for the scale.
    pupilScale = wavelength*1e-9/(donutSize.asRadians())/padFactor  # meters
    npix = _getGoodFFTSize(pupilSize//pupilScale)
    return pupilSize, npix


def _getGoodFFTSize(n):
    # Return nearest larger power_of_2 or 3*power_of_2
    exp2 = int(np.ceil(np.log(n)/np.log(2)))
    exp3 = int(np.ceil((np.log(n) - np.log(3))/np.log(2)))
    return min(2**exp2, 3*2**exp3)


def _getJacobian(detector, point):
    # Converting from PIXELS to TAN_PIXELS
    transform = detector.getTransform(afwCameraGeom.PIXELS, afwCameraGeom.TAN_PIXELS)
    return transform.getJacobian(point)


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


def rotateSrcCoords(donutSrc, theta):
    """!Return a new donutSrc catalog with columns added to hold Zernike
    coefficients and focal plane coordinates as if these were measured under
    a rotation of the focal plane by theta.

    @param donutSrc  Input donut SourceCatalog
    @oaram theta     afwGeom.Angle specifying the rotation to apply
    @returns  A new sourceCatalog
    """
    schema = donutSrc.schema
    schemaMapper = afwTable.SchemaMapper(schema, schema)
    for key, field in schema:
        schemaMapper.addMapping(key, field.getName())
    newSchema = schemaMapper.editOutputSchema()

    # Collect parameter names to transform
    paramNames = ["r0", "dx", "dy", "flux"]
    j = 4
    paramName = "z{}".format(j)
    while "zfit_"+paramName in schema:
        paramNames.append(paramName)
        j += 1
        paramName = "z{}".format(j)
    jmax = j - 1

    # Make keys for old columns and new columns
    paramKeys = []
    newParamKeys = []
    for paramName in paramNames:
        newParamKeys.append(
            newSchema.addField(
                "zfit_{}_rot".format(paramName),
                type = np.float32,
                doc = "{} param for rotated zfit".format(paramName)
            )
        )
        paramKeys.append(schema.find("zfit_{}".format(paramName)).key)
    newParamNames = [paramName+"_rot" for paramName in paramNames]
    newParamKey = afwTable.ArrayFKey(newParamKeys)
    paramKey = afwTable.ArrayFKey(paramKeys)
    newCovKey = afwTable.CovarianceMatrixXfKey.addFields(
        newSchema,
        "zfit",
        newParamNames,
        ""
    )
    covKey = afwTable.CovarianceMatrixXfKey(newSchema["zfit"], paramNames)
    fpKey = afwTable.ArrayDKey(
        [schema.find("base_FPPosition_x").key,
         schema.find("base_FPPosition_y").key]
    )
    newFpKeyList = [
        newSchema.addField(
            "base_FPPosition_rot_x",
            type = np.float64),
        newSchema.addField(
            "base_FPPosition_rot_y",
            type = np.float64)]
    newFpKey = afwTable.ArrayDKey(newFpKeyList)

    # Copy unrotated columns
    newDonutSrc = afwTable.SourceCatalog(newSchema)
    newDonutSrc.reserve(len(donutSrc))
    for donut in donutSrc:
        newDonutSrc.addNew().assign(donut, schemaMapper)

    # Collect items to be transformed
    params = np.zeros((len(donutSrc), len(paramNames)), dtype=np.float64)
    covs = np.zeros(
        (len(donutSrc), len(paramNames), len(paramNames)),
        dtype=np.float64
    )
    fps = np.zeros((len(donutSrc), 2), dtype=np.float64)
    for i, r in enumerate(donutSrc):
        params[i] = r.get(paramKey)
        covs[i] = r.get(covKey)
        fps[i] = r.get(fpKey)

    # Assemble transformation matrix
    M = np.eye(len(paramNames), dtype=np.float64)
    rotZ = zernikeRotMatrix(jmax, theta)
    # ignore first four parameters, and Zernike indices smaller than 4
    M[4:, 4:] = rotZ[3:, 3:]
    # transform dx and dy as normal though
    # we can actually just extract the 2d rotation matrix from the j=2 and j=3
    # columns of the full Zernike rotation matrix
    rot2 = rotZ[1:3, 1:3]
    M[1:3, 1:3] = rot2

    # Do the transformation
    newParams = np.dot(M, params.T).T
    newCovs = np.matmul(np.matmul(M.T, covs), M)
    newFps = np.dot(rot2, fps.T).T

    # And write into new columns
    for i, r in enumerate(newDonutSrc):
        r.set(newParamKey, newParams[i].astype(np.float32))
        r.set(newCovKey, newCovs[i].astype(np.float32))
        r.set(newFpKey, newFps[i].astype(np.float64))

    return newDonutSrc


def getDonut(icRecord, icExp, donutConfig):
    """Return numpy array containing donut cutout.
    """
    nquarter = icExp.getDetector().getOrientation().getNQuarter()
    if donutConfig.flip:
        nquarter += 2

    maskedImage = getCutout(icRecord.getX(), icRecord.getY(), icExp,
                            donutConfig.stampSize)
    maskedImage = afwMath.rotateImageBy90(maskedImage, nquarter)
    return maskedImage.getImage().getArray()


def getModel(donutRecord, icRecord, icExp, donutConfig, camera):
    """Return numpy array containing donut model.
    """
    pixelScale = icExp.getWcs().pixelScale()

    wavelength = donutConfig.wavelength
    if wavelength is None:
        wavelength = icExp.getFilter().getFilterProperty().getLambdaEff()
    jmax = donutConfig.jmaxs[-1]

    nquarter = icExp.getDetector().getOrientation().getNQuarter()
    if donutConfig.flip:
        nquarter += 2
    visitInfo = icExp.getInfo().getVisitInfo()

    pupilSize, pupilNPix = _getGoodPupilShape(
        camera.telescopeDiameter, wavelength, donutConfig.stampSize*pixelScale)
    pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, pupilNPix)

    fpX = icRecord['base_FPPosition_x']
    fpY = icRecord['base_FPPosition_y']
    pupil = pupilFactory.getPupil(afwGeom.Point2D(fpX, fpY))

    detector = icExp.getDetector()
    point = afwGeom.Point2D(icRecord.getX(), icRecord.getY())
    if donutConfig.doJacobian:
        jacobian = _getJacobian(detector, point)
        # Need to apply quarter rotations to jacobian
        th = np.pi/2*nquarter
        sth, cth = np.sin(th), np.cos(th)
        rot = np.array([[cth, sth], [-sth, cth]])
        jacobian = np.dot(rot.T, np.dot(jacobian, rot))
    else:
        jacobian = np.eye(2)

    params = {}
    keys = ['r0', 'dx', 'dy', 'flux']
    for j in range(4, jmax + 1):
        keys.append('z{}'.format(j))
    for k in keys:
        params[k] = donutRecord['zfit_jmax{}_{}'.format(jmax, k)]
    zfitter = ZernikeFitter(
        jmax,
        wavelength,
        pupil,
        camera.telescopeDiameter
    )

    model = zfitter.constructModelImage(
        params = params,
        pixelScale = pixelScale.asArcseconds(),
        jacobian = jacobian,
        shape = (donutConfig.stampSize, donutConfig.stampSize))

    return model


def getWavefront(records, exposure, donutConfig, plotConfig, camera):
    """Return numpy array containing wavefront model.
    """
    wavelength = donutConfig.wavelength
    if wavelength is None:
        wavelength = exposure.getFilter().getFilterProperty().getLambdaEff()

    jmax = donutConfig.jmaxs[-1]

    nquarter = exposure.getDetector().getOrientation().getNQuarter()
    if donutConfig.flip:
        nquarter += 2
    visitInfo = exposure.getInfo().getVisitInfo()

    pupilSize, pupilNPix = _getGoodPupilShape(
        camera.telescopeDiameter, wavelength,
        plotConfig.stampSize*plotConfig.pixelScale*afwGeom.arcseconds)
    pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, pupilNPix)

    # extra should be representative of both extra and intra
    fpX = records['extraIcRecord']['base_FPPosition_x']
    fpY = records['extraIcRecord']['base_FPPosition_y']
    pupil = pupilFactory.getPupil(afwGeom.Point2D(fpX, fpY))

    params = {}
    for k in ['z{}'.format(j) for j in range(4, jmax + 1)]:
        params[k] = 0.5*(
            records['extraDonutRecord']['zfit_jmax{}_{}'.format(jmax, k)] +
            records['intraDonutRecord']['zfit_jmax{}_{}'.format(jmax, k)]
        )

    zfitter = ZernikeFitter(
        jmax,
        wavelength,
        pupil,
        camera.telescopeDiameter,
    )

    wf = zfitter.constructWavefrontImage(params=params)
    wf = wf[wf.shape[0]//4:3*wf.shape[0]//4, wf.shape[0]//4:3*wf.shape[0]//4]

    return wf


def getPsf(records, extraIcExp, donutConfig, plotConfig, camera):
    # extra should be representative of both extra and intra
    icRecord = records['extraIcRecord']

    wavelength = donutConfig.wavelength
    if wavelength is None:
        wavelength = extraIcExp.getFilter().getFilterProperty().getLambdaEff()

    jmax = donutConfig.jmaxs[-1]

    nquarter = extraIcExp.getDetector().getOrientation().getNQuarter()
    if donutConfig.flip:
        nquarter += 2
    visitInfo = extraIcExp.getInfo().getVisitInfo()

    pupilSize, pupilNPix = _getGoodPupilShape(
        camera.telescopeDiameter, wavelength,
        plotConfig.stampSize*plotConfig.pixelScale*afwGeom.arcseconds)
    pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, pupilNPix)

    fpX = icRecord['base_FPPosition_x']
    fpY = icRecord['base_FPPosition_y']
    pupil = pupilFactory.getPupil(afwGeom.Point2D(fpX, fpY))

    detector = extraIcExp.getDetector()
    point = afwGeom.Point2D(icRecord.getX(), icRecord.getY())
    if donutConfig.doJacobian:
        jacobian = _getJacobian(detector, point)
        # Need to apply quarter rotations to jacobian
        th = np.pi/2*nquarter
        sth, cth = np.sin(th), np.cos(th)
        rot = np.array([[cth, sth], [-sth, cth]])
        jacobian = np.dot(rot.T, np.dot(jacobian, rot))
    else:
        jacobian = np.eye(2)

    params = {}
    for k in ['z{}'.format(j) for j in range(4, jmax + 1)]:
        params[k] = 0.5*(
            records['extraDonutRecord']['zfit_jmax{}_{}'.format(jmax, k)] +
            records['intraDonutRecord']['zfit_jmax{}_{}'.format(jmax, k)]
        )
    params['dx'] = 0.0
    params['dy'] = 0.0
    params['flux'] = 1.0

    zfitter = ZernikeFitter(
        jmax,
        wavelength,
        pupil,
        camera.telescopeDiameter,
        jacobian = jacobian
    )

    psf = zfitter.constructModelImage(
        params = params,
        pixelScale = plotConfig.pixelScale,
        jacobian = jacobian,
        shape = (plotConfig.stampSize, plotConfig.stampSize)
    )

    return psf
