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
from __future__ import print_function, division
from future.utils import iteritems

import numpy as np
from lsst.afw.display.ds9 import mtv
import lsstDebug

display = lsstDebug.Info(__name__).display

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
from .zernikeFitter import ZernikeFitter
from .selectDonut import SelectDonutTask


class FitDonutConfig(pexConfig.Config):

    selectDonut = pexConfig.ConfigurableField(
        target=SelectDonutTask,
        doc="""Task to select donuts:
            - Selects sources that look like donuts
            """,
    )

    zmax = pexConfig.ListField(
        dtype=int, default=(4, 11, 21),
        doc="List indicating the maximum Zernike term to fit in each fitting "
            "iteration.  The result at the end of the previous iteration "
            "will be used as the initial guess for the subsequent iteration."
    )

    wavelength = pexConfig.Field(
        dtype=float,
        doc="If specified, use this wavelength (in nanometers) to model "
            "donuts.  If not specified, then use filter effective "
            "wavelength.",
        optional=True
    )

    stampSize = pexConfig.Field(
        dtype=int, default=57,
        doc="Size of donut postage stamps [default: 57]",
    )

    ignoredPixelMask = pexConfig.ListField(
        doc="Names of mask planes to ignore when fitting donut images",
        dtype=str,
        default=["SAT", "SUSPECT", "BAD"],
        itemCheck=lambda x: x in afwImage.MaskU().getMaskPlaneDict().keys(),
    )

    flip = pexConfig.Field(
        dtype=bool, default=False,
        doc="Flip image 180 degrees to switch intra/extra focal fitting."
    )

    fitTolerance = pexConfig.Field(
        dtype=float, default=1e-2,
        doc="Relative error tolerance in fit parameters"
    )

    z4Init = pexConfig.Field(
        dtype=float, default=13.0,
        doc="Initial guess for Z4 defocus parameter in waves. "
            "[default: 13.0]"
    )

    z4Range = pexConfig.ListField(
        dtype=float, default=[9.0, 18.0],
        doc="Fitting range for Z4 defocus parameter in waves.  "
            "[default: [9.0, 18.0]]"
    )

    zRange = pexConfig.ListField(
        dtype=float, default=[-2.0, 2.0],
        doc="Fitting range for Zernike coefficients past Z4 in waves.  "
            "[default: [-2.0, 2.0-]]"
    )

    r0Init = pexConfig.Field(
        dtype=float, default=0.2,
        doc="Initial guess for Fried parameter r0 in meters.  [default: 0.2]"
    )

    r0Range = pexConfig.ListField(
        dtype=float, default=[0.1, 0.4],
        doc="Fitting range for Fried parameter r0 in meters.  "
            "[default: [0.1, 0.4]]"
    )

    centroidRange = pexConfig.ListField(
        dtype=float, default=[-2.0, 2.0],
        doc="Fitting range for donut centroid in pixels.  "
            "[default: [-2.0, 2.0]]"
    )

    fluxRelativeRange = pexConfig.ListField(
        dtype=float, default=[0.8, 1.2],
        doc="Relative fitting range for donut flux.  [default: [0.8, 1.2]]"
    )


class FitDonutTask(pipeBase.CmdLineTask):

    ConfigClass = FitDonutConfig
    _DefaultName = "fitDonut"

    def __init__(self, schema=None, **kwargs):
        """!Construct a FitDonutTask
        """
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("selectDonut")
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        # Note that order of paramNames here must be consistent with order of
        # lmfit.Parameters object setup in zernikeFitter
        paramNames = ["r0", "dx", "dy", "flux"]
        for i in range(4, max(self.config.zmax)+1):
            paramNames.append("z{}".format(i))
        self.paramDict = {}
        self.sigmaKeys = []
        self.covKeys = []
        for i, pi in enumerate(paramNames):
            self.paramDict[pi] = schema.addField(pi, type=np.float32)
            self.sigmaKeys.append(
                    schema.addField(
                            "{}Sigma".format(pi),
                            type=np.float32,
                            doc="uncertainty on {}".format(pi)))
            for pj in paramNames[:i]:
                self.covKeys.append(
                        schema.addField(
                                "{}_{}_Cov".format(pj, pi),
                                type=np.float32,
                                doc="{},{} covariance".format(pj, pi)))
        self.covMatKey = afwTable.CovarianceMatrixXfKey(
                self.sigmaKeys, self.covKeys)
        self.bic = schema.addField("bic", type=np.float32)
        self.chisqr = schema.addField("chisqr", type=np.float32)
        self.redchi = schema.addField("redchi", type=np.float32)

    @pipeBase.timeMethod
    def run(self, sensorRef, icSrc=None, icExp=None):
        """!Fit donuts
        """
        diam = 8.2  # Ack!  Hardcoded for HSC for now.

        if icSrc is None:
            icSrc = sensorRef.get("icSrc")
        if icExp is None:
            icExp = sensorRef.get("icExp")

        pixelScale = icExp.getWcs().pixelScale()
        self.log.info("display is {}".format(display))

        wavelength = self.config.wavelength
        if wavelength is None:
            wavelength = icExp.getFilter().getFilterProperty().getLambdaEff()
            self.log.info(
                    ("Using filter effective wavelength of {} nm"
                     .format(wavelength)))

        visitInfo = icExp.getInfo().getVisitInfo()
        camera = sensorRef.get("camera")
        pupilSize, npix = self._getGoodPupilShape(
                diam, wavelength, self.config.stampSize*pixelScale)
        pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, npix)
        nquarter = icExp.getDetector().getOrientation().getNQuarter()
        if self.config.flip:
            nquarter += 2
        donutCat = self.selectDonut.run(icSrc)

        for record in donutCat:
            imX = record.getX()
            imY = record.getY()
            fpX = record['base_FPPosition_x']
            fpY = record['base_FPPosition_y']
            self.log.info("Fitting donut at {}, {}".format(fpX, fpY))
            subMaskedImage = afwMath.rotateImageBy90(
                    self.cutoutDonut(imX, imY, icExp), nquarter)
            pupil = pupilFactory.getPupil(afwGeom.Point2D(fpX, fpY))

            result = None
            for zmax in self.config.zmax:
                self.log.info("Fitting with zmax = {}".format(zmax))
                zfitter = ZernikeFitter(
                        subMaskedImage, pixelScale,
                        self.config.ignoredPixelMask,
                        zmax, wavelength, pupil, diam,
                        xtol=self.config.fitTolerance)
                zfitter.initParams(
                        z4Init=self.config.z4Init,
                        z4Range=self.config.z4Range,
                        zRange=self.config.zRange,
                        r0Init=self.config.r0Init,
                        r0Range=self.config.r0Range,
                        centroidRange=self.config.centroidRange,
                        fluxRelativeRange=self.config.fluxRelativeRange)
                if result is not None:
                    zfitter.params.update(result.params)
                zfitter.fit()
                result = zfitter.result
                self.log.info(zfitter.report(show_correl=False))
                if display:
                    data = zfitter.image
                    model = zfitter.model(result.params)
                    resid = data - model
                    mtv(afwImage.ImageD(data.astype(np.float64)),
                        frame=1, title="data")
                    mtv(afwImage.ImageD(model.astype(np.float64)),
                        frame=2, title="model")
                    mtv(afwImage.ImageD(resid.astype(np.float64)),
                        frame=3, title="resid")
                    mtv(afwImage.ImageD(pupil.illuminated.astype(np.float64)),
                        frame=4, title="pupil")
                    raw_input("Press Enter to continue...")
            vals = result.params.valuesdict()
            for paramName, paramKey in iteritems(self.paramDict):
                record.set(paramKey, vals[paramName])
            record.set(self.covMatKey, result.covar.astype(np.float32))
            record.set(self.bic, result.bic)
            record.set(self.chisqr, result.chisqr)
            record.set(self.redchi, result.redchi)

        sensorRef.put(donutCat, "donut")

        return donutCat

    def _getGoodPupilShape(self, diam, wavelength, donutSize):
        """!Estimate and appropriate size and shape for the pupil array.

        @param[in]  diam    Diameter of aperture in meters.
        @param[in]  wavelength  Wavelength of light in nanometers.
        @param[in]  donutSize   Size of donut image as afwGeom.Angle.
        @returns    pupilSize, pupilScale in meters.
        """
        # pupilSize equal to twice the aperture diameter Nyquist samples the
        # focal plane.
        pupilSize = 2 * diam
        # Relation between pupil plane size `L` and scale `dL` and focal plane
        # size `theta` and scale `dtheta` is:
        # dL = lambda / theta
        # L = lambda / dtheta
        # So plug in the donut size for theta and return dL for the scale.
        pupilScale = wavelength * 1e-9 / (donutSize.asRadians())
        npix = self._getGoodFFTSize(pupilSize // pupilScale)
        self.log.info("npix = {}".format(npix))
        return pupilSize, npix

    @staticmethod
    def _getGoodFFTSize(n):
        # Return nearest larger power_of_2 or 3*power_of_2
        exp2 = int(np.ceil(np.log(n)/np.log(2)))
        exp3 = int(np.ceil((np.log(n) - np.log(3))/np.log(2)))
        return min(2**exp2, 3*2**exp3)

    @pipeBase.timeMethod
    def cutoutDonut(self, x, y, icExp):
        point = afwGeom.Point2I(int(x), int(y))
        box = afwGeom.Box2I(point, point)
        box.grow(afwGeom.Extent2I(
                self.config.stampSize//2, self.config.stampSize//2))

        subMaskedImage = icExp.getMaskedImage().Factory(
              icExp.getMaskedImage(),
              box,
              afwImage.PARENT
        )
        return subMaskedImage
