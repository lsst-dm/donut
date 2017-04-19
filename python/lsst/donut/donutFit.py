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
from .zernikeFit import ZernikeFit


class DonutFitConfig(pexConfig.Config):

    jmax = pexConfig.ListField(
        dtype=int, default=(4, 11, 21),
        doc="List indicating the maximum Zernike term to fit in each fitting iteration.  The "
            "result at the end of the previous iteration will be used as the initial guess for the "
            "subsequent iteration."
    )

    wavelength = pexConfig.Field(
        dtype=float,
        doc="If specified, use this wavelength (in nanometers) to model donuts.  "
            "If not specified, then use filter effective wavelength.",
        optional=True
    )

    padFactor = pexConfig.Field(
        dtype=float, default=3.0,
        doc="Padding factor to use for pupil image.",
    )

    r1cut = pexConfig.Field(
        dtype=float, default=50.0,
        doc="Rejection cut flux25/flux3 [default: 50.0]",
    )

    r2cut = pexConfig.Field(
        dtype=float, default=1.05,
        doc="Rejection cut flux35/flux25 [default: 1.05]",
    )

    snthresh = pexConfig.Field(
        dtype=float, default=250.0,
        doc="Donut signal-to-noise threshold [default: 250.0]",
    )

    stamp_size = pexConfig.Field(
        dtype=int, default=72,
        doc="Size of donut postage stamps [default: 72]",
    )

    bitmask = pexConfig.Field(
        dtype=int, default=130,
        doc="Bitmask indicating pixels to exclude from fit [default: 130]"
    )

    flip = pexConfig.Field(
        dtype=bool, default=False,
        doc="Flip image 180 degrees to switch intra/extra focal fitting."
    )


class DonutFitTask(pipeBase.CmdLineTask):

    ConfigClass = DonutFitConfig
    _DefaultName = "donutFit"

    def __init__(self, schema=None, **kwargs):
        """!Construct a DonutFitTask
        """
        pipeBase.Task.__init__(self, **kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        self.r0 = schema.addField("r0", type=float)
        self.z = []
        for i in range(4, max(self.config.jmax)+1):
            self.z.append(schema.addField("z{}".format(i), type=float))

    @pipeBase.timeMethod
    def run(self, sensorRef, icSrc=None, icExp=None):
        """!Fit donuts
        """
        if icSrc is None:
            icSrc = sensorRef.get("icSrc")
        if icExp is None:
            icExp = sensorRef.get("icExp")
        self.log.info("display is {}".format(display))
        visitInfo = icExp.getInfo().getVisitInfo()
        camera = sensorRef.get("camera")
        pupilFactory = camera.getPupilFactory(visitInfo, 16.8, 768)
        wavelength = self.config.wavelength
        if wavelength is None:
            wavelength = icExp.getFilter().getFilterProperty().getLambdaEff()
            self.log.info("Using filter effective wavelength of {} nm".format(wavelength))
        nquarter = icExp.getDetector().getOrientation().getNQuarter()
        if self.config.flip:
            nquarter += 2
        self.log.info("Nquarter = {}".format(nquarter))
        select = self.selectDonuts(icSrc)
        donutCat = icSrc.subset(select)

        for record in donutCat[0:1]:
            im_x, im_y = record.getX(), record.getY()
            fp_x, fp_y = record['base_FPPosition_x'], record['base_FPPosition_y']
            self.log.info("Fitting donut at {}, {}".format(fp_x, fp_y))
            subexp = afwMath.rotateImageBy90(self.cutoutDonut(im_x, im_y, icExp), nquarter)
            pupil = pupilFactory.getPupil(afwGeom.Point2D(fp_x, fp_y))
            result = None
            for jmax in self.config.jmax:
                self.log.info("Fitting with jmax = {}".format(jmax))
                zfit = ZernikeFit(subexp, jmax, self.config.bitmask, wavelength, pupil, xtol=1e-2)
                if result is not None:
                    zfit.params.update(result.params)
                zfit.fit()
                zfit.report()
                if display:
                    image = zfit.image
                    resid = np.reshape(zfit.result.residual, zfit.image.shape) * zfit.sigma
                    model = zfit.image - resid
                    mtv(afwImage.ImageD(image.astype(np.float64)), frame=1, title="image")
                    mtv(afwImage.ImageD(model.astype(np.float64)), frame=2, title="model")
                    mtv(afwImage.ImageD(resid.astype(np.float64)), frame=3, title="resid")
                    mtv(afwImage.ImageD(pupil.illuminated.astype(np.float64)),
                        frame=4, title="aperture")
                    raw_input("Press Enter to continue...")
                result = zfit.result
            vals = result.params.valuesdict()
            record.set(self.r0, vals['r0'])
            for i, z in zip(range(4, max(self.config.jmax)+1), self.z):
                record.set(z, vals['z{}'.format(i)])

        sensorRef.put(donutCat, "donut")

        return donutCat

    @pipeBase.timeMethod
    def selectDonuts(self, icSrc):
        s2n = (icSrc['base_CircularApertureFlux_25_0_flux'] /
               icSrc['base_CircularApertureFlux_25_0_fluxSigma'])
        rej1 = (icSrc['base_CircularApertureFlux_25_0_flux'] /
                icSrc['base_CircularApertureFlux_3_0_flux'])
        rej2 = (icSrc['base_CircularApertureFlux_35_0_flux'] /
                icSrc['base_CircularApertureFlux_25_0_flux'])

        select = (np.isfinite(s2n) &
                  np.isfinite(rej1) &
                  np.isfinite(rej2))
        for i, s in enumerate(select):
            if not s: continue
            if ((s2n[i] < self.config.snthresh) |
                (rej1[i] < self.config.r1cut) |
                (rej2[i] > self.config.r2cut)):
                select[i] = False
        self.log.info("Selected {} of {} detected donuts.".format(sum(select), len(select)))
        return select

    @pipeBase.timeMethod
    def cutoutDonut(self, x, y, icExp):
        point = afwGeom.Point2I(int(x), int(y))
        box = afwGeom.Box2I(point, point)
        box.grow(afwGeom.Extent2I(self.config.stamp_size//2, self.config.stamp_size//2))

        subMaskedImage = icExp.getMaskedImage().Factory(
              icExp.getMaskedImage(),
              box,
              afwImage.PARENT
        )
        return subMaskedImage
