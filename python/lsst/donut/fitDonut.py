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
from builtins import input

import numpy as np
from lsst.afw.display.ds9 import mtv
import lsstDebug
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
from .zernikeFitter import ZernikeFitter
from .selectDonut import SelectDonutTask
from .utilities import _getGoodPupilShape, cutoutDonut, _getJacobian

display = lsstDebug.Info(__name__).display


class FitDonutConfig(pexConfig.Config):
    selectDonut = pexConfig.ConfigurableField(
        target = SelectDonutTask,
        doc = """Task to select donuts.  Should yield:
              - stellar, not galactic, donuts,
              - not blended donuts,
              - High SNR donuts.
              """
    )
    jmax = pexConfig.ListField(
        dtype = int,
        default = (4, 11, 21),
        doc = "List indicating the maximum Zernike term to fit in each "
              "fitting iteration.  The result at the end of each iteration "
              "will be used as the initial guess for the subsequent iteration."
    )
    wavelength = pexConfig.Field(
        dtype = float,
        doc = "If specified, use this wavelength (in nanometers) to model "
              "donuts.  If not specified, then use filter effective "
              "wavelength.",
        optional = True
    )
    stampSize = pexConfig.Field(
        dtype = int,
        default = 57,
        doc = "Size of donut postage stamps in pixels"
    )
    ignoredPixelMask = pexConfig.ListField(
        doc = "Names of mask planes to ignore when fitting donut images",
        dtype = str,
        default = ["SAT", "SUSPECT", "BAD"],
        itemCheck = lambda x: x in afwImage.Mask().getMaskPlaneDict().keys()
    )
    flip = pexConfig.Field(
        dtype = bool,
        default = False,
        doc = "Flip image 180 degrees to switch between fitting intra-focal "
              "and extra-focal donuts."
    )
    fitTolerance = pexConfig.Field(
        dtype = float,
        default = 1e-2,
        doc = "Relative error tolerance in fit parameters for lmfit"
    )
    z4Init = pexConfig.Field(
        dtype = float,
        default = 13.0,
        doc = "Initial guess for Z4 defocus parameter in waves"
    )
    z4Range = pexConfig.ListField(
        dtype = float,
        default = [9.0, 18.0],
        doc = "Fitting range for Z4 defocus parameter in waves"
    )
    zRange = pexConfig.ListField(
        dtype = float,
        default = [-2.0, 2.0],
        doc = "Fitting range for Zernike coefficients past Z4 in waves"
    )
    r0Init = pexConfig.Field(
        dtype = float,
        default = 0.2,
        doc = "Initial guess for Fried parameter r0 in meters"
    )
    r0Range = pexConfig.ListField(
        dtype = float,
        default = [0.1, 0.4],
        doc = "Fitting range for Fried parameter r0 in meters"
    )
    centroidRange = pexConfig.ListField(
        dtype = float,
        default = [-2.0, 2.0],
        doc = "Fitting range for donut centroid in pixels with respect to "
              "stamp center"
    )
    fluxRelativeRange = pexConfig.ListField(
        dtype = float,
        default = [0.8, 1.2],
        doc = "Relative fitting range for donut flux with respect to stamp "
              "pixel sum"
    )


class FitDonutTask(pipeBase.CmdLineTask):
    """!Fit a donut images with a wavefront forward model.

    @anchor FitDonutTask_

    @section donut_FitDonut_Purpose  Description

    Selects suitable donut images, and fits a forward model to them.  See the
    task donut.zernikeFitter for more details about the model.

    @section donut_FitDonut_IO  Invoking the Task

    This task is normallly invokable as a CmdLineTask, though it can also be
    invoked as a subtask using the `run` method, in which case a sensorRef and
    optionally that reference's `icSrc` and `icExp` datasets should be passed
    as arguments.

    @section donut_FitDonut_Config  Configuration parameters

    See @ref FitDonutConfig

    @section donut_FitDonut_Debug  Debug variables

    The @link lsst.pipe.base.cmdLineTask.CmdLineTask command line task@endlink
    interface supports a flag `--debug` to import `debug.py` from your
    `$PYTHONPATH`; see @ref baseDebug for more about `debug.py`.

    FitDonutTask has a debug dictionary with the following key:
    <dl>
    <dt>display
    <dd>bool; if True, then output donut, model, and residual images for each
        donut being fit.
    </dl>

    For example, put something like:
    @code{.py}
        import lsstDebug
        def DebugInfo(name):
            di = lsstDebug.getInfo(name)
            if name == "lsst.donut.fitDonut":
                di.display = True

            return di

        lsstDebug.Info = DebugInfo
    @endcode
    into your `debug.py` file and run `fitDonut.py` with the `--debug` flag.

    @section donut_FitDonutTask_Input An example of preparing data for use with fitDonut

    FitDonutTask fits a forward model based on Fourier optics to out-of-focus
    star images.  The task is invoked by running fitDonut.py on the output of
    processCcd.py, which needs to be invoked first to perform basic ISR and
    detection tasks.  Running processCcd.py on donut images is highly likely to
    fail with its default configuration, as the normal assumptions about PSFs
    are not satified for these images.  We recommend running processCcd.py
    (or equivalently, singleFrameDriver) with the following configuration to
    enable the task to complete and produce the necessary output for running
    fitDonut.py.

    @code{.py}
        import lsst.pipe.tasks.processCcd
        assert type(config)==lsst.pipe.tasks.processCcd.ProcessCcdConfig, 'config is of type %s.%s instead of lsst.pipe.tasks.processCcd.ProcessCcdConfig' % (type(config).__module__, type(config).__name__)
        config.charImage.doMeasurePsf = False
        config.charImage.psfIterations = 1
        config.charImage.doApCorr = False
        config.charImage.installSimplePsf.width = 61
        config.charImage.installSimplePsf.fwhm = 20.0
        config.charImage.detection.thresholdValue = 1.5
        config.charImage.detection.doTempLocalBackground = False
        config.charImage.doWriteExposure = True
        config.doCalibrate = False
    @endcode
    """
    ConfigClass = FitDonutConfig
    _DefaultName = "fitDonut"

    def __init__(self, schema=None, **kwargs):
        """!Construct a FitDonutTask
        """
        pipeBase.Task.__init__(self, **kwargs)
        self.makeSubtask("selectDonut")
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        # Note that order of paramNames here must be consistent with order of
        # lmfit.Parameters object setup in zernikeFitter
        paramNames = ["r0", "dx", "dy", "flux"]
        for i in range(4, max(self.config.jmax) + 1):
            paramNames.append("z{}".format(i))
        paramKeys = []
        for paramName in paramNames:
            paramKeys.append(
                schema.addField(
                    "zfit_"+paramName,
                    type = np.float32,
                    doc = "{} param for Zernike fit".format(paramName)
                )
            )
        self.paramKey = afwTable.ArrayFKey(paramKeys)
        self.covKey = afwTable.CovarianceMatrixXfKey.addFields(
            self.schema,
            "zfit",
            paramNames,
            ""
        )
        self.bicKey = schema.addField("bic", type=np.float32)
        self.chisqrKey = schema.addField("chisqr", type=np.float32)
        self.redchiKey = schema.addField("redchi", type=np.float32)
        self.successKey = schema.addField("success", type="Flag")
        self.errorbarsKey = schema.addField("errorbars", type="Flag")

    @pipeBase.timeMethod
    def run(self, sensorRef, icSrc=None, icExp=None):
        """!Fit donuts
        """
        if icSrc is None:
            icSrc = sensorRef.get("icSrc")
        if icExp is None:
            icExp = sensorRef.get("icExp")

        pixelScale = icExp.getWcs().pixelScale()
        self.log.info("display is {}".format(display))
        wavelength = self.getWavelength(icExp)

        visitInfo = icExp.getInfo().getVisitInfo()
        camera = sensorRef.get("camera")
        detector = icExp.getDetector()
        pupilSize, npix = _getGoodPupilShape(
            camera.telescopeDiameter,
            wavelength,
            self.config.stampSize*pixelScale)
        pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, npix)
        nquarter = detector.getOrientation().getNQuarter()
        if self.config.flip:
            nquarter += 2

        donutSrc = self.selectDonut.run(
            icSrc, icExp, self.config.stampSize, self.config.ignoredPixelMask)

        for i, record in enumerate(donutSrc):
            self.log.info("Fitting donut {} of {}".format(
                i + 1, len(donutSrc)))
            result, _ = self.fitOneRecord(
                record, icExp, camera, detector, nquarter, pupilFactory,
                wavelength, pixelScale)
            record.set(self.successKey, result.success)
            if result.success:
                vals = np.array(list(result.params.valuesdict().values()),
                                dtype=np.float32)
                record.set(self.paramKey, vals)
                record.set(self.bicKey, result.bic)
                record.set(self.chisqrKey, result.chisqr)
                record.set(self.redchiKey, result.redchi)
                record.set(self.errorbarsKey, bool(result.errorbars))
                if result.errorbars:
                    record.set(self.covKey, result.covar.astype(np.float32))

        sensorRef.put(donutSrc, "donutSrc")
        return pipeBase.Struct(donutSrc=donutSrc)

    def getWavelength(self, icExp):
        wavelength = self.config.wavelength
        if wavelength is None:
            wavelength = icExp.getFilter().getFilterProperty().getLambdaEff()
            self.log.info(
                ("Using filter effective wavelength of {} nm"
                 .format(wavelength)))
        return wavelength

    def fitOneDonut(self, subMaskedImage, wavelength, pupil, camera,
                    pixelScale, jacobian, alpha):
        result = None
        for jmax in self.config.jmax:
            self.log.info("Fitting with jmax = {}".format(jmax))
            zfitter = ZernikeFitter(
                jmax, wavelength, pupil, camera.telescopeDiameter,
                alpha = alpha,
                maskedImage = subMaskedImage,
                pixelScale = pixelScale,
                jacobian = jacobian,
                ignoredPixelMask = self.config.ignoredPixelMask,
                xtol = self.config.fitTolerance)
            zfitter.initParams(
                z4Init = self.config.z4Init,
                z4Range = self.config.z4Range,
                zRange = self.config.zRange,
                r0Init = self.config.r0Init,
                r0Range = self.config.r0Range,
                centroidRange = self.config.centroidRange,
                fluxRelativeRange = self.config.fluxRelativeRange)
            if result is not None:
                zfitter.params.update(result.params)
            zfitter.fit()
            result = zfitter.result
            self.log.debug(zfitter.report(show_correl=False))
            if display:
                self.displayFitter(zfitter, pupil)
        return result, zfitter

    def getPupilFactory(self, camera, wavelength, pixelScale, visitInfo,
                        oversampling=1.0, padFactor=1.0):
        pupilSize, npix = _getGoodPupilShape(
            camera.telescopeDiameter,
            wavelength,
            self.config.stampSize*pixelScale,
            oversampling=oversampling,
            padFactor=padFactor)
        pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, npix)
        return pupilFactory

    def fitOneRecord(self, record, icExp, camera,
                     nquarter=None, pupilFactory=None,
                     wavelength=None, detector=None, pixelScale=None,
                     alpha=1.0, oversampling=1.0, padFactor=1.0):
        if pixelScale is None:
            pixelScale = icExp.getWcs().pixelScale()
        if detector is None:
            detector = icExp.getDetector()
        if wavelength is None:
            wavelength = self.getWavelength(icExp)
        if pupilFactory is None:
            visitInfo = icExp.getInfo().getVisitInfo()
            pupilFactory = self.getPupilFactory(
                camera, wavelength*alpha, pixelScale, visitInfo,
                oversampling, padFactor
            )
        if nquarter is None:
            nquarter = detector.getOrientation().getNQuarter()

        imX = record.getX()
        imY = record.getY()

        point = afwGeom.Point2D(record.getX(), record.getY())
        jacobian = _getJacobian(detector, point)
        # Need to apply quarter rotations transformation to Jacobian.
        th = np.pi/2*nquarter
        sth, cth = np.sin(th), np.cos(th)
        rot = np.array([[cth, sth], [-sth, cth]])
        jacobian = np.dot(rot.T, np.dot(jacobian, rot))

        fpX = record['base_FPPosition_x']
        fpY = record['base_FPPosition_y']
        self.log.info("Donut is at {}, {}".format(fpX, fpY))
        subMaskedImage = afwMath.rotateImageBy90(
            cutoutDonut(imX, imY, icExp, self.config.stampSize),
            nquarter)
        pupil = pupilFactory.getPupil(afwGeom.Point2D(fpX, fpY))

        return self.fitOneDonut(subMaskedImage, wavelength, pupil, camera,
                                pixelScale, jacobian, alpha=alpha)

    def displayFitter(self, zfitter, pupil):
        data = zfitter.maskedImage.getImage().getArray()
        model = zfitter.constructModelImage(zfitter.result.params)
        resid = data - model
        mtv(afwImage.ImageD(data.astype(np.float64)),
            frame = 1, title = "data")
        mtv(afwImage.ImageD(model.astype(np.float64)),
            frame = 2, title = "model")
        mtv(afwImage.ImageD(resid.astype(np.float64)),
            frame = 3, title = "resid")
        mtv(afwImage.ImageD(pupil.illuminated.astype(np.float64)),
            frame = 4, title = "pupil")
        input("Press Enter to continue...")
