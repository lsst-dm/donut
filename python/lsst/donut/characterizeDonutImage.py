#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
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
from __future__ import absolute_import, division, print_function
import numpy as np

from lsstDebug import getDebugFrame
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
from lsst.afw.math import BackgroundList
from lsst.afw.table import SourceTable, SourceCatalog, IdFactory
from lsst.meas.algorithms import SubtractBackgroundTask, SourceDetectionTask
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
from lsst.meas.astrom import displayAstrometry
from lsst.obs.base import ExposureIdInfo
from lsst.meas.base import SingleFrameMeasurementTask, CatalogCalculationTask
from lsst.pipe.tasks.repair import RepairTask

__all__ = ["CharacterizeDonutImageConfig", "CharacterizeDonutImageTask"]


class CharacterizeDonutImageConfig(pexConfig.Config):
    """!Config for CharacterizeDonutImageTask"""
    doWrite = pexConfig.Field(
        dtype = bool,
        default = True,
        doc = "Persist results?",
    )
    doWriteExposure = pexConfig.Field(
        dtype = bool,
        default = True,
        doc = "Write icExp and icExpBackground in addition to icSrc? Ignored if doWrite False.",
    )
    background = pexConfig.ConfigurableField(
        target = SubtractBackgroundTask,
        doc = "Configuration for initial background estimation",
    )
    detection = pexConfig.ConfigurableField(
        target = SourceDetectionTask,
        doc = "Detect sources"
    )
    measurement = pexConfig.ConfigurableField(
        target = SingleFrameMeasurementTask,
        doc = "Measure sources"
    )
    catalogCalculation = pexConfig.ConfigurableField(
        target = CatalogCalculationTask,
        doc = "Subtask to run catalogCalculation plugins on catalog"
    )
    installSimplePsf = pexConfig.ConfigurableField(
        target = InstallGaussianPsfTask,
        doc = "Install a simple PSF model",
    )
    repair = pexConfig.ConfigurableField(
        target = RepairTask,
        doc = "Remove cosmic rays",
    )
    checkUnitsParseStrict = pexConfig.Field(
        doc = "Strictness of Astropy unit compatibility check, can be 'raise', 'warn' or 'silent'",
        dtype = str,
        default = "raise",
    )

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        # just detect bright stars; includeThresholdMultipler=10 seems large,
        # but these are the values we have been using
        self.detection.thresholdValue = 5.0
        self.detection.includeThresholdMultiplier = 10.0
        # minimal set of measurements needed to determine PSF
        self.measurement.plugins.names = [
            "base_PixelFlags",
            "base_SdssCentroid",
            "base_SdssShape",
            "base_GaussianFlux",
            "base_PsfFlux",
            "base_CircularApertureFlux",
            "base_FPPosition",
        ]

## \addtogroup LSST_task_documentation
## \{
## \page CharacterizeDonutImageTask
## \ref CharacterizeDonutImageTask_ "CharacterizeDonutImageTask"
## \copybrief CharacterizeDonutImageTask
## \}


class CharacterizeDonutImageTask(pipeBase.CmdLineTask):
    """!Measure bright sources and use this to estimate background and PSF of an exposure

    @anchor CharacterizeDonutImageTask_

    @section pipe_tasks_CharacterizeDonutImage_Contents  Contents

     - @ref pipe_tasks_CharacterizeDonutImage_Purpose
     - @ref pipe_tasks_CharacterizeDonutImage_Initialize
     - @ref pipe_tasks_CharacterizeDonutImage_IO
     - @ref pipe_tasks_CharacterizeDonutImage_Config
     - @ref pipe_tasks_CharacterizeDonutImage_Debug


    @section pipe_tasks_CharacterizeDonutImage_Purpose  Description

    Given an exposure with defects repaired (masked and interpolated over, e.g. as output by IsrTask):
    - detect and measure bright sources
    - repair cosmic rays
    - measure and subtract background
    - measure PSF

    @section pipe_tasks_CharacterizeDonutImage_Initialize  Task initialisation

    @copydoc \_\_init\_\_

    @section pipe_tasks_CharacterizeDonutImage_IO  Invoking the Task

    If you want this task to unpersist inputs or persist outputs, then call
    the `run` method (a thin wrapper around the `characterize` method).

    If you already have the inputs unpersisted and do not want to persist the output
    then it is more direct to call the `characterize` method:

    @section pipe_tasks_CharacterizeDonutImage_Config  Configuration parameters

    See @ref CharacterizeDonutImageConfig

    @section pipe_tasks_CharacterizeDonutImage_Debug  Debug variables

    The @link lsst.pipe.base.cmdLineTask.CmdLineTask command line task@endlink interface supports a flag
    `--debug` to import `debug.py` from your `$PYTHONPATH`; see @ref baseDebug for more about `debug.py`.

    CharacterizeDonutImageTask has a debug dictionary with the following keys:
    <dl>
    <dt>frame
    <dd>int: if specified, the frame of first debug image displayed (defaults to 1)
    <dt>repair_iter
    <dd>bool; if True display image after each repair in the measure PSF loop
    <dt>background_iter
    <dd>bool; if True display image after each background subtraction in the measure PSF loop
    <dt>measure_iter
    <dd>bool; if True display image and sources at the end of each iteration of the measure PSF loop
        See @ref lsst.meas.astrom.displayAstrometry for the meaning of the various symbols.
    <dt>psf
    <dd>bool; if True display image and sources after PSF is measured;
        this will be identical to the final image displayed by measure_iter if measure_iter is true
    <dt>repair
    <dd>bool; if True display image and sources after final repair
    <dt>measure
    <dd>bool; if True display image and sources after final measurement
    </dl>

    For example, put something like:
    @code{.py}
        import lsstDebug
        def DebugInfo(name):
            di = lsstDebug.getInfo(name)  # N.b. lsstDebug.Info(name) would call us recursively
            if name == "lsst.pipe.tasks.characterizeDonutImage":
                di.display = dict(
                    repair = True,
                )

            return di

        lsstDebug.Info = DebugInfo
    @endcode
    into your `debug.py` file and run `calibrateTask.py` with the `--debug` flag.

    Some subtasks may have their own debug variables; see individual Task documentation.
    """

    # Example description used to live here, removed 2-20-2017 by MSSG

    ConfigClass = CharacterizeDonutImageConfig
    _DefaultName = "CharacterizeDonutImage"

    def __init__(self, schema=None, **kwargs):
        """!Construct a CharacterizeDonutImageTask

        @param[in,out] schema  initial schema (an lsst.afw.table.SourceTable), or None
        @param[in,out] kwargs  other keyword arguments for lsst.pipe.base.CmdLineTask
        """
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        if schema is None:
            schema = SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("background")
        self.makeSubtask("installSimplePsf")
        self.makeSubtask("repair")
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask('detection', schema=self.schema)
        self.makeSubtask('measurement', schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask('catalogCalculation', schema=self.schema)
        self._initialFrame = getDebugFrame(self._display, "frame") or 1
        self._frame = self._initialFrame
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)

    @pipeBase.timeMethod
    def run(self, dataRef, exposure=None, background=None, doUnpersist=True):
        """!Characterize a science image and, if wanted, persist the results

        This simply unpacks the exposure and passes it to the characterize method to do the work.

        @param[in] dataRef: butler data reference for science exposure
        @param[in,out] exposure  exposure to characterize (an lsst.afw.image.ExposureF or similar).
            If None then unpersist from "postISRCCD".
            The following changes are made, depending on the config:
            - set psf to the measured PSF
            - subtract background
            - interpolate over cosmic rays
            - update detection and cosmic ray mask planes
        @param[in,out] background  initial model of background already subtracted from exposure
            (an lsst.afw.math.BackgroundList). May be None if no background has been subtracted,
            which is typical for image characterization.
            A refined background model is output.
        @param[in] doUnpersist  if True the exposure is read from the repository
            and the exposure and background arguments must be None;
            if False the exposure must be provided.
            True is intended for running as a command-line task, False for running as a subtask

        @return same data as the characterize method
        """
        self._frame = self._initialFrame  # reset debug display frame
        self.log.info("Processing %s" % (dataRef.dataId))

        if doUnpersist:
            if exposure is not None or background is not None:
                raise RuntimeError("doUnpersist true; exposure and background must be None")
            exposure = dataRef.get("postISRCCD", immediate=True)
        elif exposure is None:
            raise RuntimeError("doUnpersist false; exposure must be provided")

        exposureIdInfo = dataRef.get("expIdInfo")

        charRes = self.characterize(
            exposure = exposure,
            exposureIdInfo = exposureIdInfo,
            background = background,
        )

        if self.config.doWrite:
            dataRef.put(charRes.sourceCat, "icSrc")
            if self.config.doWriteExposure:
                dataRef.put(charRes.exposure, "icExp")
                dataRef.put(charRes.background, "icExpBackground")

        return charRes

    @pipeBase.timeMethod
    def characterize(self, exposure, exposureIdInfo=None, background=None):
        """!Characterize a science image

        Peforms the following operations:
        - Iterate the following once:
            - detect and measure sources and estimate PSF (see detectMeasureAndEstimatePsf for details)
        - interpolate over cosmic rays
        - perform final measurement

        @param[in,out] exposure  exposure to characterize (an lsst.afw.image.ExposureF or similar).
            The following changes are made:
            - update or set psf
            - update detection and cosmic ray mask planes
            - subtract background and interpolate over cosmic rays
        @param[in] exposureIdInfo  ID info for exposure (an lsst.obs.base.ExposureIdInfo).
            If not provided, returned SourceCatalog IDs will not be globally unique.
        @param[in,out] background  initial model of background already subtracted from exposure
            (an lsst.afw.math.BackgroundList). May be None if no background has been subtracted,
            which is typical for image characterization.

        @return pipe_base Struct containing these fields, all from the final iteration
        of detectMeasureAndEstimatePsf:
        - exposure: characterized exposure; image is repaired by interpolating over cosmic rays,
            mask is updated accordingly, and the PSF model is set
        - sourceCat: detected sources (an lsst.afw.table.SourceCatalog)
        - background: model of background subtracted from exposure (an lsst.afw.math.BackgroundList)
        - psfCellSet: spatial cells of PSF candidates (an lsst.afw.math.SpatialCellSet)
        """
        self._frame = self._initialFrame  # reset debug display frame

        self.installSimplePsf.run(exposure=exposure)

        if exposureIdInfo is None:
            exposureIdInfo = ExposureIdInfo()

        # subtract an initial estimate of background level
        background = self.background.run(exposure).background

        dmeRes = self.detectMeasureAndEstimatePsf(
            exposure = exposure,
            exposureIdInfo = exposureIdInfo,
            background = background,
        )

        psf = dmeRes.exposure.getPsf()
        psfSigma = psf.computeShape().getDeterminantRadius()
        psfDimensions = psf.computeImage().getDimensions()
        medBackground = np.median(dmeRes.background.getImage().getArray())
        self.log.info("PSF sigma=%0.2f, dimensions=%s; median background=%0.2f" %
                      (psfSigma, psfDimensions, medBackground))

        self.display("psf", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        # perform final repair with final PSF
        self.repair.run(exposure=dmeRes.exposure)
        self.display("repair", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        # perform final measurement with final PSF, including measuring and applying aperture correction,
        # if wanted
        self.measurement.run(measCat=dmeRes.sourceCat, exposure=dmeRes.exposure,
                             exposureId = exposureIdInfo.expId)
        self.catalogCalculation.run(dmeRes.sourceCat)

        self.display("measure", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        return dmeRes

    @pipeBase.timeMethod
    def detectMeasureAndEstimatePsf(self, exposure, exposureIdInfo, background):
        """!Perform one iteration of detect, measure and estimate PSF

        Performs the following operations:
        - if config.doMeasurePsf or not exposure.hasPsf():
            - install a simple PSF model (replacing the existing one, if need be)
        - interpolate over cosmic rays with keepCRs=True
        - estimate background and subtract it from the exposure
        - detect and measure sources, and subtract a refined background model;
        - if config.doMeasurePsf:
            - measure PSF

        @param[in,out] exposure  exposure to characterize (an lsst.afw.image.ExposureF or similar)
            The following changes are made:
            - update or set psf
            - update detection and cosmic ray mask planes
            - subtract background
        @param[in] exposureIdInfo  ID info for exposure (an lsst.obs_base.ExposureIdInfo)
        @param[in,out] background  initial model of background already subtracted from exposure
            (an lsst.afw.math.BackgroundList).

        @return pipe_base Struct containing these fields, all from the final iteration
        of detect sources, measure sources and estimate PSF:
        - exposure  characterized exposure; image is repaired by interpolating over cosmic rays,
            mask is updated accordingly, and the PSF model is set
        - sourceCat  detected sources (an lsst.afw.table.SourceCatalog)
        - background  model of background subtracted from exposure (an lsst.afw.math.BackgroundList)
        - psfCellSet  spatial cells of PSF candidates (an lsst.afw.math.SpatialCellSet)
        """
        # run repair, but do not interpolate over cosmic rays (do that elsewhere, with the final PSF model)
        self.repair.run(exposure=exposure, keepCRs=True)
        self.display("repair_iter", exposure=exposure)

        if background is None:
            background = BackgroundList()

        sourceIdFactory = IdFactory.makeSource(exposureIdInfo.expId, exposureIdInfo.unusedBits)
        table = SourceTable.make(self.schema, sourceIdFactory)
        table.setMetadata(self.algMetadata)

        detRes = self.detection.run(table=table, exposure=exposure, doSmooth=True)
        sourceCat = detRes.sources
        if detRes.fpSets.background:
            background.append(detRes.fpSets.background)

        self.measurement.run(measCat=sourceCat, exposure=exposure, exposureId=exposureIdInfo.expId)

        measPsfRes = pipeBase.Struct(cellSet=None)
        self.display("measure_iter", exposure=exposure, sourceCat=sourceCat)

        return pipeBase.Struct(
            exposure = exposure,
            sourceCat = sourceCat,
            background = background,
            psfCellSet = measPsfRes.cellSet,
        )

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task.
        """
        sourceCat = SourceCatalog(self.schema)
        sourceCat.getTable().setMetadata(self.algMetadata)
        return {"icSrc": sourceCat}

    def display(self, itemName, exposure, sourceCat=None):
        """Display exposure and sources on next frame, if display of itemName has been requested

        @param[in] itemName  name of item in debugInfo
        @param[in] exposure  exposure to display
        @param[in] sourceCat  source catalog to display
        """
        val = getDebugFrame(self._display, itemName)
        if not val:
            return

        displayAstrometry(exposure=exposure, sourceCat=sourceCat, frame=self._frame, pause=False)
        self._frame += 1
