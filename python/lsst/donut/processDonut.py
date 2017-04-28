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
from lsst.ip.isr import IsrTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.donut.fitDonut import FitDonutTask

__all__ = ["ProcessDonutConfig", "ProcessDonutTask"]


class ProcessDonutConfig(pexConfig.Config):
    """Config for ProcessDonut"""
    isr = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="""Task to perform instrumental signature removal or load a
            post-ISR image; ISR consists of:
            - assemble raw amplifier images into an exposure with image,
              variance and mask planes
            - perform bias subtraction, flat fielding, etc.
            - mask known bad pixels
            - provide a preliminary WCS
            """,
    )
    charImage = pexConfig.ConfigurableField(
        target=CharacterizeImageTask,
        doc="""Task to characterize a science exposure:
            - detect sources, usually at high S/N
            - estimate the background, which is subtracted from the image and
              returned as field "background"
            - estimate a PSF model, which is added to the exposure
            - interpolate over defects and cosmic rays, updating the image,
              variance and mask planes
            """,
    )
    fitDonut = pexConfig.ConfigurableField(
        target=FitDonutTask,
        doc="""Task to select and fit donuts:
            - Selects sources that look like donuts
            - Fit a wavefront forward model to donut images
            """,
    )

    def setDefaults(self):
        self.charImage.installSimplePsf.width = 61
        self.charImage.installSimplePsf.fwhm = 20.0
        self.charImage.detection.thresholdValue = 1.5
        self.charImage.doMeasurePsf = False
        self.charImage.doApCorr = False
        self.charImage.detection.doTempLocalBackground = False


class ProcessDonutTask(pipeBase.CmdLineTask):
    """!Assemble raw data, detect and fit donuts

    @anchor ProcessDonutTask_

    @section pipe_tasks_processDonut_Contents  Contents

     - @ref pipe_tasks_processDonut_Purpose
     - @ref pipe_tasks_processDonut_Initialize
     - @ref pipe_tasks_processDonut_IO
     - @ref pipe_tasks_processDonut_Config
     - @ref pipe_tasks_processDonut_Debug

    @section pipe_tasks_processDonut_Purpose  Description

    Perform the following operations:
    - Call isr to unpersist raw data and assemble it into a post-ISR exposure
    - Call charImage subtract background, repair cosmic rays, and detect and
      measure bright sources
    - Call fitDonut to fit Zernike wavefront models to donut images

    @section pipe_tasks_processDonut_Initialize  Task initialisation

    @copydoc \_\_init\_\_

    @section pipe_tasks_processDonut_IO  Invoking the Task

    This task is primarily designed to be run from the command line.

    The main method is `run`, which takes a single butler data reference for
    the raw input data.

    @section pipe_tasks_processDonut_Config  Configuration parameters

    See @ref processDonutConfig

    @section pipe_tasks_processDonut_Debug  Debug variables

    processDonutTask has no debug output, but its subtasks do.

    Add the option `--help` to see more options.
    """
    ConfigClass = ProcessDonutConfig
    _DefaultName = "processDonut"

    def __init__(self, *args, **kwargs):
        """!
        @param[in,out] kwargs  other keyword arguments for
            lsst.pipe.base.CmdLineTask
        """
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("charImage")
        self.makeSubtask("fitDonut", schema=self.charImage.schema)

    @pipeBase.timeMethod
    def run(self, sensorRef):
        """Process donuts on one CCD

        The sequence of operations is:
        - remove instrument signature
        - characterize image to estimate background and do detection
        - fit donuts

        @param sensorRef: butler data reference for raw data

        @return pipe_base Struct containing these fields:
        - donutCat : object returned by donut fitting task
        """
        self.log.info("Processing %s" % (sensorRef.dataId))

        exposure = self.isr.runDataRef(sensorRef).exposure

        charRes = self.charImage.run(
            dataRef=sensorRef,
            exposure=exposure,
            doUnpersist=False,
        )

        donutRes = self.fitDonut.run(sensorRef, charRes.sourceCat,
                                     charRes.exposure)


