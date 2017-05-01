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

import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


class SelectDonutConfig(pexConfig.Config):
    """Config for SelectDonut"""
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


class SelectDonutTask(pipeBase.Task):
    """!Select detected objects for donut fitting.

    @anchor SelectDonutTask_

    @section donut_selectDonut_Purpose  Description

    Select detected objects for donut fitting.

    Good candidates are those that likely correspond to stars and not galaxies,
    are not blended, and have sufficient signal-to-noise ratio.  DM-8644 looked
    into these issues.  The criteria that emerged there and are implemented
    here are:

    - Use the ratio of medium-aperture flux (25 pixels) to small-aperture flux
      (3 pixels) to identify the "hole" in stellar donut images and reject
      galactic donut images.  A good candidate has F25/F3 > ~50 or so.  This
      threshold is `r1cut` in the class configuration.
    - Use the ratio of large-aperture flux (35 pixels) to medium-aperture flux
      to identify blends of donuts.  A good candidate has F35/F25 < 1.05 or so.
      This threshold is `r2cut` in the class configuration.
    - Use F25/F25err as a signal-to-noise ratio measure.  Good candidates have
      F25/F25err >~ 250 or so.  This threshold is `snthresh` in the class
      configuration.

    @section donut_selectDonut_IO  Invoking the Task

    This task is primarily designed to be run as a subtask of fitDonut.  The
    main method is `run`, which takes a SourceCatalog and returns a subset.
    """
    ConfigClass = SelectDonutConfig
    _DefaultName = "selectDonut"

    def __init__(self, *args, **kwargs):
        """!Construct a SelectDonutTask
        """
        pipeBase.Task.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def run(self, icSrc):
        """!Select donuts

        @param icSrc  A SourceCatalog of donut detections.  Must include the
                      columns:
                      base_CircularApertureFlux_35_0_flux
                      base_CircularApertureFlux_25_0_flux
                      base_CircularApertureFlux_3_0_flux
                      base_CircularApertureFlux_25_0_fluxSigma
        @returns      A subset of the input SourceCatalog.
        """
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
        self.log.info(
                ("Selected {} of {} detected donuts."
                 .format(sum(select), len(select))))

        return icSrc.subset(select)
