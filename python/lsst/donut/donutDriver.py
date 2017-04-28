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
from __future__ import absolute_import, division, print_function

from lsst.pipe.base import ArgumentParser, ConfigDatasetType
from lsst.donut.processDonut import ProcessDonutTask
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.ctrl.pool.parallel import BatchParallelTask


class DonutDriverConfig(Config):
    processDonut = ConfigurableField(
        target=ProcessDonutTask, doc="Donut processing task")
    ignoreCcdList = ListField(dtype=int, default=[],
                              doc="List of CCDs to ignore when processing")
    ccdKey = Field(dtype=str, default="ccd",
                   doc="DataId key corresponding to a single sensor")


class DonutDriverTask(BatchParallelTask):
    """Fit donuts in parallel
    """
    ConfigClass = DonutDriverConfig
    _DefaultName = "donutDriver"

    def __init__(self, *args, **kwargs):
        """!
        Constructor

        @param[in,out] kwargs  other keyword arguments for lsst.ctrl.pool.BatchParallelTask
        """
        BatchParallelTask.__init__(self, *args, **kwargs)
        self.ignoreCcds = set(self.config.ignoreCcdList)
        self.makeSubtask("processDonut")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="donutDriver", *args, **kwargs)
        parser.add_id_argument("--id",
                               datasetType=ConfigDatasetType(
                                   name="processDonut.isr.datasetType"),
                               level="sensor",
                               help="data ID, e.g. --id visit=12345 ccd=67")
        return parser

    def run(self, sensorRef):
        """Process a single CCD box of donuts, with scatter-gather-scatter using MPI.
        """
        if sensorRef.dataId[self.config.ccdKey] in self.ignoreCcds:
            self.log.warn("Ignoring %s: CCD in ignoreCcdList" %
                          (sensorRef.dataId))
            return None

        with self.logOperation("processing %s" % (sensorRef.dataId,)):
            return self.processDonut.run(sensorRef)
