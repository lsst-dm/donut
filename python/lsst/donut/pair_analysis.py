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

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.persistence import Butler, RepositoryArgs

class PairAnalysisRunner(pipeBase.TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # intraId should already have a refList, but extraId won't yet; we have
        # to add that manually since it comes from a different butler.

        # The following may be fragile...
        rootArgs = parsedCmd.butler._repos.inputs()[-1].repoArgs
        root = rootArgs.root
        extraArgs = dict(root=os.path.join(root, 'rerun', parsedCmd.extraRerun))
        extraButler = Butler(**extraArgs)

        # Make data refs for the extraIds manually, while temporarily the
        # extraButler into parsedCmd.
        intraButler, parsedCmd.butler = parsedCmd.butler, extraButler
        parsedCmd.extraId.makeDataRefList(parsedCmd)
        parsedCmd.butler = intraButler

        intraRefList = parsedCmd.intraId.refList
        extraRefList = parsedCmd.extraId.refList
        return [(ref1, dict(extraRef=ref2))
                for ref1, ref2 in zip(intraRefList, extraRefList)]


class ZernikeParamAnalysisConfig(pexConfig.Config):
    pass


class ZernikeParamAnalysisTask(pipeBase.CmdLineTask):
    ConfigClass = ZernikeParamAnalysisConfig
    _DefaultName = "ZernikeParamAnalysis"
    RunnerClass = PairAnalysisRunner

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, intraRef, extraRef=None):
        """Process a pair of exposures.
        """
        print("intra/extra visit = {}/{}".format(
            intraRef.dataId['visit'], extraRef.dataId['visit']))
        print()


    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        # Pop doBatch keyword before passing it along to the argument parser
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name="ZernikeParamAnalysis",
                                         *args, **kwargs)
        parser.add_argument("--extraRerun", required=True, help="Rerun for extrafocal data")
        parser.add_id_argument("--intraId", datasetType="donutSrc", level="visit",
                               help="intrafocal data ID, e.g. --intraId visit=12345")
        parser.add_id_argument("--extraId", datasetType="donutSrc", level="visit",
                               help="extrafocal data ID, e.g. --extraId visit=23456",
                               doMakeDataRefList=False)
        return parser

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
