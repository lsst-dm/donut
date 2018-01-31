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
import collections

import numpy as np
from scipy.spatial import KDTree

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pex.exceptions import LengthError
from lsst.daf.persistence import Butler

from .utilities import donutCoords, getCutout


class PairAnalysisRunner(pipeBase.TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # extraId should already have a refList, but intraId won't yet; we have
        # to add that manually since it comes from a different butler.

        # The following may be fragile...
        repoDir = parsedCmd.butler._repos.inputs()[0].repoArgs.root
        root = repoDir.split('rerun')[0]
        intraArgs = dict(root=os.path.join(root, 'rerun', parsedCmd.intraRerun))
        intraButler = Butler(**intraArgs)

        # Make data refs for the intraIds manually, while temporarily placing
        # the intraButler into parsedCmd.
        extraButler, parsedCmd.butler = parsedCmd.butler, intraButler
        parsedCmd.intraId.makeDataRefList(parsedCmd)
        parsedCmd.butler = extraButler

        extraRefList = parsedCmd.extraId.refList
        intraRefList = parsedCmd.intraId.refList

        assert len(extraRefList) == len(intraRefList), \
            "Ref lists are not the same length!"

        return [(ref1, dict(intraRef=ref2))
                for ref1, ref2 in zip(extraRefList, intraRefList)]


class PairBaseConfig(pexConfig.Config):
    matchRadius = pexConfig.Field(
        dtype = float, default = 20.0,
        doc = "Extra/Intra focal donuts within this distance will be"
              " considered a pair"
    )


class PairBaseTask(pipeBase.CmdLineTask):
    RunnerClass = PairAnalysisRunner

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        # Pop doBatch keyword before passing it along to the argument parser
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName,
                                         *args, **kwargs)
        parser.add_argument("--intraRerun", required=True, help="Rerun for intrafocal data")
        parser.add_id_argument("--extraId", datasetType="donutSrc", level="visit",
                               help="extrafocal data ID, e.g. --intraId visit=12345")
        parser.add_id_argument("--intraId", datasetType="donutSrc", level="visit",
                               help="intrafocal data ID, e.g. --extraId visit=23456",
                               doMakeDataRefList=False)
        return parser

    @staticmethod
    def getCatalogs(eCcdRef, iCcdRef):
        return dict(
            extraIcSrc = eCcdRef.get('icSrc'),
            extraDonutSrc = eCcdRef.get('donutSrc'),
            intraIcSrc = iCcdRef.get('icSrc'),
            intraDonutSrc = iCcdRef.get('donutSrc')
        )

    def getPairs(self, catalogs):
        extraIds = np.array([src.getId() for src in catalogs['extraDonutSrc']])
        intraIds = np.array([src.getId() for src in catalogs['intraDonutSrc']])

        extraCoords = donutCoords(catalogs['extraIcSrc'],
                                  catalogs['extraDonutSrc'])
        intraCoords = donutCoords(catalogs['intraIcSrc'],
                                  catalogs['intraDonutSrc'])

        if len(extraCoords) == 0 or len(intraCoords) == 0:
            return []

        intraKDTree = KDTree(intraCoords)

        pairs = []
        for extraIdx, extraCoord in enumerate(extraCoords):
            dist, intraIdx = intraKDTree.query(extraCoord, 1)
            if dist < self.config.matchRadius:
                pairs.append((extraIds[extraIdx], intraIds[intraIdx],
                              0.5*(extraCoord+intraCoords[intraIdx])))
        return pairs


    @staticmethod
    def getIdLists(extraRef, intraRef):
        extraIdList = dict([(ccdRef.dataId['ccd'], ccdRef)
                            for ccdRef in extraRef.subItems('ccd')
                            if ccdRef.datasetExists("donutSrc")])
        extraIdList = collections.OrderedDict(sorted(extraIdList.items()))
        intraIdList = dict([(ccdRef.dataId['ccd'], ccdRef)
                            for ccdRef in intraRef.subItems('ccd')
                            if ccdRef.datasetExists("donutSrc")])
        intraIdList = collections.OrderedDict(sorted(intraIdList.items()))
        return extraIdList, intraIdList


    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None


class TripletAnalysisRunner(pipeBase.TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # focalId should have a refList, but extraId and intraId won't.  So
        # build those manually here.

        # The following may be fragile...
        repoDir = parsedCmd.butler._repos.inputs()[0].repoArgs.root
        root = repoDir.split('rerun')[0]
        extraArgs = dict(root=os.path.join(root, 'rerun', parsedCmd.extraRerun))
        extraButler = Butler(**extraArgs)
        intraArgs = dict(root=os.path.join(root, 'rerun', parsedCmd.intraRerun))
        intraButler = Butler(**intraArgs)

        # Make data refs manually by temporarily replacing the butler in
        # parsedCmd.
        focalButler, parsedCmd.butler = parsedCmd.butler, extraButler
        parsedCmd.extraId.makeDataRefList(parsedCmd)
        parsedCmd.butler = intraButler
        parsedCmd.intraId.makeDataRefList(parsedCmd)
        parsedCmd.butler = focalButler

        focalRefList = parsedCmd.focalId.refList
        extraRefList = parsedCmd.extraId.refList
        intraRefList = parsedCmd.intraId.refList

        assert len(focalRefList) == len(extraRefList) == len(intraRefList), \
            "Ref lists are not the same length!"

        return [(ref1, dict(extraRef=ref2, intraRef=ref3))
                for ref1, ref2, ref3
                in zip(focalRefList, extraRefList, intraRefList)]


class TripletBaseConfig(pexConfig.Config):
    matchRadius = pexConfig.Field(
        dtype = float, default = 20.0,
        doc = "Extra/Intra focal donuts within this distance will be"
              " considered a pair"
    )


class TripletBaseTask(pipeBase.CmdLineTask):
    RunnerClass = TripletAnalysisRunner

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        # Pop doBatch keyword before passing it along to the argument parser
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName,
                                         *args, **kwargs)
        parser.add_argument("--extraRerun", required=True, help="Rerun for extrafocal data")
        parser.add_argument("--intraRerun", required=True, help="Rerun for intrafocal data")
        parser.add_id_argument("--focalId", datasetType="icSrc", level="visit",
                               help="focal data ID, e.g. --focalId visit=12345")
        parser.add_id_argument("--extraId", datasetType="donutSrc", level="visit",
                               help="extrafocal data ID, e.g. --intraId visit=12345",
                               doMakeDataRefList=False)
        parser.add_id_argument("--intraId", datasetType="donutSrc", level="visit",
                               help="intrafocal data ID, e.g. --extraId visit=23456",
                               doMakeDataRefList=False)
        return parser

    @staticmethod
    def getCatalogs(fCcdRef, eCcdRef, iCcdRef):
        focalIcSrc = fCcdRef.get('icSrc')
        calexp = fCcdRef.get('calexp')
        focalIcSrc = TripletBaseTask.getGoodFocalIcSrc(focalIcSrc, calexp)

        return dict(
            focalIcSrc = focalIcSrc,
            extraIcSrc = eCcdRef.get('icSrc'),
            extraDonutSrc = eCcdRef.get('donutSrc'),
            intraIcSrc = iCcdRef.get('icSrc'),
            intraDonutSrc = iCcdRef.get('donutSrc')
        )

    def getTriplets(self, catalogs):
        focalIds = np.array([src.getId() for src in catalogs['focalIcSrc']])
        extraIds = np.array([src.getId() for src in catalogs['extraDonutSrc']])
        intraIds = np.array([src.getId() for src in catalogs['intraDonutSrc']])

        extraCoords = donutCoords(catalogs['extraIcSrc'],
                                  catalogs['extraDonutSrc'])
        intraCoords = donutCoords(catalogs['intraIcSrc'],
                                  catalogs['intraDonutSrc'])
        focalCoords = np.vstack([[src['base_FPPosition_x'], src['base_FPPosition_y']]
                                 for src in catalogs['focalIcSrc']])

        if len(extraCoords) == 0 or len(intraCoords) == 0:
            return []

        intraKDTree = KDTree(intraCoords)
        extraKDTree = KDTree(extraCoords)

        triplets = []
        for focalIdx, focalCoord in enumerate(focalCoords):
            dist, extraIdx = extraKDTree.query(focalCoord, 1)
            if dist < self.config.matchRadius:
                dist, intraIdx = intraKDTree.query(focalCoord, 1)
                if dist < self.config.matchRadius:
                    extraCoord = extraCoords[extraIdx]
                    intraCoord = intraCoords[intraIdx]
                    triplets.append((
                        focalIds[focalIdx],
                        extraIds[extraIdx],
                        intraIds[intraIdx],
                        0.5*(extraCoord+intraCoord)
                    ))
        return triplets

    @staticmethod
    def getIdLists(focalRef, extraRef, intraRef):
        focalIdList = dict([(ccdRef.dataId['ccd'], ccdRef)
                            for ccdRef in focalRef.subItems('ccd')
                            if ccdRef.datasetExists("icSrc")])
        focalIdList = collections.OrderedDict(sorted(focalIdList.items()))
        extraIdList = dict([(ccdRef.dataId['ccd'], ccdRef)
                            for ccdRef in extraRef.subItems('ccd')
                            if ccdRef.datasetExists("donutSrc")])
        extraIdList = collections.OrderedDict(sorted(extraIdList.items()))
        intraIdList = dict([(ccdRef.dataId['ccd'], ccdRef)
                            for ccdRef in intraRef.subItems('ccd')
                            if ccdRef.datasetExists("donutSrc")])
        intraIdList = collections.OrderedDict(sorted(intraIdList.items()))
        return focalIdList, extraIdList, intraIdList

    @staticmethod
    def getGoodFocalIcSrc(focalIcSrc, focalCalexp):
        ignoredPixelMask = ["SAT", "SUSPECT", "BAD"]
        good = []
        for src in focalIcSrc:
            try:
                cutout = getCutout(src.getX(), src.getY(), focalCalexp, 32)
            except LengthError:
                good.append(False)
                continue
            mask = cutout.getMask()
            bitmask = 0x0
            for m in ignoredPixelMask:
                bitmask |= mask.getPlaneBitMask(m)
            badpix = (np.bitwise_and(mask.getArray().astype(np.uint16),
                                     bitmask) != 0)
            good.append(badpix.sum() == 0)
        return focalIcSrc.subset(np.array(good, dtype=np.bool))

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
