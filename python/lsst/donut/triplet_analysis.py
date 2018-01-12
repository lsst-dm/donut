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
from future.utils import iteritems

import os
import collections

import galsim
import numpy as np
from scipy.spatial import KDTree
from matplotlib.backends.backend_pdf import PdfPages, FigureCanvasPdf
from matplotlib.figure import Figure
import matplotlib.patches as patches

from lsst.pex.exceptions import LengthError
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.daf.persistence import Butler

from .zernikeFitter import ZernikeFitter
from .utilities import getDonutConfig, _getGoodPupilShape, _getJacobian


def lodToDol(lod):
    d = lod[0]
    keys = list(d.keys())
    out = {}
    for k in keys:
        out[k] = []
    for d in lod:
        for k in keys:
            out[k].append(d[k])
    return out


def subplots(nrow, ncol, **kwargs):
    fig = Figure(**kwargs)
    axes = [[fig.add_subplot(nrow, ncol, i+ncol*j+1)
             for i in range(ncol)]
            for j in range(nrow)]
    return fig, np.array(axes, dtype=object)


def getCutout(x, y, calexp, stampSize):
    point = afwGeom.Point2I(int(x), int(y))
    box = afwGeom.Box2I(point, point)
    box.grow(afwGeom.Extent2I(stampSize//2, stampSize//2))

    subMaskedImage = calexp.getMaskedImage().Factory(
        calexp.getMaskedImage(),
        box,
        afwImage.PARENT
    )
    return subMaskedImage


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


def donutPsf(records, focalCalexp, donutConfig, plotConfig, camera):
    # focal should be representative of all of focal,extra,intra for position.
    icRecord = records['extraIcRecord']

    wavelength = donutConfig.wavelength
    if wavelength is None:
        wavelength = focalCalexp.getFilter().getFilterProperty().getLambdaEff()

    jmax = donutConfig.jmaxs[-1]

    nquarter = focalCalexp.getDetector().getOrientation().getNQuarter()
    if donutConfig.flip:
        nquarter += 2
    visitInfo = focalCalexp.getInfo().getVisitInfo()

    pupilSize, pupilNPix = _getGoodPupilShape(
        camera.telescopeDiameter, wavelength,
        plotConfig.stampSize*plotConfig.pixelScale*afwGeom.arcseconds)
    pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, pupilNPix)

    fpX = icRecord['base_FPPosition_x']
    fpY = icRecord['base_FPPosition_y']
    pupil = pupilFactory.getPupil(afwGeom.Point2D(fpX, fpY))

    detector = focalCalexp.getDetector()
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
    params['r0'] = plotConfig.r0

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


class TripletWhiskerConfig(TripletBaseConfig):
    stampSize = pexConfig.Field(
        dtype = int, default = 128,
        doc = "Pixels across PSF image"
    )
    pixelScale = pexConfig.Field(
        dtype = float, default=0.015,
        doc = "Pixel scale for PSF image"
    )
    r0 = pexConfig.Field(
        dtype = float, default=0.3,
        doc = "Fried parameter for Kolmogorov profile "
              "to convolve with optical model"
    )


class TripletWhiskerTask(TripletBaseTask):
    ConfigClass = TripletWhiskerConfig
    _DefaultName = "TripletWhisker"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, focalRef, extraRef=None, intraRef=None):
        """Process an exposure triplet consisting of an in-focus exposure, an
        extra-focal exposure, and an intra-focal exposure.
        """
        focalVisit = focalRef.dataId['visit']
        self.log.info("Working on triplet: {}/{}/{}".format(
            focalRef.dataId['visit'], extraRef.dataId['visit'], intraRef.dataId['visit']))
        focalIdList, extraIdList, intraIdList = self.getIdLists(focalRef, extraRef, intraRef)

        donutConfig = getDonutConfig(extraRef)

        x = []
        y = []
        predictedMoms = []
        observedMoms = []

        for ccd, fCcdRef in focalIdList.items():
        # for ccd, fCcdRef in focalIdList.items()[::10]:
            try:
                iCcdRef = intraIdList[ccd]
                eCcdRef = extraIdList[ccd]
            except KeyError:
                continue
            catalogs = self.getCatalogs(fCcdRef, eCcdRef, iCcdRef)
            triplets = self.getTriplets(catalogs)
            focalCalexp = fCcdRef.get("calexp")
            self.log.info("Found {} triplets on ccd {}".format(len(triplets), ccd))
            for triplet in triplets:
                records = self.getTripletRecords(catalogs, triplet)
                predictedPsf = donutPsf(records, focalCalexp, donutConfig,
                                        self.config, focalRef.get("camera"))
                rec = records['focalIcRecord']
                observedPsf = getCutout(
                    rec.getX(),
                    rec.getY(),
                    focalCalexp,
                    32
                ).getImage().getArray()
                observedPixelScale = focalCalexp.getWcs().pixelScale()
                x.append(rec['base_FPPosition_x'])
                y.append(rec['base_FPPosition_y'])
                predictedMoms.append(getMoments(predictedPsf, self.config.pixelScale))
                observedMoms.append(getMoments(observedPsf, observedPixelScale.asArcseconds()))

        predictedMoms = lodToDol(predictedMoms)
        observedMoms = lodToDol(observedMoms)

        focalButler = focalRef.getButler()
        outputdir = os.path.dirname(
            os.path.join(
                focalButler.get(
                    "icSrc_filename",
                    visit=focalVisit,
                    ccd=0
                )[0]
            )
        )
        plotdir = os.path.abspath(os.path.join(outputdir, "..", "plots"))
        safeMakeDir(plotdir)

        outfn = os.path.join(
            plotdir,
            "donutTripletWhisker-{:07d}.pdf".format(focalVisit))

        with PdfPages(outfn) as pdf:
            fig, axes = subplots(1, 2, figsize=(8, 4))
            qv0 = self.whisker(axes[0,0], x, y, predictedMoms, color='k', scale=2e-4)
            axes[0,0].quiverkey(qv0, 0.1, 0.9, 0.1, "0.1", coordinates='axes')
            axes[0,0].set_title("model")
            qv1 = self.whisker(axes[0,1], x, y, observedMoms, color='k', scale=2e-4)
            axes[0,1].quiverkey(qv1, 0.1, 0.9, 0.1, "0.1", coordinates='axes')
            axes[0,1].set_title("data")
            fig.tight_layout()
            pdf.savefig(fig, dpi=100)

        outfn = os.path.join(
            plotdir,
            "donutTripletUWhisker-{:07d}.pdf".format(focalVisit))

        with PdfPages(outfn) as pdf:
            fig, axes = subplots(1, 2, figsize=(8, 4))
            qv0 = self.unnormalizedWhisker(axes[0,0], x, y, predictedMoms, color='k', scale=1e-4)
            axes[0,0].quiverkey(qv0, 0.1, 0.9, 0.1, "0.1", coordinates='axes')
            axes[0,0].set_title("model")
            qv1 = self.unnormalizedWhisker(axes[0,1], x, y, observedMoms, color='k', scale=1e-4)
            axes[0,1].quiverkey(qv1, 0.1, 0.9, 0.1, "0.1", coordinates='axes')
            axes[0,1].set_title("data")
            fig.tight_layout()
            pdf.savefig(fig, dpi=100)


    @staticmethod
    def whisker(ax, x, y, moms, **kwargs):
        beta = np.arctan2(moms['e2'], moms['e1'])
        th = 0.5*beta
        e = np.hypot(moms['e1'], moms['e2'])
        dx, dy = e*np.cos(th), e*np.sin(th)

        quiver_dict = dict(
            angles='xy',
            headlength=1e-10,
            headwidth=0,
            headaxislength=0,
            minlength=0,
            pivot='middle',
            scale_units='xy',
            width=0.003,
            scale=0.0002)
        kwargs.update(quiver_dict)
        return ax.quiver(x, y, dx, dy, **kwargs)

    @staticmethod
    def unnormalizedWhisker(ax, x, y, moms, **kwargs):
        e1 = np.array(moms['Mxx']) - np.array(moms['Myy'])
        e2 = 2*np.array(moms['Mxy'])

        beta = np.arctan2(e2, e1)
        th = 0.5*beta
        e = np.hypot(e2, e1)
        dx, dy = e*np.cos(th), e*np.sin(th)

        quiver_dict = dict(
            angles='xy',
            headlength=1e-10,
            headwidth=0,
            headaxislength=0,
            minlength=0,
            pivot='middle',
            scale_units='xy',
            width=0.003)
        kwargs.update(quiver_dict)
        return ax.quiver(x, y, dx, dy, **kwargs)


    @staticmethod
    def getTripletRecords(catalogs, triplet):
        focalId, extraId, intraId, _ = triplet
        return dict(
            focalIcRecord = catalogs['focalIcSrc'].find(focalId),
            extraIcRecord = catalogs['extraIcSrc'].find(extraId),
            extraDonutRecord = catalogs['extraDonutSrc'].find(extraId),
            intraIcRecord = catalogs['intraIcSrc'].find(intraId),
            intraDonutRecord = catalogs['intraDonutSrc'].find(intraId)
        )
