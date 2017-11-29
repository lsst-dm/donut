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

import numpy as np
from scipy.spatial import KDTree
from matplotlib.backends.backend_pdf import PdfPages, FigureCanvasPdf
from matplotlib.figure import Figure
import matplotlib.patches as patches

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.daf.persistence import Butler, RepositoryArgs, NoResults
import lsst.afw.cameraGeom as afwCameraGeom
from lsst.afw.geom import arcseconds, Point2D

from .zernikeFitter import ZernikeFitter
from .utilities import getDonutConfig, _getGoodPupilShape, _getJacobian


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


def subplots(nrow, ncol, **kwargs):
    fig = Figure(**kwargs)
    axes = [[fig.add_subplot(nrow, ncol, i+ncol*j+1)
             for i in range(ncol)]
            for j in range(nrow)]
    return fig, np.array(axes, dtype=object)


def plotCameraOutline(axes, camera):
    axes.tick_params(labelsize=6)
    axes.locator_params(nbins=6)
    axes.ticklabel_format(useOffset=False)
    camRadius = max(camera.getFpBBox().getWidth(),
                    camera.getFpBBox().getHeight())/2
    camRadius = np.round(camRadius, -2)
    camLimits = np.round(1.15*camRadius, -2)
    for ccd in camera:
        ccdCorners = ccd.getCorners(afwCameraGeom.FOCAL_PLANE)
        axes.add_patch(patches.Rectangle(
            ccdCorners[0], *list(ccdCorners[2] - ccdCorners[0]),
            fill=False, edgecolor="k", ls="solid", lw=0.5))
    axes.set_xlim(-camLimits, camLimits)
    axes.set_ylim(-camLimits, camLimits)
    axes.add_patch(patches.Circle(
        (0, 0), radius=camRadius, color="black", alpha=0.1))


def donutPsfWf(records, extraIcExp, donutConfig, plotConfig, camera):
    # extra should be representative of both extra and intra
    icRecord = records['extraIcRecord']

    wavelength = donutConfig.wavelength
    if wavelength is None:
        wavelength = extraIcExp.getFilter().getFilterProperty().getLambdaEff()

    jmax = donutConfig.jmaxs[-1]

    nquarter = extraIcExp.getDetector().getOrientation().getNQuarter()
    if donutConfig.flip:
        nquarter += 2
    visitInfo = extraIcExp.getInfo().getVisitInfo()

    pupilSize, pupilNPix = _getGoodPupilShape(
        camera.telescopeDiameter, wavelength,
        plotConfig.stampSize*plotConfig.pixelScale*arcseconds)
    pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, pupilNPix)

    fpX = icRecord['base_FPPosition_x']
    fpY = icRecord['base_FPPosition_y']
    pupil = pupilFactory.getPupil(Point2D(fpX, fpY))

    detector = extraIcExp.getDetector()
    point = Point2D(icRecord.getX(), icRecord.getY())
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

    wf = zfitter.constructWavefrontImage(params=params)
    wf = wf[wf.shape[0]//4:3*wf.shape[0]//4, wf.shape[0]//4:3*wf.shape[0]//4]

    return psf, wf


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
        parser = pipeBase.ArgumentParser(name="ZernikeParamAnalysis",
                                         *args, **kwargs)
        parser.add_argument("--intraRerun", required=True, help="Rerun for extrafocal data")
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


class ZernikeParamAnalysisConfig(PairBaseConfig):
    pass


class ZernikeParamAnalysisTask(PairBaseTask):
    ConfigClass = ZernikeParamAnalysisConfig
    _DefaultName = "ZernikeParamAnalysis"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, extraRef, intraRef=None):
        """Process a pair of exposures.
        """
        extraVisit = extraRef.dataId['visit']
        self.log.info("Working on extra/intra visits: {}/{}".format(
            extraRef.dataId['visit'], intraRef.dataId['visit']))
        extraIdList, intraIdList = self.getIdLists(extraRef, intraRef)

        donutConfig = getDonutConfig(extraRef)

        x = []
        y = []
        vals = collections.OrderedDict()
        jmax = donutConfig.jmaxs[-1]
        for k in ['z{}'.format(z) for z in range(4, jmax + 1)]:
            vals[k] = []

        for ccd, eCcdRef in extraIdList.items():
            try:
                iCcdRef = intraIdList[ccd]
            except KeyError:
                continue
            self.log.info("Working on ccd {}".format(ccd))
            catalogs = self.getCatalogs(eCcdRef, iCcdRef)
            pairs = self.getPairs(catalogs)
            for pair in pairs:
                extraId, intraId, donutCoord = pair
                x.append(donutCoord[0])
                y.append(donutCoord[1])
                extraRecord = catalogs['extraDonutSrc'].find(extraId)
                intraRecord = catalogs['intraDonutSrc'].find(intraId)
                for k, v in iteritems(vals):
                    v.append(0.5*(extraRecord['zfit_jmax{}_{}'.format(jmax, k)]
                                  + intraRecord['zfit_jmax{}_{}'.format(jmax, k)]))

        extraButler = extraRef.getButler()
        outputdir = os.path.dirname(
            os.path.join(
                extraButler.get(
                    "donutSrc_filename",
                    visit=extraVisit,
                    ccd=0
                )[0]
            )
        )
        plotdir = os.path.abspath(os.path.join(outputdir, "..", "plots"))
        safeMakeDir(plotdir)

        outfn = os.path.join(
            plotdir,
            "donutZernikeParam-{:07d}.pdf".format(extraVisit))

        with PdfPages(outfn) as pdf:
            for k, v in iteritems(vals):
                self.log.info("Plotting {}".format(k))
                fig, axes = subplots(1, 1, figsize=(8, 6.2))
                axes = axes.ravel()[0]
                scatPlot = axes.scatter(x, y, c=v, s=15, linewidths=0.5, cmap='seismic')
                axes.set_title(k)
                plotCameraOutline(axes, extraRef.get("camera"))
                fig.tight_layout()
                fig.colorbar(scatPlot)
                pdf.savefig(fig, dpi=100)


class PairStampAnalysisConfig(PairBaseConfig):
    stampSize = pexConfig.Field(
        dtype = int, default = 128,
        doc = "Pixels across PSF image"
    )
    pixelScale = pexConfig.Field(
        dtype = float, default=0.005,
        doc = "Pixel scale for PSF image"
    )


class PairStampAnalysisTask(PairBaseTask):
    ConfigClass = PairStampAnalysisConfig
    _DefaultName = "PairStampAnalysis"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, extraRef, intraRef=None):
        """Process a pair of exposures.
        """
        extraVisit = extraRef.dataId['visit']
        camera = extraRef.get("camera")
        self.log.info("Working on extra/intra visits: {}/{}".format(
            extraRef.dataId['visit'], intraRef.dataId['visit']))

        extraIdList, intraIdList = self.getIdLists(extraRef, intraRef)

        donutConfig = getDonutConfig(extraRef)

        # Collect 1 image per ccd
        psfs = {}
        wavefronts = {}
        for ccd, eCcdRef in extraIdList.items():
            try:
                iCcdRef = intraIdList[ccd]
            except KeyError:
                continue
            self.log.info("Working on ccd {}".format(ccd))
            catalogs = self.getCatalogs(eCcdRef, iCcdRef)
            pairs = self.getPairs(catalogs)
            s2n = []
            for pair in pairs:
                extraId, intraId, donutCoord = pair
                extraRecord = catalogs['extraIcSrc'].find(extraId)
                intraRecord = catalogs['intraIcSrc'].find(intraId)
                s2n.append(
                    0.5*(
                        (extraRecord['base_CircularApertureFlux_25_0_flux'] /
                         extraRecord['base_CircularApertureFlux_25_0_fluxSigma']) +
                        (intraRecord['base_CircularApertureFlux_25_0_flux'] /
                         intraRecord['base_CircularApertureFlux_25_0_fluxSigma'])
                    )
                )
            if len(s2n) == 0:
                continue
            idx = int(np.argsort(s2n)[-1])
            records = self.getPairRecords(catalogs, pairs[idx])
            extraIcExp = eCcdRef.get("icExp")
            psf, wf = donutPsfWf(records, extraIcExp, donutConfig, self.config,
                                 extraRef.get("camera"))
            psfs[ccd] = psf
            wavefronts[ccd] = wf

        extraButler = extraRef.getButler()
        outputdir = os.path.dirname(
            os.path.join(
                extraButler.get(
                    "donutSrc_filename",
                    visit=extraVisit,
                    ccd=0
                )[0]
            )
        )
        plotdir = os.path.abspath(os.path.join(outputdir, "..", "plots"))
        safeMakeDir(plotdir)

        # Make plots
        psfFn = "donutPairStampPsf-{:07d}.pdf".format(extraVisit)
        wavefrontFn = "donutPairStampWavefront-{:07d}.pdf".format(extraVisit)
        for data, fn, cmap in zip((psfs, wavefronts),
                                  (psfFn, wavefrontFn),
                                  ('viridis', 'seismic')):
            outfn = os.path.join(plotdir, fn)
            with PdfPages(outfn) as pdf:
                fig, axes = subplots(1, 1, figsize=(8, 6.2))
                axes = axes.ravel()[0]
                plotCameraOutline(axes, camera)
                for ccd, eCcdRef in extraIdList.items():
                    try:
                        self.imshow(data[ccd], camera[ccd], axes, cmap=cmap)
                    except KeyError:
                        pass
                fig.tight_layout()
                pdf.savefig(fig, dpi=200)

    @staticmethod
    def imshow(img, det, axes, **kwargs):
        corners = det.getCorners(afwCameraGeom.FOCAL_PLANE)
        left = min(c[0] for c in corners)
        right = max(c[0] for c in corners)
        top = max(c[1] for c in corners)
        bottom = min(c[1] for c in corners)
        # Shrink long axis so that donut has equal aspect ratio
        exttb = top-bottom
        extlr = right-left
        if exttb > extlr:
            medtb = 0.5*(top+bottom)
            top = medtb + 0.5*extlr
            bottom = medtb - 0.5*extlr
        else:
            medlr = 0.5*(left+right)
            left = medlr - 0.5*exttb
            right = medlr + 0.5*exttb
        axes.imshow(img, extent=[left, right, bottom, top],
                    aspect='equal', origin='lower', **kwargs)

    @staticmethod
    def getPairRecords(catalogs, pair):
        extraId, intraId, _ = pair
        return dict(
            extraIcRecord = catalogs['extraIcSrc'].find(extraId),
            extraDonutRecord = catalogs['extraDonutSrc'].find(extraId),
            intraIcRecord = catalogs['intraIcSrc'].find(intraId),
            intraDonutRecord = catalogs['intraDonutSrc'].find(intraId)
        )
