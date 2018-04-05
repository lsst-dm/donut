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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplots

import lsst.pex.config as pexConfig
import lsst.afw.cameraGeom as afwCameraGeom
# from lsst.afw.geom import arcseconds, Point2D

from .runner import PairBaseConfig, PairBaseTask
from .utilities import getDonutConfig
from .utilities import _noll_to_zern
from .utilities import getPsf, getWavefront
from .plotUtils import getPlotDir, plotCameraOutline


class ZernikeParamAnalysisConfig(PairBaseConfig):
    pass


class ZernikeParamAnalysisTask(PairBaseTask):
    ConfigClass = ZernikeParamAnalysisConfig
    _DefaultName = "ZernikeParamAnalysis"

    def __init__(self, *args, **kwargs):
        PairBaseTask.__init__(self, *args, **kwargs)

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

        plotDir = getPlotDir(extraRef.getButler(), "donutSrc", extraIdList[0].dataId)
        outfn = os.path.join(
            plotDir, "donutZernikeParam-{:07d}.pdf".format(extraVisit))

        with PdfPages(outfn) as pdf:
            for k, v in iteritems(vals):
                self.log.info("Plotting {}".format(k))
                fig, axes = subplots(1, 1, figsize=(8, 6.2))
                vmin = -max(np.abs(v))
                vmax = max(np.abs(v))
                scatPlot = axes.scatter(x, y, c=v, s=15, linewidths=0.5,
                                        vmin=vmin, vmax=vmax, cmap='Spectral_r')
                axes.set_title(k)
                plotCameraOutline(axes, extraRef.get("camera"))
                fig.tight_layout()
                fig.colorbar(scatPlot)
                pdf.savefig(fig, dpi=100)


class PairStampCcdAnalysisConfig(PairBaseConfig):
    stampSize = pexConfig.Field(
        dtype = int, default = 128,
        doc = "Pixels across PSF image"
    )
    pixelScale = pexConfig.Field(
        dtype = float, default=0.005,
        doc = "Pixel scale for PSF image"
    )


class PairStampCcdAnalysisTask(PairBaseTask):
    ConfigClass = PairStampCcdAnalysisConfig
    _DefaultName = "PairStampCcdAnalysis"

    def __init__(self, *args, **kwargs):
        PairBaseTask.__init__(self, *args, **kwargs)

    def run(self, extraRef, intraRef=None):
        """Process a pair of exposures.  Producing output binned by CCD.
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
            camera = extraRef.get("camera")
            psf = getPsf(records, extraIcExp, donutConfig, self.config,
                         camera)
            wf = getWavefront(records, extraIcExp, donutConfig, self.config,
                              camera)
            psfs[ccd] = psf
            wavefronts[ccd] = wf

        plotDir = getPlotDir(extraRef.getButler(), "donutSrc", extraIdList[0].dataId)

        # Make plots
        psfFn = "donutPairStampCcdPsf-{:07d}.pdf".format(extraVisit)
        wavefrontFn = "donutPairStampCcdWavefront-{:07d}.pdf".format(extraVisit)
        for data, fn, cmap in zip((psfs, wavefronts),
                                  (psfFn, wavefrontFn),
                                  ('viridis', 'Spectral_r')):
            outfn = os.path.join(plotDir, fn)
            with PdfPages(outfn) as pdf:
                fig, axes = subplots(1, 1, figsize=(8, 6.2))
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


class PairStampAnalysisConfig(PairBaseConfig):
    stampSize = pexConfig.Field(
        dtype = int, default = 128,
        doc = "Pixels across PSF image"
    )
    pixelScale = pexConfig.Field(
        dtype = float, default=0.005,
        doc = "Pixel scale for PSF image"
    )
    nStamp = pexConfig.Field(
        dtype = int, default=10,
        doc = "Number of stamps across FoV"
    )


class PairStampAnalysisTask(PairBaseTask):
    ConfigClass = PairStampAnalysisConfig
    _DefaultName = "PairStampAnalysis"

    def __init__(self, *args, **kwargs):
        PairBaseTask.__init__(self, *args, **kwargs)

    def run(self, extraRef, intraRef=None):
        """Process a pair of exposures.
        """
        extraVisit = extraRef.dataId['visit']
        camera = extraRef.get("camera")
        self.log.info("Working on extra/intra visits: {}/{}".format(
            extraRef.dataId['visit'], intraRef.dataId['visit']))

        extraIdList, intraIdList = self.getIdLists(extraRef, intraRef)

        donutConfig = getDonutConfig(extraRef)

        # Pass through data and read in donut src catalogs
        s2ns = []
        allPairs = []
        ccds = []
        for ccd, eCcdRef in extraIdList.items():
            try:
                iCcdRef = intraIdList[ccd]
            except KeyError:
                continue
            self.log.info("Working on ccd {}".format(ccd))
            catalogs = self.getCatalogs(eCcdRef, iCcdRef)
            pairs = self.getPairs(catalogs)
            allPairs.extend(pairs)
            for pair in pairs:
                extraId, intraId, donutCoord = pair
                extraRecord = catalogs['extraIcSrc'].find(extraId)
                intraRecord = catalogs['intraIcSrc'].find(intraId)
                s2ns.append(
                    0.5*(
                        (extraRecord['base_CircularApertureFlux_25_0_flux'] /
                         extraRecord['base_CircularApertureFlux_25_0_fluxSigma']) +
                        (intraRecord['base_CircularApertureFlux_25_0_flux'] /
                         intraRecord['base_CircularApertureFlux_25_0_fluxSigma'])
                    )
                )
                ccds.append(ccd)
        s2ns = np.array(s2ns)
        ccds = np.array(ccds)

        # Determine grid cells
        bds = camera.getFpBBox()
        xbds = np.linspace(bds.getMinX(), bds.getMaxX(), self.config.nStamp+1)
        ybds = np.linspace(bds.getMinY(), bds.getMaxY(), self.config.nStamp+1)

        # Find brightest donut in each grid cell and load it.
        psfDict = {}
        wfDict = {}
        xs = np.array([p[2][0] for p in allPairs])
        ys = np.array([p[2][1] for p in allPairs])
        for ix, (xmin, xmax) in enumerate(zip(xbds[:-1], xbds[1:])):
            for iy, (ymin, ymax) in enumerate(zip(ybds[:-1], ybds[1:])):
                w = (xs > xmin) & (xs <= xmax) & (ys > ymin) & (ys <= ymax)
                if not np.any(w):
                    continue
                widx = int(np.argsort(s2ns[w])[-1])
                idx = w.nonzero()[0][widx]
                ccd = ccds[idx]
                pair = allPairs[idx]
                eCcdRef = extraIdList[ccd]
                iCcdRef = intraIdList[ccd]
                catalogs = self.getCatalogs(eCcdRef, iCcdRef)
                records = self.getPairRecords(catalogs, pair)
                extraIcExp = eCcdRef.get("icExp")
                psf = getPsf(records, extraIcExp, donutConfig, self.config,
                             camera)
                wf = getWavefront(records, extraIcExp, donutConfig, self.config,
                                  camera)
                psfDict[(ix, iy)] = psf
                wfDict[(ix, iy)] = wf

        plotDir = getPlotDir(extraRef.getButler(), "donutSrc", extraIdList[0].dataId)

        self.makePlot(
            psfDict,
            os.path.join(
                plotDir,
                "donutPairStampPsf-{:07d}.pdf".format(extraVisit)
            ),
            xbds,
            ybds,
            camera,
            cmap = 'viridis'
        )

        # For wavefronts, it makes sense to use a consistent colorbar
        vmin, vmax = np.percentile(np.array(list(wfDict.values())),
                                   [0.5, 99.5])
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs

        self.makePlot(
            wfDict,
            os.path.join(
                plotDir,
                "donutPairStampWavefront-{:07d}.pdf".format(extraVisit)
            ),
            xbds,
            ybds,
            camera,
            cmap = 'Spectral_r',
            vmin = vmin,
            vmax = vmax
        )

    @staticmethod
    def makePlot(data, fn, xbds, ybds, camera, **kwargs):
        with PdfPages(fn) as pdf:
            fig, axes = subplots(1, 1, figsize=(8, 6.2))
            plotCameraOutline(axes, camera, doCcd=False)
            for ix, (xmin, xmax) in enumerate(zip(xbds[:-1], xbds[1:])):
                for iy, (ymin, ymax) in enumerate(zip(ybds[:-1], ybds[1:])):
                    if (ix, iy) not in data:
                        continue
                    axes.imshow(
                        data[(ix, iy)],
                        extent=[xmin, xmax, ymin, ymax],
                        aspect='equal',
                        origin='lower',
                        **kwargs
                    )
            fig.tight_layout()
            pdf.savefig(fig, dpi=200)

    @staticmethod
    def getPairRecords(catalogs, pair):
        extraId, intraId, _ = pair
        return dict(
            extraIcRecord = catalogs['extraIcSrc'].find(extraId),
            extraDonutRecord = catalogs['extraDonutSrc'].find(extraId),
            intraIcRecord = catalogs['intraIcSrc'].find(intraId),
            intraDonutRecord = catalogs['intraDonutSrc'].find(intraId)
        )


class PairZernikePyramidConfig(PairBaseConfig):
    pass


class PairZernikePyramidTask(PairBaseTask):
    ConfigClass = PairZernikePyramidConfig
    _DefaultName = "PairZernikePyramid"

    def __init__(self, *args, **kwargs):
        PairBaseTask.__init__(self, *args, **kwargs)

    def run(self, extraRef, intraRef=None):
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

        plotDir = getPlotDir(extraRef.getButler(), "donutSrc", extraIdList[0].dataId)

        outfn = os.path.join(
            plotDir,
            "donutZernikePyramid-{:07d}.pdf".format(extraVisit))

        nrow = _noll_to_zern(jmax)[0] - 1
        ncol = nrow + 2
        gridspec = GridSpec(nrow, ncol)


        def shift(pos, amt):
            return [pos.x0+amt, pos.y0, pos.width, pos.height]

        def shiftAxes(axes, amt):
            for ax in axes:
                ax.set_position(shift(ax.get_position(), amt))

        with PdfPages(outfn) as pdf:
            fig = Figure(figsize=(13, 8))
            axes = {}
            shiftLeft = []
            shiftRight = []
            for j in range(4, jmax+1):
                n, m = _noll_to_zern(j)
                if n%2 == 0:
                    row, col = n-2, m//2+ncol//2
                else:
                    row, col = n-2, (m-1)//2+ncol//2
                subplotspec = gridspec.new_subplotspec((row, col))
                axes[j] = fig.add_subplot(subplotspec)
                axes[j].set_aspect('equal')
                if nrow%2==0 and n%2==0:
                    shiftLeft.append(axes[j])
                if nrow%2==1 and n%2==1:
                    shiftRight.append(axes[j])

            for j, ax in axes.items():
                k = "z{}".format(j)
                ax.set_title(k)
                ax.scatter(x, y, c=vals[k], s=1, linewidths=0.5, cmap='Spectral_r',
                           rasterized=True, vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout()
            amt = 0.5*(axes[4].get_position().x0 - axes[5].get_position().x0)
            shiftAxes(shiftLeft, -amt)
            shiftAxes(shiftRight, amt)

            pdf.savefig(fig, dpi=100)
