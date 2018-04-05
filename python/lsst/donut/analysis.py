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
from matplotlib.pyplot import subplots

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.cameraGeom as afwCameraGeom
from lsst.daf.persistence import NoResults
from lsst.pipe.drivers.utils import getDataRef, ButlerTaskRunner
from .utilities import first
from .utilities import getDonutConfig
from .utilities import getDonut, getModel

from .plotUtils import getPlotDir, plotCameraOutline


class SelectionAnalysisConfig(pexConfig.Config):
    pass


class SelectionAnalysisTask(pipeBase.CmdLineTask):
    ConfigClass = SelectionAnalysisConfig
    _DefaultName = "selectionAnalysis"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, sensorRef):
        dataId = sensorRef.dataId
        visit = dataId['visit']
        ccd = dataId['ccd']
        self.log.info("Running on visit {}, ccd {}".format(visit, ccd))

        donutConfig = getDonutConfig(sensorRef)
        try:
            donutSrc = sensorRef.get("donutSrc")
            icSrc = sensorRef.get("icSrc")
            assert len(icSrc) > 0
        except NoResults:
            self.log.debug("Could not load donutSrc or icSrc")
            return
        self.log.info("Found {} donuts to plot".format(len(icSrc)))
        icExp = sensorRef.get("icExp")

        s2n = (icSrc['base_CircularApertureFlux_25_0_flux'] /
               icSrc['base_CircularApertureFlux_25_0_fluxSigma'])

        plotDir = getPlotDir(sensorRef.getButler(), "icSrc", dataId)
        outfn = os.path.join(
            plotDir, "donutSelection-{:07d}-{:03d}.pdf".format(visit, ccd))

        pixelScale = icExp.getWcs().pixelScale()
        extent = [0.5*donutConfig.stampSize*pixelScale.asArcseconds()*e
                  for e in [-1, 1, -1, 1]]

        with PdfPages(outfn) as pdf:
            i = 0
            for si in reversed(np.argsort(s2n)):
                if i % 12 == 0:
                    fig, axes = subplots(4, 3, figsize=(8.5, 11))
                src = icSrc[int(si)]
                cmap = 'Blues' if src['id'] in donutSrc['id'] else 'Reds'
                axes.ravel()[i%12].imshow(
                    # (getCutout(x[si], y[si], icExp, donutConfig.stampSize)
                    #  .getImage()
                    #  .getArray()),
                    getDonut(src, icExp, donutConfig),
                    cmap=cmap,
                    interpolation='nearest',
                    extent=extent)
                axes.ravel()[i%12].set_title(
                    "S/N={:4d}".format(int(s2n[si])),
                    fontsize=11)
                if i % 12 == 11:
                    fig.suptitle("visit = {}  ccd = {}".format(visit, ccd))
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.94)
                    pdf.savefig(fig, dpi=100)
                i += 1
            # Clean up potentially partially-filled page
            if i % 12 != 11:
                fig.suptitle("visit = {}  ccd = {}".format(visit, ccd))
                fig.tight_layout()
                fig.subplots_adjust(top=0.94)
                pdf.savefig(fig, dpi=100)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None


class GoodnessOfFitAnalysisConfig(pexConfig.Config):
    psfStampSize = pexConfig.Field(
        dtype = int, default=32,
        doc = "Size of PSF stamp in pixels",
    )
    psfPixelScale = pexConfig.Field(
        dtype = float, default=0.025,
        doc = "Pixel scale of PSF stamp in arcsec"
    )


class GoodnessOfFitAnalysisTask(pipeBase.CmdLineTask):
    ConfigClass = GoodnessOfFitAnalysisConfig
    _DefaultName = "goodnessOfFitAnalysis"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, sensorRef):
        dataId = sensorRef.dataId
        visit = dataId['visit']
        ccd = dataId['ccd']
        self.log.info("Running on visit {}, ccd {}".format(visit, ccd))

        donutConfig = getDonutConfig(sensorRef)

        try:
            icSrc = sensorRef.get('icSrc')
            donutSrc = sensorRef.get('donutSrc')
        except NoResults:
            self.log.debug("Could not find one of icSrc, donutSrc")
            return
        self.log.info("Found {} donuts to plot".format(len(donutSrc)))
        assert len(donutSrc) > 0
        icExp = sensorRef.get('icExp')
        camera = sensorRef.get('camera')

        plotDir = getPlotDir(sensorRef.getButler(), "donutSrc", dataId)
        outfn = os.path.join(
            plotDir, "donutGoodnessOfFit-{:07d}-{:03d}.pdf".format(visit, ccd))
        pixelScale = icExp.getWcs().pixelScale()
        donutExtent = [0.5*donutConfig.stampSize*pixelScale.asArcseconds()*e
                       for e in [-1, 1, -1, 1]]
        kwargs = {'cmap':'viridis', 'interpolation':'nearest', 'extent':donutExtent}
        residKwargs = dict(kwargs)
        residKwargs['cmap'] = 'Spectral_r'
        nrow = 5
        ncol = 3
        with PdfPages(outfn) as pdf:
            i = 0
            for donutRecord in donutSrc:
                icRecord = icSrc.find(donutRecord.getId())
                if i % nrow == 0:
                    fig, axes = subplots(nrow, ncol, figsize=(8.5, 11))
                data = getDonut(icRecord, icExp, donutConfig)
                model = getModel(donutRecord, icRecord, icExp, donutConfig, camera)
                resid = data - model

                axes[i%nrow, 0].imshow(data, **kwargs)
                axes[i%nrow, 1].imshow(model, **kwargs)
                axes[i%nrow, 2].imshow(resid, **residKwargs)

                if i % nrow == nrow-1:
                    fig.tight_layout()
                    pdf.savefig(fig, dpi=100)
                i += 1
            if i % nrow != nrow-1:
                fig.tight_layout()
                pdf.savefig(fig, dpi=100)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None


class FitParamAnalysisConfig(pexConfig.Config):
    pass


class FitParamAnalysisTask(pipeBase.CmdLineTask):
    ConfigClass = FitParamAnalysisConfig
    _DefaultName = "FitParamAnalysis"
    RunnerClass = ButlerTaskRunner

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, expRef, butler):
        """Process a single exposure, with scatter-gather-scatter using MPI.
        """
        dataIdList = dict([(ccdRef.get("ccdExposureId"), ccdRef.dataId)
                           for ccdRef in expRef.subItems("ccd")
                           if ccdRef.datasetExists("donutSrc")])
        dataIdList = collections.OrderedDict(sorted(dataIdList.items()))
        visit = expRef.dataId['visit']
        self.log.info("Running on visit {}".format(visit))

        donutConfig = getDonutConfig(expRef)
        x = []
        y = []
        vals = collections.OrderedDict()
        jmax = donutConfig.jmaxs[-1]
        for k in ['r0']+['z{}'.format(z) for z in range(4, jmax + 1)]:
            vals[k] = []

        for dataId in dataIdList.values():
            self.log.info("Loading ccd {}".format(dataId['ccd']))
            sensorRef = getDataRef(butler, dataId)
            donutSrc = sensorRef.get("donutSrc")
            icSrc = sensorRef.get("icSrc")
            x.extend([icSrc.find(donut.getId())['base_FPPosition_x'] for donut in donutSrc])
            y.extend([icSrc.find(donut.getId())['base_FPPosition_y'] for donut in donutSrc])
            for k, v in iteritems(vals):
                v.extend(donut['zfit_jmax{}_{}'.format(jmax, k)] for donut in donutSrc)

        plotDir = getPlotDir(sensorRef.getButler(), "donutSrc", first(dataIdList.values()))
        outfn = os.path.join(
            plotDir,
            "donutFitParam-{:07d}.pdf".format(visit))
        with PdfPages(outfn) as pdf:
            for k, v in iteritems(vals):
                self.log.info("Plotting {}".format(k))
                if k.startswith('z') and k != 'z4':
                    cmap = 'Spectral_r'
                    vmin = -max(np.abs(v))
                    vmax = max(np.abs(v))
                else:
                    cmap = 'viridis'
                    vmin = min(v)
                    vmax = max(v)
                fig, axes = subplots(1, 1, figsize=(8, 6.2))
                scatPlot = axes.scatter(x, y, c=v, s=15, linewidths=0.5,
                                        cmap=cmap, vmin=vmin, vmax=vmax)
                axes.set_title(k)
                plotCameraOutline(axes, expRef.get("camera"))
                fig.tight_layout()
                fig.colorbar(scatPlot)
                pdf.savefig(fig, dpi=100)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        # Pop doBatch keyword before passing it along to the argument parser
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name="FitParamAnalysis",
                                         *args, **kwargs)
        parser.add_id_argument("--id", datasetType="donutSrc", level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None


class StampCcdAnalysisConfig(pexConfig.Config):
    psfStampSize = pexConfig.Field(
        dtype = int, default=32,
        doc = "Size of PSF stamp in pixels",
    )
    psfPixelScale = pexConfig.Field(
        dtype = float, default=0.025,
        doc = "Pixel scale of PSF stamp in arcsec"
    )


class StampCcdAnalysisTask(pipeBase.CmdLineTask):
    ConfigClass = StampCcdAnalysisConfig
    _DefaultName = "StampCcdAnalysisTask"
    RunnerClass = ButlerTaskRunner

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, expRef, butler):
        """Make a stamp analysis image for a single exposure, binned by CCD.
        """
        dataIdList = dict([(ccdRef.get("ccdExposureId"), ccdRef.dataId)
                           for ccdRef in expRef.subItems("ccd")
                           if ccdRef.datasetExists("donutSrc")])
        dataIdList = collections.OrderedDict(sorted(dataIdList.items()))
        visit = expRef.dataId['visit']
        self.log.info("Running on visit {}".format(visit))
        camera = expRef.get("camera")

        donutConfig = getDonutConfig(expRef)

        plotDir = getPlotDir(expRef.getButler(),
                            "donutSrc",
                            first(dataIdList.values()))

        # Collect images
        images = {}
        models = {}
        resids = {}
        for dataId in dataIdList.values():
            ccd = dataId['ccd']
            self.log.info("Loading ccd {}".format(ccd))
            sensorRef = getDataRef(butler, dataId)
            icExp = sensorRef.get("icExp")
            donutSrc = sensorRef.get("donutSrc")
            icSrc = sensorRef.get("icSrc")

            if len(donutSrc) == 0:
                continue
            s2n = []
            for donut in donutSrc:
                icRec = icSrc.find(donut.getId())
                s2n.append(icRec['base_CircularApertureFlux_25_0_flux'] /
                           icRec['base_CircularApertureFlux_25_0_fluxSigma'])

            idx = int(np.argsort(s2n)[-1])
            donutRecord = donutSrc[idx]
            icRecord = icSrc.find(donutRecord.getId())
            data = getDonut(icRecord, icExp, donutConfig)
            model = getModel(donutRecord, icRecord, icExp, donutConfig, camera)
            resid = data - model
            images[ccd] = data
            models[ccd] = model
            resids[ccd] = resid

        # Make plots
        self.makePlot(
            images,
            os.path.join(
                plotDir,
                "donutStampCcdData-{:07d}.pdf".format(visit),
            ),
            camera,
            dataIdList,
            cmap = 'viridis'
        )

        self.makePlot(
            models,
            os.path.join(
                plotDir,
                "donutStampCcdModel-{:07d}.pdf".format(visit),
            ),
            camera,
            dataIdList,
            cmap = 'viridis'
        )

        self.makePlot(
            resids,
            os.path.join(
                plotDir,
                "donutStampCcdResid-{:07d}.pdf".format(visit),
            ),
            camera,
            dataIdList,
            cmap = 'Spectral_r'
        )

    @classmethod
    def makePlot(cls, data, fn, camera, dataIdList, **kwargs):
        with PdfPages(fn) as pdf:
            fig, axes = subplots(1, 1, figsize=(8, 6.2))
            plotCameraOutline(axes, camera)
            for dataId in dataIdList.values():
                ccd = dataId['ccd']
                try:
                    cls.imshow(data[ccd], camera[ccd], axes, **kwargs)
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

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        # Pop doBatch keyword before passing it along to the argument parser
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name="StampCcdAnalysis",
                                         *args, **kwargs)
        parser.add_id_argument("--id", datasetType="donutSrc", level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None


class StampAnalysisConfig(pexConfig.Config):
    psfStampSize = pexConfig.Field(
        dtype = int, default=32,
        doc = "Size of PSF stamp in pixels"
    )
    psfPixelScale = pexConfig.Field(
        dtype = float, default=0.025,
        doc = "Pixel scale of PSF stamp in arcsec"
    )
    nStamp = pexConfig.Field(
        dtype = int, default=10,
        doc = "Number of stamps across FoV"
    )


class StampAnalysisTask(pipeBase.CmdLineTask):
    ConfigClass = StampAnalysisConfig
    _DefaultName = "StampAnalysisTask"
    RunnerClass = ButlerTaskRunner

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, expRef, butler):
        """Make a stamp analysis image for a single exposure
        """
        dataIdList = dict([(ccdRef.get("ccdExposureId"), ccdRef.dataId)
                           for ccdRef in expRef.subItems("ccd")
                           if ccdRef.datasetExists("donutSrc")])
        dataIdList = collections.OrderedDict(sorted(dataIdList.items()))
        visit = expRef.dataId['visit']
        self.log.info("Running on visit {}".format(visit))
        camera = expRef.get("camera")

        donutConfig = getDonutConfig(expRef)

        plotDir = getPlotDir(expRef.getButler(),
                            "donutSrc",
                            first(dataIdList.values()))

        xs = []
        ys = []
        s2ns = []
        sensorRefs = []
        ids = []
        for dataId in dataIdList.values():
            ccd = dataId['ccd']
            self.log.info("Loading ccd {}".format(ccd))
            sensorRef = getDataRef(butler, dataId)
            donutSrc = sensorRef.get("donutSrc")
            icSrc = sensorRef.get("icSrc")
            if len(donutSrc) == 0:
                continue
            for donut in donutSrc:
                donutId = donut.getId()
                icRec = icSrc.find(donutId)
                s2ns.append(icRec['base_CircularApertureFlux_25_0_flux'] /
                            icRec['base_CircularApertureFlux_25_0_fluxSigma'])
                xs.append(icRec['base_FPPosition_x'])
                ys.append(icRec['base_FPPosition_y'])
                sensorRefs.append(sensorRef)
                ids.append(donutId)

        bds = camera.getFpBBox()
        xbds = np.linspace(bds.getMinX(), bds.getMaxX(), self.config.nStamp+1)
        ybds = np.linspace(bds.getMinY(), bds.getMaxY(), self.config.nStamp+1)

        # Find brightest donut in each grid cell and retrieve that one.
        s2ns = np.array(s2ns)
        xs = np.array(xs)
        ys = np.array(ys)
        imageDict = {}
        modelDict = {}
        residDict = {}
        for ix, (xmin, xmax) in enumerate(zip(xbds[:-1], xbds[1:])):
            for iy, (ymin, ymax) in enumerate(zip(ybds[:-1], ybds[1:])):
                w = (xs > xmin) & (xs <= xmax) & (ys > ymin) & (ys <= ymax)
                if not np.any(w):
                    continue
                widx = int(np.argsort(s2ns[w])[-1])
                idx = w.nonzero()[0][widx]
                sensorRef = sensorRefs[idx]
                donutId = ids[idx]
                icExp = sensorRef.get("icExp")
                icSrc = sensorRef.get("icSrc")
                donutSrc = sensorRef.get("donutSrc")
                icRec = icSrc.find(donutId)
                donut = donutSrc.find(donutId)

                data = getDonut(icRec, icExp, donutConfig)
                model = getModel(donut, icRec, icExp, donutConfig, camera)
                imageDict[(ix, iy)] = data
                modelDict[(ix, iy)] = model
                residDict[(ix, iy)] = data - model

        # Make plots
        self.makePlot(
            imageDict,
            os.path.join(
                plotDir,
                "donutStampData-{:07d}.pdf".format(visit)
            ),
            xbds,
            ybds,
            camera,
            cmap = 'viridis'
        )

        self.makePlot(
            modelDict,
            os.path.join(
                plotDir,
                "donutStampModel-{:07d}.pdf".format(visit)
            ),
            xbds,
            ybds,
            camera,
            cmap = 'viridis'
        )

        self.makePlot(
            residDict,
            os.path.join(
                plotDir,
                "donutStampResid-{:07d}.pdf".format(visit)
            ),
            xbds,
            ybds,
            camera,
            cmap = 'Spectral_r'
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

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        # Pop doBatch keyword before passing it along to the argument parser
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name="StampAnalysis",
                                         *args, **kwargs)
        parser.add_id_argument("--id", datasetType="donutSrc", level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
