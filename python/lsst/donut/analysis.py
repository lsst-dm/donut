#!/usr/bin/env python
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
import matplotlib.patches as patches

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.geom as afwGeom
from lsst.afw.geom import arcseconds
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.cameraGeom as afwCameraGeom
from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.daf.persistence import NoResults
from lsst.pipe.drivers.utils import getDataRef, ButlerTaskRunner
from .zernikeFitter import ZernikeFitter
from .fitDonut import FitDonutTask


def cutoutDonut(x, y, icExp, stampSize):
    """!Cut out a postage stamp image of a single donut

    @param x  X-coordinate of postage stamp center
    @param y  Y-coordinate of postage stamp center
    @param icExp  Exposure from which to cutout stamp.
    @param stampSize  Size of cutout.
    @returns  MaskedImage with cutout.
    """
    point = afwGeom.Point2I(int(x), int(y))
    box = afwGeom.Box2I(point, point)
    box.grow(afwGeom.Extent2I(stampSize//2, stampSize//2))

    subMaskedImage = icExp.getMaskedImage().Factory(
        icExp.getMaskedImage(),
        box,
        afwImage.PARENT
    )
    return subMaskedImage


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


def filedir(butler, dataset, dataId):
    return os.path.dirname(butler.get(dataset+"_filename", dataId)[0])


def donutDataModelPsf(donut, donutConfig, icExp, camera,
                      psfStampSize=None, psfPixelScale=None):
    """Return numpy arrays of donut cutout and corresponding model
    """
    if psfStampSize is None:
        psfStampSize = donutConfig.stampSize
    pixelScale = icExp.getWcs().pixelScale()
    if psfPixelScale is None:
        psfPixelScale = pixelScale

    wavelength = donutConfig.wavelength
    if wavelength is None:
        wavelength = icExp.getFilter().getFilterProperty().getLambdaEff()
    zmax = donutConfig.zmax[-1]

    nquarter = icExp.getDetector().getOrientation().getNQuarter()
    if donutConfig.flip:
        nquarter += 2
    visitInfo = icExp.getInfo().getVisitInfo()

    pupilSize, pupilNPix = FitDonutTask._getGoodPupilShape(
        camera.telescopeDiameter, wavelength, donutConfig.stampSize*pixelScale)
    pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, pupilNPix)

    fpX = donut['base_FPPosition_x']
    fpY = donut['base_FPPosition_y']
    pupil = pupilFactory.getPupil(afwGeom.Point2D(fpX, fpY))
    maskedImage = cutoutDonut(donut.getX(), donut.getY(), icExp,
                              donutConfig.stampSize)
    maskedImage = afwMath.rotateImageBy90(maskedImage, nquarter)

    zfitter = ZernikeFitter(
        zmax,
        wavelength,
        pupil,
        camera.telescopeDiameter)
    params = {}
    keys = ['r0', 'dx', 'dy', 'flux']
    for j in range(4, zmax + 1):
        keys.append('z{}'.format(j))
    for k in keys:
        params[k] = donut[k]
    data = maskedImage.getImage().getArray()
    model = zfitter.constructModelImage(
        params = params,
        pixelScale = pixelScale.asArcseconds(),
        shape = (donutConfig.stampSize, donutConfig.stampSize))
    params['z4'] = 0.0
    params['dx'] = 0.0
    params['dy'] = 0.0
    params['flux'] = 1.0
    del params['r0']
    psf = zfitter.constructModelImage(
        params = params,
        pixelScale = psfPixelScale.asArcseconds(),
        shape = (psfStampSize, psfStampSize))
    return data, model, psf


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

        try:
            donutConfig = sensorRef.get("processDonut_config").fitDonut
        except NoResults:
            donutConfig = (sensorRef.get("donutDriver_config")
                           .processDonut.fitDonut)
        try:
            donutSrc = sensorRef.get("donutSrc")
            icSrc = sensorRef.get("icSrc")
            assert len(icSrc) > 0
        except:
            return
        icExp = sensorRef.get("icExp")

        x = icSrc.getX()
        y = icSrc.getY()
        s2n = (icSrc['base_CircularApertureFlux_25_0_flux'] /
               icSrc['base_CircularApertureFlux_25_0_fluxSigma'])

        outputdir = filedir(sensorRef.getButler(), "icSrc", dataId)
        plotdir = os.path.abspath(os.path.join(outputdir, "..", "plots"))
        safeMakeDir(plotdir)
        outfn = os.path.join(
            plotdir, "donutSelection-{:07d}-{:03d}.pdf".format(visit, ccd))

        pixelScale = icExp.getWcs().pixelScale()
        extent = [0.5*donutConfig.stampSize*pixelScale.asArcseconds()*e
                  for e in [-1, 1, -1, 1]]

        with PdfPages(outfn) as pdf:
            i = 0
            for si in reversed(np.argsort(s2n)):
                if i % 12 == 0:
                    fig, axes = subplots(4, 3, figsize=(8.5, 11))
                src = icSrc[si]
                cmap = 'viridis' if src['id'] in donutSrc['id'] else 'inferno'
                axes.ravel()[i%12].imshow(
                    (cutoutDonut(x[si], y[si], icExp, donutConfig.stampSize)
                     .getImage()
                     .getArray()),
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
                    pdf.savefig(fig)
                i += 1
            # Clean up potentially partially-filled page
            if i % 12 != 11:
                fig.suptitle("visit = {}  ccd = {}".format(visit, ccd))
                fig.tight_layout()
                fig.subplots_adjust(top=0.94)
                pdf.savefig(fig)

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

        try:
            donutConfig = sensorRef.get("processDonut_config").fitDonut
        except NoResults:
            donutConfig = (sensorRef.get("donutDriver_config")
                           .processDonut.fitDonut)
        try:
            donutSrc = sensorRef.get('donutSrc')
            assert len(donutSrc) > 0
        except:
            return
        icExp = sensorRef.get('icExp')
        camera = sensorRef.get('camera')

        outputdir = filedir(sensorRef.getButler(), "donutSrc", dataId)
        plotdir = os.path.abspath(os.path.join(outputdir, "..", "plots"))
        safeMakeDir(plotdir)
        outfn = os.path.join(
            plotdir, "donutGoodnessOfFit-{:07d}-{:03d}.pdf".format(visit, ccd))
        pixelScale = icExp.getWcs().pixelScale()
        donutExtent = [0.5*donutConfig.stampSize*pixelScale.asArcseconds()*e
                       for e in [-1, 1, -1, 1]]
        psfExtent = [0.5*self.config.psfStampSize*e*self.config.psfPixelScale
                     for e in [-1, 1, -1, 1]]
        with PdfPages(outfn) as pdf:
            i = 0
            for donut in donutSrc:
                if i % 5 == 0:
                    fig, axes = subplots(5, 4, figsize=(8.5, 11))
                data, model, psf = donutDataModelPsf(
                    donut, donutConfig, icExp, camera,
                    psfStampSize = self.config.psfStampSize,
                    psfPixelScale = self.config.psfPixelScale*arcseconds)
                resid = data - model
                axes[i%5, 0].imshow(data, cmap='viridis',
                                    interpolation='nearest',
                                    extent=donutExtent)
                axes[i%5, 1].imshow(model, cmap='viridis',
                                    interpolation='nearest',
                                    extent=donutExtent)
                axes[i%5, 2].imshow(resid, cmap='viridis',
                                    interpolation='nearest',
                                    extent=donutExtent)
                axes[i%5, 3].imshow(psf, cmap='viridis',
                                    interpolation='nearest',
                                    extent=psfExtent)
                if i % 5 == 4:
                    fig.tight_layout()
                    pdf.savefig(fig)
                i += 1
            if i % 5 != 4:
                fig.tight_layout()
                pdf.savefig(fig)

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

        try:
            donutConfig = expRef.get("processDonut_config").fitDonut
        except NoResults:
            donutConfig = (expRef.get("donutDriver_config")
                           .processDonut.fitDonut)
        x = []
        y = []
        vals = collections.OrderedDict()
        zmax = donutConfig.zmax[-1]
        for k in ['r0']+['z{}'.format(z) for z in range(4, zmax + 1)]:
            vals[k] = []

        for dataId in dataIdList.values():
            self.log.info("Loading ccd {}".format(dataId['ccd']))
            sensorRef = getDataRef(butler, dataId)
            donutSrc = sensorRef.get("donutSrc")
            x.extend(list(donutSrc['base_FPPosition_x']))
            y.extend(list(donutSrc['base_FPPosition_y']))
            for k, v in iteritems(vals):
                v.extend(donutSrc[k])

        outputdir = filedir(expRef.getButler(),
                            "donutSrc",
                            dataIdList.values()[0])
        plotdir = os.path.abspath(os.path.join(outputdir, "..", "plots"))
        safeMakeDir(plotdir)
        outfn = os.path.join(
            plotdir,
            "donutFitParam-{:07d}.pdf".format(visit))
        with PdfPages(outfn) as pdf:
            for k, v in iteritems(vals):
                self.log.info("Plotting {}".format(k))
                fig, axes = subplots(1, 1, figsize=(8, 6.2))
                axes = axes.ravel()[0]
                scatPlot = axes.scatter(x, y, c=v, s=15, linewidths=0.5)
                axes.set_title(k)
                plotCameraOutline(axes, expRef.get("camera"))
                fig.tight_layout()
                fig.colorbar(scatPlot)
                pdf.savefig(fig)

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


class StampAnalysisConfig(pexConfig.Config):
    psfStampSize = pexConfig.Field(
        dtype = int, default=32,
        doc = "Size of PSF stamp in pixels",
    )
    psfPixelScale = pexConfig.Field(
        dtype = float, default=0.025,
        doc = "Pixel scale of PSF stamp in arcsec"
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

        try:
            donutConfig = expRef.get("processDonut_config").fitDonut
        except NoResults:
            donutConfig = (expRef.get("donutDriver_config")
                           .processDonut.fitDonut)

        outputdir = filedir(expRef.getButler(),
                            "donutSrc",
                            dataIdList.values()[0])
        plotdir = os.path.abspath(os.path.join(outputdir, "..", "plots"))
        safeMakeDir(plotdir)

        # Collect images
        images = {}
        models = {}
        resids = {}
        psfs = {}
        for dataId in dataIdList.values():
            ccd = dataId['ccd']
            self.log.info("Loading ccd {}".format(ccd))
            sensorRef = getDataRef(butler, dataId)
            icExp = sensorRef.get("icExp")
            donutSrc = sensorRef.get("donutSrc")

            if len(donutSrc) == 0:
                continue
            s2n = (donutSrc['base_CircularApertureFlux_25_0_flux'] /
                   donutSrc['base_CircularApertureFlux_25_0_fluxSigma'])
            idx = np.argsort(s2n)[-1]

            data, model, psf = donutDataModelPsf(
                donutSrc[idx], donutConfig, icExp, camera,
                psfStampSize = self.config.psfStampSize,
                psfPixelScale = self.config.psfPixelScale*arcseconds)
            resid = data - model
            images[ccd] = data
            models[ccd] = model
            resids[ccd] = resid
            psfs[ccd] = psf

        # Make plots
        datafn = "donutStampData-{:07d}.pdf".format(visit)
        modelfn = "donutStampModel-{:07d}.pdf".format(visit)
        residfn = "donutStampResid-{:07d}.pdf".format(visit)
        psffn = "donutStampPsf-{:07d}.pdf".format(visit)
        for data, fn in zip((images, models, resids, psfs),
                            (datafn, modelfn, residfn, psffn)):
            outfn = os.path.join(plotdir, fn)
            with PdfPages(outfn) as pdf:
                fig, axes = subplots(1, 1, figsize=(8, 6.2))
                axes = axes.ravel()[0]
                plotCameraOutline(axes, camera)
                for dataId in dataIdList.values():
                    ccd = dataId['ccd']
                    try:
                        self.imshow(data[ccd], camera[ccd], axes,
                                    cmap='viridis')
                    except KeyError:
                        pass
                fig.tight_layout()
                pdf.savefig(fig)

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
        parser = pipeBase.ArgumentParser(name="StampAnalysis",
                                         *args, **kwargs)
        parser.add_id_argument("--id", datasetType="donutSrc", level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
