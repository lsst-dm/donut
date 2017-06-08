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
import galsim
from matplotlib.backends.backend_pdf import PdfPages, FigureCanvasPdf
from matplotlib.figure import Figure
import matplotlib.patches as patches

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.geom as afwGeom
from lsst.afw.geom import arcseconds
import lsst.afw.math as afwMath
import lsst.afw.cameraGeom as afwCameraGeom
from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.daf.persistence import NoResults
from lsst.pipe.drivers.utils import getDataRef, ButlerTaskRunner
from .zernikeFitter import ZernikeFitter
from .utilities import cutoutDonut, markGoodDonuts, _getGoodPupilShape
from .utilities import _getJacobian


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


def donutDataModelWfPsf(donut, donutConfig, icExp, camera,
                        psfStampSize=None, psfPixelScale=None,
                        psfOnly=False):
    """Return numpy arrays of donut cutout, corresponding model, wavefront,
    and implied in-focus PSF.
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

    pupilSize, pupilNPix = _getGoodPupilShape(
        camera.telescopeDiameter, wavelength, donutConfig.stampSize*pixelScale)
    pupilFactory = camera.getPupilFactory(visitInfo, pupilSize, pupilNPix)

    fpX = donut['base_FPPosition_x']
    fpY = donut['base_FPPosition_y']
    pupil = pupilFactory.getPupil(afwGeom.Point2D(fpX, fpY))

    detector = icExp.getDetector()
    point = afwGeom.Point2D(donut.getX(), donut.getY())
    jacobian = _getJacobian(detector, point)
    # Need to apply quarter rotations to jacobian
    th = np.pi/2*nquarter
    sth, cth = np.sin(th), np.cos(th)
    rot = np.array([[cth, sth], [-sth, cth]])
    jacobian = np.dot(rot.T, np.dot(jacobian, rot))

    params = {}
    keys = ['r0', 'dx', 'dy', 'flux']
    for j in range(4, zmax + 1):
        keys.append('z{}'.format(j))
    for k in keys:
        params[k] = donut[k]
    zfitter = ZernikeFitter(
        zmax,
        wavelength,
        pupil,
        camera.telescopeDiameter)

    if not psfOnly:
        maskedImage = cutoutDonut(donut.getX(), donut.getY(), icExp,
                                  donutConfig.stampSize)
        maskedImage = afwMath.rotateImageBy90(maskedImage, nquarter)
        data = maskedImage.getImage().getArray()
        model = zfitter.constructModelImage(
            params = params,
            pixelScale = pixelScale.asArcseconds(),
            jacobian = jacobian,
            shape = (donutConfig.stampSize, donutConfig.stampSize))

    # Use less well-sampled pupil for in-focus PSF
    psfPupilSize, psfPupilNPix = _getGoodPupilShape(
        camera.telescopeDiameter, wavelength, 2*psfStampSize*psfPixelScale)
    psfPupilFactory = camera.getPupilFactory(
        visitInfo, psfPupilSize, psfPupilNPix)
    psfPupil = psfPupilFactory.getPupil(afwGeom.Point2D(fpX, fpY))
    zfitter.aper = galsim.Aperture(
        diam = camera.telescopeDiameter,
        pupil_plane_im = psfPupil.illuminated.astype(np.int16),
        pupil_plane_scale = psfPupil.scale,
        pupil_plane_size = psfPupil.size)

    params['z4'] = 0.0
    params['dx'] = 0.0
    params['dy'] = 0.0
    params['flux'] = 1.0
    del params['r0']
    psf = zfitter.constructModelImage(
        params = params,
        pixelScale = psfPixelScale.asArcseconds(),
        jacobian = jacobian,
        shape = (psfStampSize, psfStampSize))
    wf = zfitter.constructWavefrontImage(params=params)
    wf = wf[wf.shape[0]//4:3*wf.shape[0]//4, wf.shape[0]//4:3*wf.shape[0]//4]
    if psfOnly:
        return psf
    return data, model, wf, psf


def moments(image, scale=1.0):
    x, y = np.meshgrid(np.arange(image.shape[0])*scale,
                       np.arange(image.shape[1])*scale)
    I0 = image.sum()
    Ix = (image*x).sum()/I0
    Iy = (image*y).sum()/I0
    Ixx = (image*(x-Ix)*(x-Ix)).sum()/I0
    Ixy = (image*(x-Ix)*(y-Iy)).sum()/I0
    Iyy = (image*(y-Iy)*(y-Iy)).sum()/I0
    rsqr = Ixx + Iyy
    e1 = (Ixx - Iyy)/rsqr
    e2 = 2*Ixy/rsqr
    r = np.sqrt(rsqr)
    e = np.hypot(e1, e2)
    return dict(I0=I0, Ix=Ix, Iy=Iy, Ixx=Ixx, Ixy=Ixy, Iyy=Iyy,
                e1=e1, e2=e2, rsqr=rsqr, r=r, e=e)


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
        wfExtent = [0.5*camera.telescopeDiameter*e for e in [-1, 1, -1, 1]]
        nrow = 7
        ncol = 5
        with PdfPages(outfn) as pdf:
            i = 0
            for donut in donutSrc:
                if i % nrow == 0:
                    fig, axes = subplots(nrow, ncol, figsize=(8.5, 11))
                data, model, wf, psf = donutDataModelWfPsf(
                    donut, donutConfig, icExp, camera,
                    psfStampSize = self.config.psfStampSize,
                    psfPixelScale = self.config.psfPixelScale*arcseconds)
                resid = data - model
                axes[i%nrow, 0].imshow(data, cmap='viridis',
                                       interpolation='nearest',
                                       extent=donutExtent)
                axes[i%nrow, 1].imshow(model, cmap='viridis',
                                       interpolation='nearest',
                                       extent=donutExtent)
                axes[i%nrow, 2].imshow(resid, cmap='viridis',
                                       interpolation='nearest',
                                       extent=donutExtent)
                axes[i%nrow, 3].imshow(psf, cmap='viridis',
                                       interpolation='nearest',
                                       extent=psfExtent)
                axes[i%nrow, 4].imshow(wf, cmap='viridis',
                                       interpolation='nearest',
                                       extent=wfExtent)
                if i % nrow == nrow-1:
                    fig.tight_layout()
                    pdf.savefig(fig)
                i += 1
            if i % nrow != nrow-1:
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
            icExp = sensorRef.get("icExp")
            goodDonuts = markGoodDonuts(
                donutSrc, icExp,
                donutConfig.stampSize, donutConfig.ignoredPixelMask)
            donutSrc = donutSrc.subset(goodDonuts)
            x.extend([donut['base_FPPosition_x'] for donut in donutSrc])
            y.extend([donut['base_FPPosition_y'] for donut in donutSrc])
            for k, v in iteritems(vals):
                v.extend(donut[k] for donut in donutSrc)

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
        wfs = {}
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
            indices = np.arange(len(s2n))
            goodDonuts = markGoodDonuts(
                donutSrc, icExp,
                donutConfig.stampSize, donutConfig.ignoredPixelMask)
            if len(goodDonuts) < 1:
                continue

            indices = indices[goodDonuts]
            s2n = s2n[goodDonuts]
            idx = indices[np.argsort(s2n)[-1]]

            data, model, wf, psf = donutDataModelWfPsf(
                donutSrc[idx], donutConfig, icExp, camera,
                psfStampSize = self.config.psfStampSize,
                psfPixelScale = self.config.psfPixelScale*arcseconds)
            resid = data - model
            images[ccd] = data
            models[ccd] = model
            resids[ccd] = resid
            psfs[ccd] = psf
            wfs[ccd] = wf

        # Make plots
        datafn = "donutStampData-{:07d}.pdf".format(visit)
        modelfn = "donutStampModel-{:07d}.pdf".format(visit)
        residfn = "donutStampResid-{:07d}.pdf".format(visit)
        psffn = "donutStampPsf-{:07d}.pdf".format(visit)
        wffn = "donutStampWavefront-{:07d}.pdf".format(visit)
        for data, fn in zip((images, models, resids, psfs, wfs),
                            (datafn, modelfn, residfn, psffn, wffn)):
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


class PsfMomentsAnalysisConfig(pexConfig.Config):
    psfStampSize = pexConfig.Field(
        dtype = int,
        default = 32,
        doc = "Size of PSF stamp in pixels"
    )
    psfPixelScale = pexConfig.Field(
        dtype = float,
        default = 0.025,
        doc = "Pixel scale of PSF stamp in arcsec"
    )


class PsfMomentsAnalysisTask(pipeBase.CmdLineTask):
    ConfigClass = PsfMomentsAnalysisConfig
    _DefaultName = "PsfMomentsAnalysisTask"
    RunnerClass = ButlerTaskRunner

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    def run(self, expRef, butler):
        """Do moments analysis for single exposure
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

        x = []
        y = []
        vals = collections.OrderedDict()
        for k in ['Ixx', 'Ixy', 'Iyy', 'e1', 'e2', 'e', 'rsqr', 'r']:
            vals[k] = []

        for dataId in dataIdList.values():
            ccd = dataId['ccd']
            self.log.info("Processing ccd {}".format(ccd))
            sensorRef = getDataRef(butler, dataId)
            icExp = sensorRef.get("icExp")
            donutSrc = sensorRef.get("donutSrc")
            goodDonuts = markGoodDonuts(
                donutSrc, icExp,
                donutConfig.stampSize, donutConfig.ignoredPixelMask)

            for donut in donutSrc.subset(goodDonuts):
                psf = donutDataModelWfPsf(
                    donut, donutConfig, icExp, camera,
                    psfStampSize = self.config.psfStampSize,
                    psfPixelScale = self.config.psfPixelScale*arcseconds,
                    psfOnly=True)
                mom = moments(psf, self.config.psfPixelScale)
                x.append(donut['base_FPPosition_x'])
                y.append(donut['base_FPPosition_y'])
                for k in vals:
                    vals[k].append(mom[k])

        outputdir = filedir(expRef.getButler(),
                            "donutSrc",
                            dataIdList.values()[0])
        plotdir = os.path.abspath(os.path.join(outputdir, "..", "plots"))
        safeMakeDir(plotdir)
        outfn = os.path.join(
            plotdir,
            "donutPsfMoments-{:07d}.pdf".format(visit))
        with PdfPages(outfn) as pdf:
            for k, v in iteritems(vals):
                self.log.info("Plotting {}".format(k))
                fig, axes = subplots(1, 1, figsize=(8, 6.2))
                axes = axes.ravel()[0]
                scatPlot = axes.scatter(x, y, c=v, s=15, linewidths=0.5)
                axes.set_title(k)
                plotCameraOutline(axes, camera)
                fig.tight_layout()
                fig.colorbar(scatPlot)
                pdf.savefig(fig)

            # A bit of matplotlib trickery to get the whisker plot to align
            # with the scatter plots generated above.
            rect = axes.get_position()
            fig = Figure(figsize=(8, 6.2))
            FigureCanvasPdf(fig)
            axes = fig.add_axes(rect)

            wth = np.arctan2(vals['e2'], vals['e1'])/2
            wx = vals['e']*np.cos(wth)
            wy = vals['e']*np.sin(wth)
            axes.quiver(
                x, y, wx, wy,
                headwidth = 0,
                pivot = 'middle',
                headlength = 1e-10,  # If zero emits divide-by-zero warnings.
                headaxislength = 0,
                minlength = 0,
                scale_units = 'xy',
                width = 0.002)
            plotCameraOutline(axes, camera)
            pdf.savefig(fig)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        # Pop doBatch keyword before passing it along to the argument parser
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name="PsfMomentsAnalysis",
                                         *args, **kwargs)
        parser.add_id_argument("--id", datasetType="donutSrc", level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
