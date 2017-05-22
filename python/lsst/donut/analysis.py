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

import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.daf.persistence import NoResults


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


def filedir(butler, dataset, dataId):
    return os.path.dirname(butler.get(dataset+"_filename", dataId)[0])


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
                    interpolation='nearest')
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
