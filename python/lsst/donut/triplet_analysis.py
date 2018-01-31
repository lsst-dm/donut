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

import galsim
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import subplots

import lsst.pex.config as pexConfig

from .runner import TripletBaseConfig, TripletBaseTask
from .utilities import getDonutConfig, getCutout, lodToDol, getPsf, getMoments
from .plotUtils import getPlotDir


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
        dtype = float, default=0.2,
        doc = "Fried parameter for Kolmogorov profile "
              "to convolve with optical model"
    )


class TripletWhiskerTask(TripletBaseTask):
    ConfigClass = TripletWhiskerConfig
    _DefaultName = "TripletWhisker"

    def __init__(self, *args, **kwargs):
        TripletBaseTask.__init__(self, *args, **kwargs)

    def run(self, focalRef, extraRef=None, intraRef=None):
        """Process an exposure triplet consisting of an in-focus exposure, an
        extra-focal exposure, and an intra-focal exposure.
        """
        focalVisit = focalRef.dataId['visit']
        self.log.info("Working on triplet: {}/{}/{}".format(
            focalRef.dataId['visit'], extraRef.dataId['visit'], intraRef.dataId['visit']))
        focalIdList, extraIdList, intraIdList = self.getIdLists(focalRef, extraRef, intraRef)

        camera = focalRef.get("camera")
        donutConfig = getDonutConfig(extraRef)

        x = []
        y = []
        predictedMoms = []
        observedMoms = []

        for ccd, fCcdRef in focalIdList.items():
            try:
                iCcdRef = intraIdList[ccd]
                eCcdRef = extraIdList[ccd]
            except KeyError:
                continue
            # if ccd > 5: #  Comment me out when not testing
            #     break
            catalogs = self.getCatalogs(fCcdRef, eCcdRef, iCcdRef)
            triplets = self.getTriplets(catalogs)
            focalCalexp = fCcdRef.get("calexp")
            self.log.info("Found {} triplets on ccd {}".format(len(triplets), ccd))
            for triplet in triplets:
                records = self.getTripletRecords(catalogs, triplet)

                extraIcExp = eCcdRef.get('icExp')
                predictedPsf = getPsf(records, extraIcExp, donutConfig, self.config, camera)

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
        plotDir = getPlotDir(focalButler, "calexp", extraIdList[0].dataId)
        outfn = os.path.join(
            plotDir,
            "donutTripletWhisker-{:07d}.pdf".format(focalVisit))

        with PdfPages(outfn) as pdf:
            fig, axes = subplots(1, 2, figsize=(8, 4))
            qv0 = self.whisker(axes[0], x, y, predictedMoms, color='k', scale=2e-4)
            axes[0].quiverkey(qv0, 0.1, 0.9, 0.1, "0.1", coordinates='axes')
            axes[0].set_title("model")
            qv1 = self.whisker(axes[1], x, y, observedMoms, color='k', scale=2e-4)
            axes[1].quiverkey(qv1, 0.1, 0.9, 0.1, "0.1", coordinates='axes')
            axes[1].set_title("data")
            fig.tight_layout()
            pdf.savefig(fig, dpi=100)

        outfn = os.path.join(
            plotDir,
            "donutTripletUWhisker-{:07d}.pdf".format(focalVisit))

        with PdfPages(outfn) as pdf:
            fig, axes = subplots(1, 2, figsize=(8, 4))
            qv0 = self.unnormalizedWhisker(axes[0], x, y, predictedMoms, color='k', scale=1e-4)
            axes[0].quiverkey(qv0, 0.1, 0.9, 0.1, "0.1", coordinates='axes')
            axes[0].set_title("model")
            qv1 = self.unnormalizedWhisker(axes[1], x, y, observedMoms, color='k', scale=1e-4)
            axes[1].quiverkey(qv1, 0.1, 0.9, 0.1, "0.1", coordinates='axes')
            axes[1].set_title("data")
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
            scale=0.0001)
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
