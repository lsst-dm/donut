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
import matplotlib.patches as patches

import lsst.afw.cameraGeom as afwCameraGeom
from lsst.daf.persistence.safeFileIo import safeMakeDir


def getPlotDir(butler, dataset, dataId):
    dirName = os.path.dirname(butler.get(dataset+"_filename", dataId)[0])
    plotDir = os.path.abspath(os.path.join(dirName, "..", "plots"))
    safeMakeDir(plotDir)
    return plotDir


def plotCameraOutline(axes, camera, doCcd=True):
    axes.tick_params(labelsize=6)
    axes.locator_params(nbins=6)
    axes.ticklabel_format(useOffset=False)
    camRadius = max(camera.getFpBBox().getWidth(),
                    camera.getFpBBox().getHeight())/2
    camRadius = np.round(camRadius, -2)
    camLimits = np.round(1.15*camRadius, -2)
    if doCcd:
        for ccd in camera:
            ccdCorners = ccd.getCorners(afwCameraGeom.FOCAL_PLANE)
            axes.add_patch(patches.Rectangle(
                ccdCorners[0], *list(ccdCorners[2] - ccdCorners[0]),
                fill=False, edgecolor="k", ls="solid", lw=0.5))
    axes.set_xlim(-camLimits, camLimits)
    axes.set_ylim(-camLimits, camLimits)
    axes.add_patch(patches.Circle(
        (0, 0), radius=camRadius, color="black", alpha=0.1))
