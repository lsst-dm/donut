#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import absolute_import, division, print_function
import unittest
import numpy as np
import galsim

import lsst.utils.tests
from lsst.donut.utilities import zernikeRotMatrix
import lsst.afw.geom as afwGeom


def evalZernike(coefficients, coords):
    # Use GalSim internals to evaluate Zernike polynomials.
    aberrations = [0]
    aberrations.extend(coefficients)
    screen = galsim.OpticalScreen(aberrations=aberrations)
    rho = coords[:, 0] + 1j * coords[:, 1]
    # Fake a galsim.Aperture
    class Aperture():
        pass
    aper = Aperture()
    aper.rho = rho
    return screen.wavefront(aper, compact=False)


class ZernikeRotMatrixTest(lsst.utils.tests.TestCase):
    """Test lsst.donut.utilities.zernikeRotMatrix"""

    def setUp(self):
        testXCoords = np.random.uniform(size=100)
        testYCoords = np.random.uniform(size=100)
        w = np.hypot(testXCoords, testYCoords) < 1
        self.testCoords = np.vstack([testXCoords[w], testYCoords[w]]).T

        self.coefficientArrays = []
        self.zernikeValues = []
        self.jmax = [4, 10, 11, 15, 21]
        for jmax in self.jmax:
            coefficients = np.array([0, 0, 0], dtype=np.float64)
            coefficients = np.append(
                coefficients,
                np.random.uniform(size=(jmax-3))
            )
            self.coefficientArrays.append(coefficients)
            self.zernikeValues.append(
                evalZernike(coefficients, self.testCoords)
            )

        self.thetas = np.random.uniform(low=0.0, high=2.0*np.pi, size=10)
        self.rotatedTestCoords = []
        for theta in self.thetas:
            sth, cth = np.sin(theta), np.cos(theta)
            rot = np.array([[cth, sth], [-sth, cth]])
            self.rotatedTestCoords.append(np.dot(rot, self.testCoords.T).T)

    def testRot(self):
        for jmax, coefficients, zernikeValues in zip(self.jmax,
                                                     self.coefficientArrays,
                                                     self.zernikeValues):
            for theta, rotatedTestCoords in zip(self.thetas,
                                                self.rotatedTestCoords):
                M = zernikeRotMatrix(jmax, -theta * afwGeom.radians)
                rotatedCoefficients = np.dot(M, coefficients)
                # We can use GalSim OpticalScreen to see if we actually rotated.
                rotatedValues = evalZernike(
                    rotatedCoefficients,
                    rotatedTestCoords
                )
                np.testing.assert_allclose(
                    zernikeValues,
                    rotatedValues,
                    rtol = 0,
                    atol = 1e-7
                )

    def testInput(self):
        # All of the below jmax fail because they need a slot for jmax+1 in
        # the output coefficient array.
        for jmax in [5, 7, 9, 12, 14, 16]:
            with self.assertRaises(ValueError):
                zernikeRotMatrix(jmax, 1*afwGeom.radians)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == '__main__':
    lsst.utils.tests.init()
    unittest.main()
