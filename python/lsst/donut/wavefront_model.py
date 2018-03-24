import numpy as np
from galsim.zernike import Zernike, zernikeRotMatrix
from sklearn.decomposition import PCA


# http://stackoverflow.com/a/6849299
class lazy_property(object):
    """
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value


# Dict storing which Zernikes indices are fit together, singly,
# or are invalid (by their omission)
_nj = {
    4:1,
    5:2,
    7:2,
    9:2,
    11:1,
    12:2,
    14:2,
    16:2,
    18:2,
    20:2
}


def getFP(cat):
    """Get focal plane coordinates.

    Parameters
    ----------
    cat : (nDonut,) astropy.table.Table
        Table with at least the following columns
            x, y : float
                Telescope frame coordinates of donuts.
            hra : float
                Horizontal rotation angle in radians.

    Returns
    -------
    xFP, yFP : (nDonut,) ndarray
        Focal plane coordinates.
    """
    x = np.atleast_1d(cat['x'])
    y = np.atleast_1d(cat['y'])
    visit = np.atleast_1d(cat['visit'])
    hra = np.atleast_1d(cat['hra'])

    visits = np.unique(visit)
    nVisit = len(visits)

    xFP = np.zeros_like(x)
    yFP = np.zeros_like(y)

    for iVisit in range(nVisit):
        w = np.where(visits[iVisit] == visit)[0]
        thisHRA = hra[w[0]]
        xFP[w] = x[w]*np.cos(thisHRA) + y[w]*np.sin(thisHRA)
        yFP[w] = -x[w]*np.sin(thisHRA) + y[w]*np.cos(thisHRA)

    return xFP, yFP


class WFModelConfig:
    """Configuration for modeling wavefront as series of Double Zernike
    polynomials, including separate terms for the telescope, CCDs, and
    visit-level perturbations.

    Parameters
    ----------
    jMaxTel : int, optional
        Maximum pupil Zernike index to use for telescope contribution to
        wavefront.
    kMaxTel : int, optional
        Maximum focal Zernike index to use for telescope contribution to
        wavefront.
    jMaxCCD : int, optional
        Maximum pupil Zernike index to use for CCD contribution to
        wavefront.
    kMaxCCD : int, optional
        Maximum focal Zernike index to use for CCD contribution to
        wavefront.
    jMaxVisit : int, optional
        Maximum pupil Zernike index to use for visit contribution to
        wavefront.
    kMaxVisit : int, optional
        Maximum focal Zernike index to use for visit contribution to
        wavefront.
    regularizeCCD : float, optional
        Regularization coefficient to use for CCD contribution to
        wavefront.
    regularizeVisit : float, optional
        Regularization coefficient to use for visit contribution to
        wavefront.
    zeroCCDSum : bool, optional
        Require CCD solutions to sum to zero?
    zeroVisitSum : bool, optional
        Require visit solutions to sum to zero?

    Notes
    -----

    The CCD term is defined in the CCD coordinate frame, whereas the
    telescope and visit terms are defined in the telescope coordinate
    frame.  This means we need to apply a rotation to the CCD
    coefficients when fitting or evaluating the wavefront model.

    When zeroCCDSum (zeroVisitSum) is False, the least squares fit is
    underconstrained since it's possible in that case to exactly trade
    off between the telescope contribution to the wavefront and CCD
    (visit) contributions to the wavefront.  However, since the least
    squares fit is done with SVD of the design matrix, the returned
    solution will be the one which minimizes the L2-norm.  Since there
    are many more CCD (visit) terms than telescope terms, the telescope
    terms will tend to dominate.  The regularizeCCD (regularizeVisit)
    keyword can be used to adjust the relative contribution of the
    telescope and CCD (visit) by multipling the appropriate elements of
    the design matrix and solution vector by regularizeCCD
    (regularizeVisit) and its inverse.

    When zeroCCDSum (zeroVisitSum) is True, then the regularizeCCD
    (regularizeVisit) argument has no effect.
    """
    def __init__(self,
                 jMaxTel=21, kMaxTel=21,
                 jMaxCCD=21, kMaxCCD=1,
                 jMaxVisit=21, kMaxVisit=3,
                 regularizeCCD=1.0, regularizeVisit=1.0,
                 zeroCCDSum=False, zeroVisitSum=False):

        self.jMaxTel = jMaxTel
        self.kMaxTel = kMaxTel
        self.jMaxVisit = jMaxVisit
        self.kMaxVisit = kMaxVisit
        self.jMaxCCD = jMaxCCD
        self.kMaxCCD = kMaxCCD
        self.regularizeCCD = regularizeCCD
        self.regularizeVisit = regularizeVisit
        self.zeroCCDSum = zeroCCDSum
        self.zeroVisitSum = zeroVisitSum

        self.jMax = max(jMaxTel, jMaxVisit, jMaxCCD)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class WFDesignMatrix:
    """Class to hold the design matrix for the linear least squares fit
    to measure donut Zernike data.

    Parameters
    ----------
    config : WFModelConfig
        Configuration parameters for this fit.
    cat : (nDonut,) astropy.table.Table
        Table with at least the following columns
            x, y : float
                Telescope frame coordinates of donuts.
            visit : int
                Visit number for each donut.  0 to nVisit-1.
            ccd : int
                CCD number for each donut.  0 to nCCD-1.
            hra : float
                Horizontal rotation angle in radians.
            z4, ..., zj : float
                Columns for measured pupil Zernike coefficients.
    j : int
        Starting Zernike index for which to construct design matrix.
        Note that some Zernike terms must be fit together, like z5
        and z6.  In this case j=5 will work and j=6 will raise an
        Exception.  Either one or two Zernikes will be fit together.
    focalRadius : float, optional
        Radius to use for Zernikes defined over the focal plane.
        Default: 1.0.
    ccdRadius : float, optional
        Radius to use for Zernikes defined over a single CCD.  Only
        relevant if kMaxCCD > 4.  By default, this is set equal to
        focalRadius (which is fine if kMaxCCD <= 4, but may be
        problematic otherwise.)
    ccdCenter : dict of int -> 2-tuple of float, optional
        Dictionary of CCD centers in focal plane coordinates indexed
        by ccd number.  The default is (0.0, 0.0) for every CCD, which
        should be fine if you're only fitting a constant Zernike
        offset for each CCD, but may be suboptimal if fitting higher
        order CCD field dependence.
    """
    def __init__(self, config, cat, j,
                 focalRadius=1.0, ccdRadius=None, ccdCenter=None):
        # cat should have columns for:
        # x, y, ccd, visit, hra, z4, z5, etc.
        self.config = config
        self.cat = cat
        self.j = j

        self.nDonut = len(cat)
        self.nCCD = len(np.unique(cat['ccd']))
        self.nVisit = len(np.unique(cat['visit']))

        self.focalRadius = focalRadius
        if ccdRadius is None:
            ccdRadius = focalRadius
        self.ccdRadius = ccdRadius

        if ccdCenter is None:
            ccdCenter = {i:(0.0, 0.0) for i in range(self.nCCD)}
        self.ccdCenter = ccdCenter

    @lazy_property
    def array(self):
        """The actual design matrix array.
        """
        return self._getDesignMatrix()

    @lazy_property
    def colSlices(self):
        """Column slices indicating the different variables to be fit by
        the design matrix.
        """
        return self._colSlices()

    @lazy_property
    def rowSlices(self):
        """Row slices indicating the different data or constraint
        equations being solved.
        """
        return self._getRowSlices()

    def _colSlices(self):
        """Create column slices.

        Notes
        -----
        Columns are organized hierarchically as:
        - j in {j, j+1}
          - kTel in range(kMaxTel)
          - iCCD in range(nCCD)
            - kCCD in range(kMaxCCD)
          - iVisit in range(nVisit)
            - kVisit in range(kMaxVisit)

        Returns
        -------
        cSlices : dict
            Dictionary with keys for:
                - tel : List of slices
                    Which columns refer to telescope wavefront variables.
                    Either 1-element or 2-elements long depending on if
                    this Zernike j-value is fit independently or together
                    with j+1.
                - ccd : List of slices
                    Which columns refer to all CCD wavefront variables.
                - ccds : List of list of slices
                    Which columns refer to individual CCD wavefront
                    variables.
                - visit : List of slices
                    Which columns refer to visit wavefront variables.
                - visits : List of list of slices
                    Which columns refer to individual visit wavefront
                    variables.
                - njCol : int
                    Total number of columns per j value.
                - nCol : int
                    Total number of columns.
        """
        j, cfg = self.j, self.config

        nTelCol = cfg.kMaxTel if j <= cfg.jMaxTel else 0
        nCCDCol = cfg.kMaxCCD*self.nCCD if j <= cfg.jMaxCCD else 0
        nVisitCol = cfg.kMaxVisit*self.nVisit if j <= cfg.jMaxVisit else 0

        njCol = nTelCol + nCCDCol + nVisitCol
        nCol = _nj[j] * njCol

        telSlices = []
        ccdSlices = []
        ccdsSlices = []
        visitSlices = []
        visitsSlices = []
        offset = 0
        for i in range(_nj[j]):
            telSlices.append(
                    slice(offset,
                          offset + nTelCol)
            )
            ccdSlices.append(
                    slice(offset + nTelCol,
                          offset + nTelCol + nCCDCol)
            )
            ccdsSlices.append(
                [slice(offset + nTelCol + iCCD*cfg.kMaxCCD,
                       offset + nTelCol + (iCCD+1)*cfg.kMaxCCD)
                 for iCCD in range(self.nCCD)]
            )
            visitSlices.append(
                    slice(offset + nTelCol + nCCDCol,
                          offset + nTelCol + nCCDCol + nVisitCol)
            )
            visitsSlices.append(
                [slice(offset + nTelCol + nCCDCol + iVisit*cfg.kMaxVisit,
                       offset + nTelCol + nCCDCol + (iVisit+1)*cfg.kMaxVisit)
                 for iVisit in range(self.nVisit)]
            )

            offset += njCol

        return dict(
            tel=telSlices,
            ccd=ccdSlices,
            ccds=ccdsSlices,
            visit=visitSlices,
            visits=visitsSlices,
            njCol=njCol,
            nCol=nCol,
        )

    def _getRowSlices(self):
        """Create row slices.

        Notes
        -----
        Rows are organized hierarchically as:
        - j in {j, j+1}
          - iDonut in range(nDonut)
          - zeroCCDSum constraint
          - zeroVisitSum constraint

        Returns
        -------
        rSlices : dict
            Dictionary with keys for:
                - donut : List of slices
                    Which rows refer to donut measurements.  Either
                    1-element or 2-elements long depending on if this
                    Zernike j-value is fit independently or together with
                    j+1.
                - zeroCCDSum : List of slices
                    Which rows enforce the CCD sum constraint.
                - zeroVisitSum : List of slices
                    Which rows enforce the visit sum constraint.
                - njRow : int
                    Total number of rows per j value.
                - nRow : int
                    Total number of rows.
        """
        j, cfg = self.j, self.config

        offset = 0
        donutSlices = []
        zeroCCDSumSlices = []
        zeroVisitSumSlices = []
        for i in range(_nj[j]):
            donutSlices.append(
                slice(offset, offset + self.nDonut)
            )
            offset += self.nDonut
            if j <= cfg.jMaxCCD and cfg.zeroCCDSum:
                zeroCCDSumSlices.append(
                    slice(offset, offset + cfg.kMaxCCD)
                )
                offset += cfg.kMaxCCD
            if j <= cfg.jMaxVisit and cfg.zeroVisitSum:
                zeroVisitSumSlices.append(
                    slice(offset, offset + cfg.kMaxVisit)
                )
                offset += cfg.kMaxVisit

        nRow = offset
        njRow = offset // _nj[j]

        return dict(
            donut=donutSlices,
            zeroCCDSum=zeroCCDSumSlices,
            zeroVisitSum=zeroVisitSumSlices,
            njRow=njRow,
            nRow=nRow
        )

    def _getDesignMatrix(self):
        """Fill in design matrix array.

        Returns
        -------
        array : ndarray
            Design matrix.
        """
        cSlices = self.colSlices
        rSlices = self.rowSlices
        array = np.zeros((rSlices['nRow'], cSlices['nCol']), dtype=float)
        self._fillTelDesign(array)
        self._fillCCDDesign(array)
        self._fillVisitDesign(array)
        return array

    def _fillTelDesign(self, array):
        """Fill in telescope part of design matrix

        Parameters
        ----------
        array : ndarray
            Full design matrix array to be modified in place.
        """
        j, cfg = self.j, self.config
        if j > cfg.jMaxTel:
            return
        cSlices = self.colSlices
        rSlices = self.rowSlices

        for k in range(1, cfg.kMaxTel+1):
            Z = Zernike([0]*k+[1], R_outer=self.focalRadius)
            coefs = Z.evalCartesian(self.cat['x'], self.cat['y'])
            for i in range(_nj[j]):
                array[rSlices['donut'][i], cSlices['tel'][i].start+k-1] = coefs

    def _fillCCDDesign(self, array):
        """Fill in CCD part of design matrix

        Parameters
        ----------
        array : ndarray
            Full design matrix array to be modified in place.
        """
        j, cfg, cat = self.j, self.config, self.cat
        if j > cfg.jMaxCCD:
            return
        cSlices = self.colSlices
        rSlices = self.rowSlices

        visits = np.unique(cat['visit'])
        ccds = np.unique(cat['ccd'])

        xFP, yFP = getFP(cat)

        for iCCD in range(self.nCCD):
            for iVisit in range(self.nVisit):
                w = np.where(np.logical_and(
                    visits[iVisit] == cat['visit'],
                    ccds[iCCD] == cat['ccd']
                ))[0]

                if len(w) == 0:
                    continue

                thisHRA = cat['hra'][w[0]]
                # It's possible that I've got the sign of thisHRA wrong below,
                # or equivalently, transposed the R matrix.  Really need a good
                # unit test for this.
                if _nj[j] == 2:
                    R = zernikeRotMatrix(j+1, -thisHRA)
                    # R = zernikeRotMatrix(j+1, thisHRA)
                    R = R[j:j+2, j:j+2]

                for k in range(1, cfg.kMaxCCD+1):
                    Z = Zernike([0]*k+[1], R_outer=self.ccdRadius)

                    coefs = Z.evalCartesian(xFP[w]-self.ccdCenter[iCCD][0],
                                            yFP[w]-self.ccdCenter[iCCD][1])
                    coefs *= cfg.regularizeCCD
                    if _nj[j] == 1:
                        array[w, cSlices['ccds'][0][iCCD].start+k-1] = coefs
                    else:
                        array[w, cSlices['ccds'][0][iCCD].start+k-1] \
                            = coefs*R[0,0]
                        array[w, cSlices['ccds'][1][iCCD].start+k-1] \
                            = coefs*R[0,1]
                        array[w+rSlices['njRow'],
                              cSlices['ccds'][0][iCCD].start+k-1] \
                            = coefs*R[1,0]
                        array[w+rSlices['njRow'],
                              cSlices['ccds'][1][iCCD].start+k-1] \
                            = coefs*R[1,1]

                    if cfg.zeroCCDSum:
                        for i in range(_nj[j]):
                            array[rSlices['zeroCCDSum'][i].start+k-1,
                                  cSlices['ccds'][i][iCCD].start+k-1] = 1

    def _fillVisitDesign(self, array):
        """Fill in visit part of design matrix

        Parameters
        ----------
        array : ndarray
            Full design matrix array to be modified in place.
        """
        j, cfg, cat = self.j, self.config, self.cat
        if j > cfg.jMaxVisit:
            return
        cSlices = self.colSlices
        rSlices = self.rowSlices

        visits = np.unique(cat['visit'])

        for iVisit in range(self.nVisit):
            w = np.where(visits[iVisit] == cat['visit'])[0]
            for k in range(1, cfg.kMaxVisit+1):
                Z = Zernike([0]*k+[1], R_outer=self.focalRadius)
                coefs = Z.evalCartesian(self.cat['x'][w], self.cat['y'][w])
                for i in range(_nj[j]):
                    array[rSlices['donut'][i].start + w,
                          cSlices['visit'][i].start+iVisit*cfg.kMaxVisit+k-1] \
                        = coefs
                if cfg.zeroVisitSum:
                    for i in range(_nj[j]):
                        array[rSlices['zeroVisitSum'][i].start+k-1,
                        cSlices['visit'][i].start+iVisit*cfg.kMaxVisit+k-1] = 1


class WFFit:
    """Class to hold wavefront field-of-view dependence fit results.

    Parameters
    ----------
    config: WFModelConfig
        Configuration for fit.
    wTel : ndarray
        Telescope part of wavefront
    wCCD : ndarray
        CCD part of wavefront
    wVisit : ndarray
        Visit part of wavefront
    focalRadius : float
        Focal plane radius
    ccdCenter : dict of int -> 2-tuple of float, optional
        Dictionary of CCD centers in focal plane coordinates indexed by
        ccd number.  The default is (0.0, 0.0) for every CCD, which
        should be fine if you're only fitting a constant Zernike offset
        for each CCD, but may be suboptimal if fitting higher order CCD
        field dependence.

    """
    def __init__(self, config, wTel, wCCD, wVisit,
                 focalRadius, ccdRadius, ccdCenter):
        self.config = config
        self.wTel = wTel
        self.wCCD = wCCD
        self.wVisit = wVisit
        self.focalRadius = focalRadius
        self.ccdRadius = ccdRadius
        self.ccdCenter = ccdCenter

    def getTelWF(self, cat):
        """Retrieve the telescope part of the wavefront model from fit.

        Parameters
        ----------
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.

        Returns
        -------
        wf : (nDonut, jMax+1), ndarray
            Pupil Zernike coefficients of wavefront (in telescope
            coordinates) for each location in cat.
        """
        cfg = self.config

        x = np.atleast_1d(cat['x'])
        y = np.atleast_1d(cat['y'])

        out = np.zeros((len(x), cfg.jMaxTel+1), dtype=float)
        for j in range(1, cfg.jMaxTel+1):
            Z = Zernike(self.wTel[j], R_outer=self.focalRadius)
            out[:, j] = Z.evalCartesian(x, y)
        return out

    def getCCDWF(self, cat, doFocal=False):
        """Retrieve the CCD part of the wavefront model from fit.

        Parameters
        ----------
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.
                ccd : int
                    CCD number for each donut.  0 to nCCD-1.
                hra : float
                    Horizontal rotation angle in radians.
        doFocal : bool, optional
            Rotate Zernikes from telescope to focal coordinate system?
            Default is False, which is appropriate for combining CCD
            component with telescope or visit components.  However, it
            can be useful to use focal coords for plotting CCD
            contribution against focal plane coordinates.

        Returns
        -------
        wf : (nDonut, jMax+1), ndarray
            Pupil Zernike coefficients of CCD component of wavefront (in
            telescope coordinates) for each entry in cat.
        """
        cfg = self.config

        ccd = np.atleast_1d(cat['ccd'])
        hra = np.atleast_1d(cat['hra'])
        xFP, yFP = getFP(cat)

        hras = np.unique(hra)
        ccds = np.unique(ccd)

        nHRA = len(hras)
        nCCD = len(ccds)

        out = np.zeros((len(ccd), cfg.jMaxCCD+1), dtype=float)

        for iCCD in range(nCCD):
            w = np.where(ccds[iCCD] == ccd)[0]
            if len(w) == 0:
                continue
            for j in range(4, cfg.jMaxCCD+1):
                Z = Zernike(self.wCCD[iCCD, j], R_outer=self.ccdRadius)
                out[w, j] = Z.evalCartesian(
                    xFP[w] - self.ccdCenter[iCCD][0],
                    yFP[w] - self.ccdCenter[iCCD][1]
                )
        # Solution held by wffit is in focal plane coordinates.  For telescope
        # coordinates, we need to rotate:
        if not doFocal:
            for iHRA in range(nHRA):
                w = np.where(hras[iHRA] == hra)[0]
                if len(w) == 0:
                    continue
                thisHRA = hra[w[0]]
                R = zernikeRotMatrix(cfg.jMaxCCD, thisHRA)
                out[w, :] = np.dot(R, out[w, :].T).T
        return out

    def getVisitWF(self, cat):
        """Retrieve the visit part of the wavefront model from fit.

        Parameters
        ----------
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.
                visit : int
                    Visit number for each donut.  0 to nVisit-1.

        Returns
        -------
        wf : (nDonut, jMaxVisit+1), ndarray
            Pupil Zernike coefficients of visit component of wavefront
            (in telescope coordinates) for each entry in cat.
        """
        cfg = self.config

        x = np.atleast_1d(cat['x'])
        y = np.atleast_1d(cat['y'])
        visit = np.atleast_1d(cat['visit'])

        visits = np.unique(visit)
        nVisit = len(visits)

        out = np.zeros((len(x), cfg.jMaxVisit+1), dtype=float)
        for iVisit in range(nVisit):
            w = np.where(visits[iVisit] == visit)[0]
            for j in range(1, cfg.jMaxVisit+1):
                Z = Zernike(self.wVisit[iVisit, j], R_outer=self.focalRadius)
                out[w, j] = Z.evalCartesian(x[w], y[w])
        return out

    def getWF(self, cat):
        """Retrieve the complete wavefront model from fit.

        Parameters
        ----------
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.
                visit : int
                    Visit number for each donut.  0 to nVisit-1.
                ccd : int
                    CCD number for each donut.  0 to nCCD-1.
                hra : float
                    Horizontal rotation angle in radians.

        Returns
        -------
        wf : (nDonut, jMax+1), ndarray
            Pupil Zernike coefficients of wavefront (in telescope
            coordinates) for each entry in cat.

        """
        cfg = self.config
        nDonut = len(cat)

        out = np.zeros((nDonut, cfg.jMax+1), dtype=float)
        out[:, :cfg.jMaxTel+1] += self.getTelWF(cat)
        out[:, :cfg.jMaxCCD+1] += self.getCCDWF(cat)
        out[:, :cfg.jMaxVisit+1] += self.getVisitWF(cat)

        return out

class WFFoVFitter:
    """Class to use to fit the field-of-view dependence of measured
    Zernike coefficients, presumably from donut images.

    Parameters
    ----------
    config : WFModelConfig
        Configuration instance for fitter.
    """
    def __init__(self, config):
        self.config = config

    def fit(self, cat, focalRadius=1.0, ccdRadius=None, ccdCenter=None):
        """Fit the model.

        Parameters
        ----------
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.
                visit : int
                    Visit number for each donut.  0 to nVisit-1.
                ccd : int
                    CCD number for each donut.  0 to nCCD-1.
                hra : float
                    Horizontal rotation angle in radians.
                z4, ..., zj : float
                    Columns for measured pupil Zernike coefficients.
        focalRadius : float, optional
            Value to use for focal plane radius.
        ccdRadius : float, optional
            Value to use for CCD term radius.  Default is to use the same
            as for focalRadius.
        ccdCenter : dict of int -> 2-tuple of float, optional
            Dictionary of CCD centers in focal plane coordinates indexed
            by ccd number.  The default is (0.0, 0.0) for every CCD,
            which should be fine if you're only fitting a constant
            Zernike offset for each CCD, but may be suboptimal if fitting
            higher order CCD field dependence.

        Returns
        -------
        wffit : WFFitter
            Results of the fit.
        """
        cfg = self.config

        if ccdRadius is None:
            ccdRadius = focalRadius


        nCCD = len(np.unique(cat['ccd']))
        nVisit = len(np.unique(cat['visit']))

        if ccdCenter is None:
            ccdCenter = {iCCD:(0.0, 0.0) for iCCD in range(nCCD)}

        wTel = np.zeros((cfg.jMaxTel+1, cfg.kMaxTel+1), dtype=float)
        wCCD = np.zeros((nCCD, cfg.jMaxCCD+1, cfg.kMaxCCD+1), dtype=float)
        wVisit = np.zeros((nVisit, cfg.jMaxVisit+1, cfg.kMaxVisit+1),
                          dtype=float)

        wffit = WFFit(
            self.config,
            wTel, wCCD, wVisit,
            focalRadius, ccdRadius, ccdCenter
        )

        j = 4
        while j <= cfg.jMax:
            self._fitJ(j, cat, wffit)
            j += _nj[j]

        return wffit

    def _fitJ(self, j, cat, wffit):
        """Fit submodel.

        Parameters
        ----------
        j : int
            Pupil Zernike index to fit.
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.
                visit : int
                    Visit number for each donut.  0 to nVisit-1.
                ccd : int
                    CCD number for each donut.  0 to nCCD-1.
                hra : float
                    Horizontal rotation angle in radians.
                z4, ..., zj : float
                    Columns for measured pupil Zernike coefficients.
        wffit : WFFit
            Output location which will be updated.
        """
        cfg = self.config

        nCCD = wffit.wCCD.shape[0]
        nVisit = wffit.wVisit.shape[0]

        a = np.array([], dtype=float)
        for i in range(_nj[j]):
            key = 'z{}'.format(j+i)
            a = np.hstack([a, cat[key]])
            if j+i <= cfg.jMaxCCD and cfg.zeroCCDSum:
                a = np.hstack([a, [0]*cfg.kMaxCCD])
            if j+i <= cfg.jMaxVisit and cfg.zeroVisitSum:
                a = np.hstack([a, [0]*cfg.kMaxVisit])

        design = WFDesignMatrix(
            cfg, cat, j, wffit.focalRadius, wffit.ccdRadius, wffit.ccdCenter)
        result, res, rank, s = np.linalg.lstsq(design.array, a, rcond=1e-8)

        # Extract results...
        if j <= cfg.jMaxTel:
            for i in range(_nj[j]):
                wffit.wTel[j+i, 1:] = result[design.colSlices['tel'][i]]

        if j <= cfg.jMaxCCD:
            for i in range(_nj[j]):
                wffit.wCCD[:, j+i, 1:] = np.reshape(
                    result[design.colSlices['ccd'][i]],
                    (nCCD, cfg.kMaxCCD))

        if j <= cfg.jMaxVisit:
            for i in range(_nj[j]):
                wffit.wVisit[:, j+i, 1:] = np.reshape(
                    result[design.colSlices['visit'][i]],
                    (nVisit, cfg.kMaxVisit))


class WFFoVPCAPredictor:
    """Class to help predict wavefronts for previously unseen exposures.

    Parameters
    ----------
    wffit : WFFit
        Results of a donut Zernike fit.
    **kwargs
        Keyword arguments to pass forward to sklearn.decomposition.PCA
    """
    def __init__(self, wffit, **kwargs):
        self.wffit = wffit
        self.config = wffit.config
        self.pca = PCA(**kwargs)
        nVisit = self.wffit.wVisit.shape[0]
        self.pca.fit(self.wffit.wVisit.reshape(nVisit, -1))

    @lazy_property
    def pcs(self):
        return self.pca.components_.reshape(-1, *self.wffit.wVisit.shape[1:3])

    def getTelWF(self, cat):
        """Retrieve the telescope part of the wavefront model.

        Parameters
        ----------
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.

        Returns
        -------
        wf : (nDonut, jMax+1), ndarray
            Pupil Zernike coefficients of wavefront (in telescope
            coordinates) for each location in cat.
        """
        return self.wffit.getTelWF(cat)

    def getCCDWF(self, cat, doFocal=False):
        """Retrieve the CCD part of the wavefront model from fit.

        Parameters
        ----------
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.
                ccd : int
                    CCD number for each donut.  0 to nCCD-1.
                hra : float
                    Horizontal rotation angle in radians.
        doFocal : bool, optional
            Rotate Zernikes from telescope to focal coordinate system?
            Default is False, which is appropriate for combining CCD
            component with telescope or visit components.  However, it
            can be useful to use focal coords for plotting CCD
            contribution against focal plane coordinates.

        Returns
        -------
        wf : (nDonut, jMax+1), ndarray
            Pupil Zernike coefficients of CCD component of wavefront (in
            telescope coordinates) for each entry in cat.
        """
        return self.wffit.getCCDWF(cat, doFocal)

    def getVisitWF(self, cat, coefs):
        """Retrieve the per-visit part of the wavefront model from PC
        coefficients.

        Parameters
        ----------
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.
        coefs : (nPCA,), array_like
            PC coefficients to use to construct visit part of wavefront.

        Returns
        -------
        wf : (nDonut, jMax+1), ndarray
            Pupil Zernike coefficients of visit component of wavefront (in
            telescope coordinates) for each entry in cat.
        """
        cfg = self.config

        x = np.atleast_1d(cat['x'])
        y = np.atleast_1d(cat['y'])

        out = np.zeros((len(x), cfg.jMaxVisit+1), dtype=float)
        wVisit = np.dot(self.pcs.T, coefs).T

        for j in range(4, cfg.jMaxVisit+1):
            Z = Zernike(wVisit[j], R_outer=self.wffit.focalRadius)
            out[:, j] = Z.evalCartesian(x, y)
        return out

    def getWF(self, cat, coefs):
        """Retrieve wavefront model given PC coefficients.

        Parameters
        ----------
        cat : (nDonut,) astropy.table.Table
            Table with at least the following columns
                x, y : float
                    Telescope frame coordinates of donuts.
                ccd : int
                    CCD number for each donut.  0 to nCCD-1.
                hra : float
                    Horizontal rotation angle in radians.
        coefs : (nPCA,), array_like
            PC coefficients to use to construct visit part of wavefront.

        Returns
        -------
        wf : (nDonut, jMax+1), ndarray
            Pupil Zernike coefficients of wavefront (in telescope
            coordinates) for each entry in cat.
        """
        cfg = self.config
        nDonut = len(cat)

        out = np.zeros((nDonut, cfg.jMax+1), dtype=float)
        out[:, :cfg.jMaxTel+1] += self.getTelWF(cat)
        out[:, :cfg.jMaxCCD+1] += self.getCCDWF(cat)
        out[:, :cfg.jMaxVisit+1] += self.getVisitWF(cat, coefs)

        return out

    def getProjections(self):
        """Retrieve projections of training data onto principal
        components.

        Returns
        -------
        projections : (nVisit, nPC), ndarray
            Projections of each visit onto PCs.
        """
        wVisit = self.wffit.wVisit
        nVisit = wVisit.shape[0]
        wVisit = wVisit.reshape(nVisit, -1)

        pcs = self.pcs
        nPC = pcs.shape[0]
        pcs = pcs.reshape(nPC, -1)

        out = np.empty((nVisit, nPC), dtype=float)

        for i, visit in enumerate(wVisit):
            out[i] = np.dot(pcs, visit)
        return out
