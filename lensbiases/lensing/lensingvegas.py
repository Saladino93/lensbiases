import numpy as np

import lensingresponse as lr

import vegas

from typing import Callable

import numpy as np

import lensutils as lu








@vegas.batchintegrand
class VegasLensingResponse(lr.LensingResponse):
    """
    Class to hold the lensing responses. It is specialezed for the vegas sampler as it allows the use of batches input ell modes.
    This is useful for the sampling phase where batches of ells are used for Monte Carlo evaluating integrals.
    """

    def __init__(self, Ls: np.ndarray):
        """
        Parameters
        ----------
        Ls : array
            Array of ells where to calculate the given quantity.
        function : callable, optional
            Function to compute the lensing response for a XY quadratic estimator, by default None. The TT spectrum, e.g. gradient spectrum.
        tot_function : callable, optional
            Function to return the total power spectrum, signal+noise for given ells, by default None.
        """
        
        super().__init__()

        self.function = function
        self.tot_function = tot_function
        self.lmbdasin, self.lmbdacos = lu.get_sin_cos_exprs()


    def fgradEBbatch(l1v, l2v, l1n, l2n, gradientEE, gradientBB):
        Lv = l1v+l2v
        #calculate sin of double the angle between l1v and l2v
        factor = lmbdasin(l1n, l2n, l1v[0, :], l1v[1, :], l2v[0, :], l2v[1, :])
        return dotbatch(Lv, l1v)*gradientEE(l1n)+dotbatch(Lv, l2v)*gradientBB(l2n)*factor

    def funlTTbatch(l1v, l2v, l1n, l2n):
        Lv = l1v+l2v
        return dotbatch(Lv, l1v)*uTT(l1n)+dotbatch(Lv, l2v)*uTT(l2n)

    def flenTTbatch(l1v, l2v, l1n, l2n):
        Lv = l1v+l2v
        return dotbatch(Lv, l1v)*lTT(l1n)+dotbatch(Lv, l2v)*lTT(l2n)

    def fgradTTbatch(l1v, l2v, l1n, l2n, gradientTT):
        Lv = l1v+l2v
        return dotbatch(Lv, l1v)*gradientTT(l1n)+dotbatch(Lv, l2v)*gradientTT(l2n)

    def ftotTTfTTbatch(l):
        return tTT(l)

    def ftotEEfEEbatch(l):
        return tEE(l)

    def ftotBBfBBbatch(l):
        return tBB(l)

    def gfEBbatch(lv, Lv, l1n, l2n, gradientEE, gradientBB):
        l1v, l2v = lv, Lv-lv
        return fgradEBbatch(l1v, l2v, l1n, l2n, gradientEE, gradientBB)/(2*ftotEEfEEbatch(l1n)*ftotBBfBBbatch(l2n))

    def gfTTbatch(lv, Lv, l1n, l2n, gradientTT):
        l1v, l2v = lv, Lv-lv
        return fgradTTbatch(l1v, l2v, l1n, l2n, gradientTT)/(2*ftotTTfTTbatch(l1n)*ftotTTfTTbatch(l2n))

    def gfTTbatch_for_modes(l1v, l2v, l1n, l2n, gradientTT):
        return fgradTTbatch(l1v, l2v, l1n, l2n, gradientTT)/(2*ftotTTfTTbatch(l1n)*ftotTTfTTbatch(l2n))




    @staticmethod
    def dotbatch(a, b):
        """
        Dot product of two arrays of shape (2, batchsize).
        2
        Parameters
        ----------
        a : array
            First array.
        b : array
            Second array.
        Returns
        -------
        array
            Dot product of the two arrays along their first axis.
        """
        return a[0, :]*b[0, :]+a[1, :]*b[1, :] 
    

    @staticmethod
    def einsumdotbatch(a, b):
        """
        Dot product of two arrays of shape (2, batchsize). It uses einsum.

        Parameters
        ----------
        a : array
            First array.
        b : array
            Second array.
        Returns
        -------
        array
            Dot product of the two arrays along their first axis.
        """
        return np.dot("ij, ij -> j", a, b)
    

    def totalcl(self, l):
        """
        Function to return the total power spectrum, signal+noise for given ells.
        """
        return self.tot_function(l)
    

    def __call__(self, x):
        pass
    

@vegas.batchintegrand
class VegasAL(object):

    def __init__(self, f: Callable, g: Callable, lmin: int, lmax: int, nitn: int = 100, neval: int = 1e3):

        self.integ = vegas.Integrator([[lmin, lmax], [0, 2*np.pi]])
        self.nitn = nitn
        self.neval = neval
        self._current_L = None
        self.f = f
        self.g = g
        self.lmin, self.lmax = lmin, lmax


    def getALs(self, Ls):
        """
        Execute the vegas integration to get the response.
        """
        result = [self.integ(self, nitn = self.nitn, neval = self.neval).mean for self._current_L in Ls]
        return result
    
    def filter_batch(self, x):
        """
        Filter the batch of ells to remove the ones which are not valid.
        """
        return (x >= self.lmin) & (x <= self.lmax)


    def __call__(self, x):

        l1, theta1 = x.T
        l1v = np.array([l1*np.cos(theta1), l1*np.sin(theta1)])
        L = np.ones_like(l1)*self._current_L
        Lv = np.c_[L, np.zeros_like(l1)].T
        l3v = Lv-l1v

        l3 = np.linalg.norm(l3v, axis = 0)

        fXY = self.f(l1v, l3v, l1, l3)

        gXY = self.g(l1v, l3v, l1, l3)*self.filter_batch(l3)

        product = fXY*gXY     
        common = l1/(2*np.pi)**2

        return product*common
    

