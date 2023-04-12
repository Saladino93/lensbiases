import numpy as np

import lensingresponse as lr

import vegas

from typing import Callable


@vegas.batchintegrand
class VegasLensingResponse(lr.LensingResponse):
    """
    Class to hold the lensing responses. It is specialezed for the vegas sampler as it allows the use of batches input ell modes.
    This is useful for the sampling phase where batches of ells are used for Monte Carlo evaluating integrals.
    """

    def __init__(self, Ls: np.ndarray, function: Callable = None, tot_function: Callable = None):
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
    

