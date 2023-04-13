"""
Module to get the perturbation theory tree level bispectrum from different models.
"""

from .import nonlinearcosmology as nlc

import numpy as np

import itertools


class Bispectrum3D(nlc.NonLinear):

    def F2ptker_vector(self, k1, k2, theta12, z, model = 'TR'):
        """
        Calculates F2 kernel from PT.

        Parameters
        ----------
        k1vec : array_like
            3-vector of k1, shape, (3, n), where n is the number of points to be calculated at.
        k2vec : array_like
            3-vector of k2, shape, (3, n), where n is the number of points to be calculated at.
        z : array_like
            Redshifts, shape, (1, n).
        """
        afunc, bfunc, cfunc = self.getfuncs(model)
        resultG = 5/7*afunc(z, k1, grid = False)*afunc(z, k2, grid = False)
        resultS = bfunc(z, k1, grid = False)*bfunc(z, k2, grid = False)*1/2*(k1/k2 + k2/k1)*np.cos(theta12)
        resultT = 2/7*(np.cos(theta12))**2*cfunc(z, k1, grid = False)*cfunc(z, k2, grid = False)
        return resultG + resultS + resultT
    
    def bispectrum_matter(self, k1, k2, k3, theta12, theta13, theta23, z, model = 'TR'):
        ksvec = [k1, k2, k3]
        combinations = list(itertools.combinations([0,1,2], 2))
        thetas = [theta12, theta13, theta23] #assume this is the order too from combinations
        return sum([2*self.F2ptker_vector(ksvec[comb[0]], ksvec[comb[1]], thetaij, z, model = model)*self.PK.P(z, ksvec[comb[0]], grid = False)*self.PK.P(z, ksvec[comb[1]], grid = False) for comb, thetaij in zip(combinations, thetas)])



