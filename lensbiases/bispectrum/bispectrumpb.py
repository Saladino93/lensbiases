"""
Postborn bispectrum.

https://github.com/cmbant/notebooks/blob/master/PostBorn.ipynb 

with modifications and on numba steroids.

TODO: save quantities for numba version.
TODO: work with galaxy and galaxy-lensing support
"""

import numpy as np

import numba as nb

from . import utils, cosmo

from typing import Callable

from scipy import interpolate as interp


class PostBorn(cosmo.Cosmology):

    def __init__(self, Lprimes: np.ndarray, **kwargs):
        """
        Lprimes: array of L' values to use in the calculation of the M matrix.
        """
        super().__init__(**kwargs)

        self.Lprimes = []
        self.Pminterpolator = lambda z, k: self.PKW.P(z, k, grid = False)

        self.M = self.getMmatrix(Lprimes)



    @staticmethod
    #@nb.njit
    def Ca2kappa_Lprime_atchi(Ls, chis, zs_at_chis, gaussian_weights, Wa_ofchi, Wkappa_ofchi, Pminterpolator, kmax: float = 100):

        result = np.empty(Ls.shape)
        kernel = Wa_ofchi*Wkappa_ofchi/chis**2
        Ns = len(Ls)
        w = np.ones(chis.shape)
        for i in range(Ns):
            L = Ls[i]
            ks = (L+1/2)/chis
            w[:] = 1
            w[ks < 1e-4] = 0
            w[ks >= kmax] = 0
            result[i] = np.dot(gaussian_weights, Pminterpolator(zs_at_chis, ks)*kernel*w/ks**4)
        result *= Ls**4
        return result
    
    def _Ca2kappa_Lprime_atchi(self, Ls, chis, zs_at_chis, gaussian_weights, Wa_ofchi, Wkappa_ofchi, Pminterpolator, kmax: float = 100):

        result = np.empty(Ls.shape)
        kernel = Wa_ofchi*Wkappa_ofchi/chis**2
        Ns = len(Ls)
        w = np.ones(chis.shape)
        for i in range(Ns):
            L = Ls[i]
            ks = (L+1/2)/chis
            w[:] = 1
            w[ks < 1e-4] = 0
            w[ks >= kmax] = 0
            result[i] = np.dot(gaussian_weights, Pminterpolator(zs_at_chis, ks)*kernel*w/ks**4)
        result *= Ls**4 
        return result
    
    def Ca2kappa_Lprime_atchi_from_chisource(self, Ls, chis, zs_at_chis, gaussian_weights, chi_source_A, chi_source_B, Pminterpolator, kmax: float = 100.):
        Wa_ofchi = (1/chis-1/chi_source_A)
        Wkappa_ofchi = (1/chis-1/chi_source_B)
        return self.Ca2kappa_Lprime_atchi(Ls, chis, zs_at_chis, gaussian_weights, Wa_ofchi, Wkappa_ofchi, Pminterpolator, kmax)
    
    def get_cl_chi(self, Lprimes, Nquadrature: int = 50):
        nchimax = 100*2
        chimaxs = np.linspace(0, self.chistar, nchimax)
        cls = np.zeros((nchimax, Lprimes.size))
        for i, chimax in enumerate(chimaxs[1:]):
            chis, gaussian_weights = utils.gaussxw(0, chimax, Nquadrature)
            zs_at_chis = self.results.redshift_at_comoving_radial_distance(chis)
            clkappa = self.Ca2kappa_Lprime_atchi_from_chisource(Lprimes, chis, zs_at_chis, gaussian_weights, chimax, self.chistar, self.Pminterpolator)
            cls[i+1, :] = clkappa
        cls[0, :]=0    
        return interp.RectBivariateSpline(chimaxs, Lprimes, cls)


    def getMmatrix(self, Lprimes, Nquadrature: int = 50, kmax: float = 100.):

        chis, gaussian_weights = utils.gaussxw(0, self.chistar, Nquadrature)
        zs = self.results.redshift_at_comoving_radial_distance(chis)

        win = (1/chis-1/self.chistar)**2/chis**2
        cl = np.zeros(Lprimes.shape)
        w = np.ones(chis.shape)
        self.cl_chi = self.get_cl_chi(Lprimes)
        cchi = self.cl_chi(chis, Lprimes, grid = True)

        M = np.zeros((Lprimes.size, Lprimes.size))

        for i, l in enumerate(Lprimes):
            k = (l+0.5)/chis
            w[:] = 1
            w[k < 1e-4] = 0
            w[k >= kmax] = 0
            cl = np.dot(gaussian_weights*w*self.Pminterpolator(zs, k)*win/k**4, cchi)
            M[i, :] = cl*l**4 #(l*(l+1))**2

        Mf = interp.RectBivariateSpline(Lprimes, Lprimes, M)
        return Mf


    #@np.vectorize
    def one_term_bispectrum_born(self, L1, L2, L3):
        return self._one_term_bispectrum_born(L1, L2, L3, self.M)

    @staticmethod
    def _one_term_bispectrum_born(L1, L2, L3, M: Callable):
        cos12 = utils.get_angle_cos12(L1, L2, L3)
        cos13 = utils.get_angle_cos12(L1, L3, L2)
        cos23 = utils.get_angle_cos12(L2, L3, L1)
        result = 2*cos12/(L1*L2)*(L1*L3*cos13*M(L1, L2)+L2*L3*cos23*M(L2, L1))
        return result

    @utils.vectorize(signature = '(),(),(),()->()')
    def bispectrum_PB(self, L1, L2, L3):
        Lslist = [L1, L2, L3]
        options = [[0, 1, 2], [2, 1, 0], [0, 2, 1]] #list, list inverse, list last elements inverse
        return np.sum([self.one_term_bispectrum_born(Lslist[option[0]], Lslist[option[1]], Lslist[option[2]]) for option in options], axis = 0)

    def bispectrum_PB_phi(self, L1, L2, L3):
        return self.bispectrum_PB(L1, L2, L3)*utils.ktophi_bispec(L1, L2, L3)