"""
Helper to calculate non-linear models for the matter power spectrum.

Most of the code is mine. Found helpful looking at VBohm and TNamikawa's codes. 

In particular for definition of kNL and pars for smoothing the slope of the power spectrum.

"""

import numpy as np

import cosmo

import nonlinearmodels as nlm

from scipy import optimize as sopt, interpolate as interp

import findiff

from typing import Callable


class NonLinear(cosmo.Cosmology):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        zm = np.logspace(-9, np.log10(1089), 140)
        zm = np.append(0, zm)

        self.kNLzf = self._compute_NL_scale_function(self.zs, self.PKlin.P)
        self.kNLz = self.kNLzf(self.zs)

        self.nslopef = self._compute_slope(self.PKlin.P)

        self.s8f = self._compute_s8(self.zs, self.s8)

        self.getfuncs = self._process_NL_models(zm, self.s8f, self.Q, self.kNLzf, self.nslopef)

    @staticmethod
    def Q(x):
        return (4-2**x)/(1+2**(x+1))


    @staticmethod
    def nonlinearscale(z, k, PL):
        return PL(z, k)*k**3/(2*np.pi**2.)-1 #4*np.pi*k**3*PKlin.P(z, k)-1
    
    def _compute_NL_scale_function(self, zs, PL):
        kNLz = []
        for z in zs:
            nonlinearscale_ = lambda k: self.nonlinearscale(z, k, PL)
            kstar = sopt.brentq(nonlinearscale_, 1e-5, 1e5)
            kNLz.append(kstar)
        kNLzf = interp.interp1d(zs, kNLz, kind = 'cubic', fill_value = 'extrapolate')
        return kNLzf

    def _compute_slope(self, PL):
        """
        Smooth a bit the derivative of the power spectrum. See 1111.4477v2, Figure 1.
        
        This is done by hand here... For now.
        """

        kgrid = np.log(np.logspace(-5, 2, 1000))
        zgrid = np.append(0, np.logspace(-5, 3, 1000))
        
        Pgrid = np.log(PL(zgrid, np.exp(kgrid), grid = True))

        dkgrid = kgrid[1]-kgrid[0]
        #another way would just to have a differentiable P...
        d2_dx1 = findiff.FinDiff(0, dkgrid, 1)

        neff2D = []

        ksneff = np.exp(kgrid) #np.logspace(-3, 2, 500)
        kneff_min, kneff_max = 0.001, 1
        weights = np.ones_like(ksneff)
        weights[ksneff < kneff_min] = 100
        weights[ksneff > kneff_max] = 100

        for i, Pgrid_ in enumerate(Pgrid):
            neff = d2_dx1(Pgrid_)
            #neff_temp = interp.interp1d(np.exp(kgrid), neff, kind = 'cubic', fill_value = 'extrapolate')
            
            yhat = interp.UnivariateSpline(ksneff, neff, s = 40, w = weights) #s=40, by hand.

            neff2D += [yhat(ksneff)]
            #neff2D += [neff]

        neff2D = np.array(neff2D)

        #from Antony Lewis' code, 
        #found this from looking at VBohm's code
        #https://github.com/cmbant/notebooks/blob/master/PostBorn.ipynb

        nk = PL(0.1, np.log(ksneff), grid = False, dy = 1)
        w = np.ones(nk.size)
        w[ksneff < 5e-3]=100
        w[ksneff > 1]=10
        nksp =  interp.UnivariateSpline(np.log(ksneff), nk, s = 30, w = w)

        nefff_ = interp.RectBivariateSpline(zgrid, np.exp(kgrid), neff2D)
        nefff = lambda z, k: nksp(np.log(k)) 

        return nefff
    
    def _interpolate_s8(self, zs, s8z):
        s8zf = interp.interp1d(zs, s8z, kind = 'cubic', fill_value = 'extrapolate')
        return s8zf

    def _process_NL_models(self, zm: np.ndarray, s8: Callable, Q: Callable, kNLzf: Callable, nefff: Callable):
        ks = np.logspace(-5, 2, 300)
        zmmesh, ksmesh = np.meshgrid(zm, ks, indexing = 'ij')
        fit_funcs = {}
        for model in ['GM', 'SC']:
            fit_funcs[model] = [interp.RectBivariateSpline(zm, ks, fun(zmmesh, ksmesh)) for fun in nlm.get_afuncs_form_coeffs_for_interp(model, s8, Q, kNLzf, nefff)]
        
        def get_afuncs_form_coeffs(model):
            return fit_funcs[model]
        
        afuncTR, bfuncTR, cfuncTR = [lambda k, z, grid: np.ones_like(k)]*3

        def getfuncs(model):
            if model == 'TR':
                return afuncTR, bfuncTR, cfuncTR
            elif model in ['GM', 'SC']:
                return get_afuncs_form_coeffs(model)

        return getfuncs
    
    @staticmethod
    def dot(a, b):
        return np.einsum('ij, ij->j', a, b)