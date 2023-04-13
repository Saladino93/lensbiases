"""
Just an interface for camb.
"""

import camb
from camb import model as cmodel

import numpy as np

import scipy.interpolate as sinterp




class Cosmology(object):
    def __init__(self, H0:float, ombh2:float, omch2:float, As:float, ns:float, mnu:float = 0, num_massive_neutrinos:float = 0, tau:float = 0.0925, RECFAST_fudge: float = 1.14, halofit_version: str = 'takahashi'):
        pars = camb.CAMBparams()
        ommh2 = ombh2+omch2
        h = H0/100
        Omegam = ommh2/h**2

        self.H0 = H0
        self.Omegam = Omegam
    
        pars.set_cosmology(H0 = H0, ombh2 = ombh2, omch2 = omch2, mnu = mnu, num_massive_neutrinos = num_massive_neutrinos)
        pars.InitPower.set_params(As = As, ns = ns)

        # reionization and recombination 
        pars.Reion.use_optical_depth = True
        pars.Reion.optical_depth = tau #tau
        pars.Reion.delta_redshift = 0.5
        pars.Recomb.RECFAST_fudge = RECFAST_fudge

        #non linearity
        pars.NonLinear = cmodel.NonLinear_both
        pars.NonLinearModel.halofit_version = halofit_version
        pars.Accuracy.AccurateBB = True #need this to avoid small-scale ringing

        self.pars = pars
        self.results = camb.get_background(pars)

        self.compute_quantities()

    
    def compute_quantities(self, nz: int = 6000, kmax: float = 100., var1: str = 'delta_nonu', var2: str = 'delta_nonu'):
        """
        Compute the quantities needed for the lensing bispectrum.

        Parameters
        ----------
        nz : int, optional
            Number of steps to use for the radial/redshift integration, by default 6000
        kmax : float, optional
            kmax to use, by default 100
        var1 : str, optional
            First variable to use, by default 'delta_nonu'
        var2 : str, optional
            Second variable to use, by default 'delta_nonu'
        
        Returns
        -------
        """

        self.chistar = self.results.conformal_time(0)- self.results.tau_maxvis
        self.chis = np.linspace(0, self.chistar, nz)
        self.zs = self.results.redshift_at_comoving_radial_distance(self.chis)

        self.aofchis = 1/(1+self.zs)

        self.zofchi = sinterp.interp1d(self.chis, self.zs, kind = 'cubic', fill_value = 'extrapolate', bounds_error = False)

        self.Hzs = self.results.hubble_parameter(self.zs)

        self.PK = camb.get_matter_power_interpolator(self.pars, nonlinear = True, 
    hubble_units = False, k_hunit = False, kmax = kmax, k_per_logint = None,
    var1 = var1, var2 = var2, zmax = self.zs[-1])

        self.PKlin = camb.get_matter_power_interpolator(self.pars, nonlinear = False, 
            hubble_units = False, k_hunit = False, kmax = kmax, k_per_logint = None,
            var1 = var1, var2 = var2, zmax = self.zs[-1])
        

        zm = np.logspace(-9, np.log10(1089), 140)
        zm = np.append(0, zm)
        self.pars.set_matter_power(redshifts = zm, kmax = kmax)
        results = camb.get_results(self.pars)
        self.s8 = np.array(results.get_sigma8()[::-1])
        self.zm = zm