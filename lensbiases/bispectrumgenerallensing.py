"""
Gets the lensing bispectrum from different models.
"""

import numpy as np

import bispectrumgeneral as bg

from angularcls import windows, cosmoconstants

import scipy.interpolate as interp, scipy.integrate as sinteg


class BispectrumLensing(bg.Bispectrum3D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.Wkk = windows.cmblensingwindow_ofchi(self.chis, self.aofchis, self.H0, self.Omegam, interp1d = True, chistar = self.chistar)
        
        Wphiphiv = np.nan_to_num(-2*(self.chistar-self.chis)/(self.chistar*self.chis))
        Wphiphiv[0] = 0
        self.Wphiphi = interp.interp1d(self.chis, Wphiphiv, bounds_error = True)

        gammav = 3/2*self.H0**2*self.Omegam/(cosmoconstants.CSPEEDKMPERSEC**2)/self.aofchis
        gamma = interp.interp1d(self.chis, gammav, bounds_error = True)#, fill_value = 'extrapolate')

    def bispectrum_matter_2d(self, l1, l2, l3, theta12, theta13, theta23, z, model = 'TR'):
        return self.bispectrum_matter(l1, l2, l3, theta12, theta13, theta23, z, model = model)
    
    @np.vectorize
    def bispectrum_k_with_angles(self, l1, l2, l3, angle12, angle13, angle23, model = 'TR', maxiter = 100, miniter = 50, rtol = 1e-12):
        assert l1.shape == l2.shape == l3.shape == angle12.shape == angle13.shape == angle23.shape
        bispectrum_at_ells_of_chi = lambda chi: chi**(-4)*self.Wkk(chi)**3*self.bispectrum_matter(l1/chi, l2/chi, l3/chi, angle12, angle13, angle23, self.zofchi(chi), model = model)
        #return sinteg.quadrature(bispectrum_at_ells_of_chi, 0, integrated_bispectrum.chistar, maxiter = 50, rtol = 1e-8)[0]
        return sinteg.quadrature(bispectrum_at_ells_of_chi, 1e-12, self.chistar, maxiter = maxiter, miniter = miniter, rtol = rtol)[0]
    
    @staticmethod
    def ktophi_bispec(l1, l2, l3):
        return 8/(l1*l2*l3)**2
    
    @staticmethod
    def get_angle_12(L1, L2, L3):
        term = (L1**2+L2**2-L3**2)/(2*L1*L2)
        return np.arccos(term)
    
    @staticmethod
    def get_angle_cos12(L1, L2, L3):
        return (L1**2+L2**2-L3**2)/(2*L1*L2)

    def bispectrum_phi_with_angles(self, l1, l2, l3, angle12, angle13, angle23, model = 'TR', maxiter = 100, miniter = 50, rtol = 1e-12):
        factor = self.ktophi_bispec(l1, l2, l3)
        return factor*self.bispectrum_k(l1, l2, l3, angle12, angle13, angle23, model = model, maxiter = maxiter, miniter = miniter, rtol = rtol)
    
    
    def bispectrum_equilateral(self, l: np.ndarray, model: str):
        angle12, angle13, angle23 = np.pi/3, np.pi/3, np.pi/3
        return self.bispectrum_k_with_angles(l, l, l, angle12, angle13, angle23, model = model)
    
    def bispectrum_k(self, l1, l2, l3, model = 'TR', maxiter = 100, miniter = 50, rtol = 1e-12):
        angle12, angle13, angle23 = self.get_angle_12(l1, l2, l3), self.get_angle_12(l1, l3, l2), self.get_angle_12(l2, l3, l1)
        return self.bispectrum_k_with_angles(l1, l2, l3, angle12, angle13, angle23, model = model, maxiter = maxiter, miniter = miniter, rtol = rtol)

    def bispectrum_phi(self, l1, l2, l3, model = 'TR', maxiter = 100, miniter = 50, rtol = 1e-12):
        angle12, angle13, angle23 = self.get_angle_12(l1, l2, l3), self.get_angle_12(l1, l3, l2), self.get_angle_12(l2, l3, l1)
        return self.bispectrum_phi_with_angles(l1, l2, l3, angle12, angle13, angle23, model = model, maxiter = maxiter, miniter = miniter, rtol = rtol)