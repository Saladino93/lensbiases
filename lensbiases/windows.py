from angularcls import windows
from numba import jit
import numpy as np
from scipy import interpolate

def get_useful_window_functions(Products, zoftruncation_low = 3., zoftruncation_high = 1100.):
    chikk, Wkkarr = Products.chikk, Products.Wkkarr

    @jit(nopython = True)
    def kindow_function(newchis, l = 0.):
        return np.interp(newchis, chikk, Wkkarr)
    
    zs = Products.zofchi(chikk)
    selection = 1-((zs > zoftruncation_low) & (zs < zoftruncation_high))
    Wkkarr_truncated = Wkkarr*selection

    @jit(nopython = True)
    def truncated_kwindow_function(newchis, l = 0.):
        return np.interp(newchis, chikk, Wkkarr_truncated)
    
    chis = Products.chi_precalculated
    zs = Products.zofchi(chis)

    phiwindow = (1/chis-1/Products.chistar)**2/chis**2
    phiwindow[0] = 0
    phiwindow_function_ = interpolate.interp1d(chis, phiwindow)
    phiwindow_function = lambda chi, l: phiwindow_function_(chi)

    phiwindow_function_truncated_ = interpolate.interp1d(chis, phiwindow*selection)
    phiwindow_function_truncated = lambda chi, l: phiwindow_function_truncated_(chi)
    
    return kindow_function, truncated_kwindow_function, phiwindow_function, phiwindow_function_truncated
    

    



