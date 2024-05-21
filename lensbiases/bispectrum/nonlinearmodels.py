"""
Implemented SC and GM models.
"""

import numpy as np
from typing import Callable

def getSCcoeffs():
    a1 = 0.250
    a2 = 3.50 
    a3 = 2.00 
    a4 = 1.00 
    a5 = 2.00
    a6 = -0.200 
    a7 = 1.00 
    a8 = 0.00 
    a9 = 0.00
    return a1, a2, a3, a4, a5, a6, a7, a8, a9

def getGMcoeffs():
    a1 = 0.484
    a2 = 3.74
    a3 = -0.849
    a4 = 0.392
    a5 = 1.01
    a6 = -0.575
    a7 = 0.128
    a8 = -0.722
    a9 = -0.926
    return a1, a2, a3, a4, a5, a6, a7, a8, a9

def getTRcoeffs():
    return [0 for _ in range(9)]

def get_coeffs(model):
    if model == "GM":
        return getGMcoeffs()
    elif model == "SC":
        return getSCcoeffs()
    elif model == "TR":
        return getTRcoeffs()
    

def get_afuncs_form_coeffs_for_interp(model, s8: Callable, Q: Callable, nefff: Callable, kNLzf: Callable):
    """
    NOTE: if you want to use TR model better not call this here. This one supports TR model just for generality. But you still call useless functions for the TR model.
    """
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = get_coeffs(model)
    afuncGM = lambda z, k: (1+s8(z)**a6*np.sqrt(0.7*Q(nefff(z, k)))*(k/kNLzf(z)*a1)**(nefff(z, k)+a2))/(1+(a1*k/kNLzf(z))**(nefff(z, k)+a2))
    bfuncGM = lambda z, k: (1+0.2*a3*(nefff(z, k)+3)*(k/kNLzf(z)*a7)**(nefff(z, k)+3+a8))/(1+(a7*k/kNLzf(z))**(nefff(z, k)+3.5+a8))
    cfuncGM = lambda z, k: (1+(4.5*a4/(1.5+(nefff(z, k)+3)**4)))*(k/kNLzf(z)*a5)**(nefff(z, k)+3+a9)/(1+(a5*k/kNLzf(z))**(nefff(z, k)+3.5+a9))
    return [afuncGM, bfuncGM, cfuncGM]