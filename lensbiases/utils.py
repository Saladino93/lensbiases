from functools import wraps

import numpy as np


def gaussxw(a, b, N):
    x, w = np.polynomial.legendre.leggauss(N)
    return 0.5*(b-a)*x + 0.5*(b+a), 0.5*(b-a)*w


def get_angle_12(L1, L2, L3):
        term = (L1**2+L2**2-L3**2)/(2*L1*L2)
        return np.arccos(term)
    
def get_angle_cos12(L1, L2, L3):
    return (L1**2+L2**2-L3**2)/(2*L1*L2)

def ktophi_bispec(l1, l2, l3):
    return 8/(l1*l2*l3)**2


#Inspired from https://stackoverflow.com/a/50556493
def vectorize(otypes = None, signature = None, excluded = None):
    """Numpy vectorization wrapper that works with instance methods."""
    def decorator(fn):
        vectorized = np.vectorize(fn, otypes = otypes, signature = signature, excluded = excluded)
        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)
        return wrapper
    return decorator