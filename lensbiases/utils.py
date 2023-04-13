from functools import wraps

import numpy as np


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