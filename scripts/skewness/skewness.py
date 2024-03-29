"""
This script computes the skewness of the CMB lensing potential. 

Useful to compare theory with non-Gaussian/N-Body simulations.
"""

import numpy as np

import pickle

from joblib import Parallel, delayed

from numba import jit, prange

import bispectrum_3D_numba as b3n

import interpolated_quantities_numba as iqn

import time

import threej

#https://github.com/toshiyan/cmblensplus/blob/dcd212906da8039f63839d69e8bb45ebccd55d09/F90/src_utils/bstool.f90#L988
@jit(nopython = True, fastmath = True)
def W3j_approx(l1,l2,l3):
  #ind = np.where((l1+l2+l3)%2 != 0)
  if (l1+l2+l3)%2 != 0:
    result = 0
  else:
    Lh = (l1+l2+l3)*0.5
    a1 = ((Lh-l1+0.5)/(Lh-l1+1))**(Lh-l1+0.25)
    a2 = ((Lh-l2+0.5)/(Lh-l2+1))**(Lh-l2+0.25)
    a3 = ((Lh-l3+0.5)/(Lh-l3+1))**(Lh-l3+0.25)
    b = 1/((Lh-l1+1)*(Lh-l2+1)*(Lh-l3+1))**(0.25)
    result = (-1)**Lh/np.sqrt(2*np.pi) * np.exp(1.5)* (Lh+1)**(-0.25) * a1*a2*a3*b
  #result[ind] = 0
  return result

@jit(nopython = True, fastmath = True)
def geom_factor(l1, l2, l3):
    result = np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*np.pi))
    result *= W3j_approx(l1, l2, l3)
    return result

@jit(nopython = True, fastmath = True)
def get_angle_cos12(L1, L2, L3):
    return (L1**2+L2**2-L3**2)/(2*L1*L2)

def WR(l, sigma):
    return np.exp(-(l*(l+1))/2*sigma**2)

def sigma_from_R(Rradians):
    return Rradians / (2.0 * np.sqrt(2.0 * np.log(2.0)))

#Rdeg = R/60
#Rradians = np.deg2rad(Rdeg)
#sigma = Rradians / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@jit(nopython = True, fastmath = True)
def bispecTR(l1, l2, l3):
    cangle12, cangle13, cangle23 = b3n.get_angle_cos12(l1, l2, l3), b3n.get_angle_cos12(l1, l3, l2), b3n.get_angle_cos12(l2, l3, l1)
    result = b3n.chipow_4_times_Wkk3_pre_calc
    bispec_arr = np.empty(b3n.xsgauss.size)
    #for i, x in enumerate(b3n.xsgauss):
    for i in prange(b3n.xsgauss.size):
        x = b3n.xsgauss[i]
        bispec_arr[i] = b3n.bispectrum_matter_cos_TR(l1/x, l2/x, l3/x, cangle12, cangle13, cangle23, iqn.zofchi(x))
    somma = np.dot(result*bispec_arr, b3n.wsgauss)
    return somma


@jit(nopython = True, fastmath = True)
def bispecGM(l1, l2, l3):
    cangle12, cangle13, cangle23 = b3n.get_angle_cos12(l1, l2, l3), b3n.get_angle_cos12(l1, l3, l2), b3n.get_angle_cos12(l2, l3, l1)
    result = b3n.chipow_4_times_Wkk3_pre_calc
    bispec_arr = np.empty(b3n.xsgauss.size)
    #for i, x in enumerate(b3n.xsgauss):
    for i in prange(b3n.xsgauss.size):
        x = b3n.xsgauss[i]
        bispec_arr[i] = b3n.bispectrum_matter_cos_GM(l1/x, l2/x, l3/x, cangle12, cangle13, cangle23, iqn.zofchi(x))
    somma = np.dot(result*bispec_arr, b3n.wsgauss)
    return somma

@jit(nopython = True, fastmath = True)
def bispecSC(l1, l2, l3):
    cangle12, cangle13, cangle23 = b3n.get_angle_cos12(l1, l2, l3), b3n.get_angle_cos12(l1, l3, l2), b3n.get_angle_cos12(l2, l3, l1)
    result = b3n.chipow_4_times_Wkk3_pre_calc
    bispec_arr = np.empty(b3n.xsgauss.size)
    #for i, x in enumerate(b3n.xsgauss):
    for i in prange(b3n.xsgauss.size):
        x = b3n.xsgauss[i]
        bispec_arr[i] = b3n.bispectrum_matter_cos_SC(l1/x, l2/x, l3/x, cangle12, cangle13, cangle23, iqn.zofchi(x))
    somma = np.dot(result*bispec_arr, b3n.wsgauss)
    return somma

lmaxes = [10, 30, 40, 50, 80, 100, 200, 300, 400, 600, 1000] #[10, 20, 30, 40, 50, 80, 100, 200, 300, 400, 600, 1000, 2000] #[10, 20, 30, 40, 50, 80, 100, 200, 300, 400, 600, 1000, 1100, 2000, 2500, 3000]
#lmaxes = [600, 800, 1000, 1100, 2000, 2500, 3000]

dirnew = "allowed_configs_results"
#dirthree = "allowed_configs_results_threej"
direc = dirnew

GM, TR, SC = "GM", "TR", "SC"
models = [TR, SC, GM]
caso = GM
#index = models.index(caso)

if caso == TR:
    bispec = bispecTR
elif caso == GM:
    bispec = bispecGM
elif caso == SC:
    bispec = bispecSC

batch_size = 'auto'
n_jobs = 4
backend = "loky"

@jit(nopython = True, fastmath = True, parallel = True)
def loop(l1, lmin, lmax):
    somma = 0.
    #lmin = l1
    for l2 in prange(lmin, lmax):
        """
        #numba float array 
        out = np.empty(3*lmax, dtype=np.float64)
        lista = threej.threejj(l1, l2, 0, 0, out)
        l3min, numbers = lista
        l3min = (l3min+1) if l3min in [0, 1] else l3min
        numbers = numbers[1:] if l3min in [0, 1] else numbers
        for i, l3 in enumerate(range(l3min, l3min+len(numbers))) :#range(l2, lmax):
        """
        for l3 in prange(l2, lmax):
            #check triangle conditions
            if (l3>l1+l2) or (l3<abs(l1-l2)):
                continue
            if (l1>l2+l3) or (l1<abs(l2-l3)):
                continue
            if (l2>l3+l1) or (l2<abs(l3-l1)):
                continue
            if (((l1+l2+l3)%2)==1):
                continue      

            #factor = numbers[i]*np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*np.pi))
            factor = geom_factor(l1, l2, l3)

            somma += bispec(l1, l2, l3)*factor**2
    
            #results.append(W3j_approx(l1, l2, l3))
            #results += [(l1, l2, l3)] #=W3j_approx(l1, l2, l3)
    return somma


#note: for now I am recomputing the same modes for same lmax. 
#Have to find a clean way to save intermediate results.
for lmax in lmaxes:

    lmin = 2

    ells = np.arange(lmin, lmax, 1)

    start = time.time()
    results = Parallel(n_jobs = n_jobs, batch_size = batch_size, backend = backend, verbose = 0)(delayed(loop)(l1, lmin, lmax) for l1 in ells)
    #print(results)
    #[loop(l1) for l1 in ells]
    end = time.time()
    print(f"Time (s) for {lmax} is", end - start)

    with open(f"{direc}/{caso}_allowed_configs_{lmax}", "wb") as fp: 
        pickle.dump(results, fp)

#with open("allowed_configs", "rb") as fp:
#    results = pickle.load(fp)
