"""
Calculating kkk postborn results. Adapted from an Antony Lewis' notebook.


#TODO
Generalise the window used.
"""


import numpy as np
import scipy
import itertools


import integrated_bispectrum
results = integrated_bispectrum.results

import camb
from camb import model as cmodel



def Ma1a2kappa(weights, Wa_ofchi, Wkappa_ofchi, Ca2kappa_Lprime_ofchi, Pm):
    return np.dot(weights, Wa_ofchi, Wkappa_ofchi, Ca2kappa_Lprime_ofchi, Pm)

def Ca2kappa_Lprime_atchi(Ls, chis, zs_at_chis, gaussian_weights, Wa_ofchi, Wkappa_ofchi, Pminterpolator, kmax = 100):

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


def Ca2kappa_Lprime_atchi_from_chisource(Ls, chis, zs_at_chis, gaussian_weights, chi_source_A, chi_source_B, Pminterpolator, kmax = 100):
    Wa_ofchi = (1/chis-1/chi_source_A)
    Wkappa_ofchi = (1/chis-1/chi_source_B)
    return Ca2kappa_Lprime_atchi(Ls, chis, zs_at_chis, gaussian_weights, Wa_ofchi, Wkappa_ofchi, Pminterpolator, kmax)


def gaussxw(a, b, N):
    x, w = np.polynomial.legendre.leggauss(N)
    return 0.5*(b-a)*x + 0.5*(b+a), 0.5*(b-a)*w

Nquadrature = 100
chistar = integrated_bispectrum.chistar
chis, gaussian_weights = gaussxw(0, chistar, Nquadrature)
zs_at_chis = results.redshift_at_comoving_radial_distance(chis)
kmax = 100
PK = camb.get_matter_power_interpolator(integrated_bispectrum.pars, nonlinear = True, 
    hubble_units=False, k_hunit=False, kmax=kmax,k_per_logint=None,
    var1=cmodel.Transfer_Weyl,var2=cmodel.Transfer_Weyl, zmax=1100)


Pminterpolator = lambda z, k: PK.P(z, k, grid = False)


Lprimes = np.arange(1, 6000, 5)

nchimax = 100

chimaxs = np.linspace(0 ,chistar, nchimax)

cls = np.zeros((nchimax, Lprimes.size))

chi_source = chistar
Wa_ofchi = (1/chis-1/chi_source)
Wkappa_ofchi = (1/chis-1/chi_source)
clkappa = Ca2kappa_Lprime_atchi(Lprimes, chis, zs_at_chis, gaussian_weights, Wa_ofchi, Wkappa_ofchi, Pminterpolator)
plt.semilogx(Lprimes, clkappa)

def cl_kappa(chi_source, chi_source2=None):
    chi_source = np.float64(chi_source)
    if chi_source2 is None: 
        chi_source2 = chi_source
    else:
        chi_source2 = np.float64(chi_source2)
    chis = np.linspace(0,chi_source,2*100, dtype=np.float64)
    zs=results.redshift_at_comoving_radial_distance(chis)
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]
    win = (1/chis-1/chi_source)*(1/chis-1/chi_source2)/chis**2
    cl=np.zeros(Lprimes.shape)
    w = np.ones(chis.shape)
    for i, l in enumerate(Lprimes):
        k=(l+0.5)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=kmax]=0
        cl[i] = np.dot(dchis,
            w*PK.P(zs, k, grid=False)*win/k**4)
    cl*= Lprimes**4 #(ls*(ls+1))**2
    return cl

nchimax = 100*2
chimaxs = np.linspace(0 ,chistar, nchimax)
cls = np.zeros((nchimax,Lprimes.size))
for i, chimax in enumerate(chimaxs[1:]):
    #cl = cl_kappa(chimax)
    #p = plt.semilogx(Lprimes,cl, alpha = 0.4)

    chis, gaussian_weights = gaussxw(0, chimax, Nquadrature)
    zs_at_chis = results.redshift_at_comoving_radial_distance(chis)
    clkappa = Ca2kappa_Lprime_atchi_from_chisource(Lprimes, chis, zs_at_chis, gaussian_weights, chimax, chistar, Pminterpolator)
    cls[i+1,:] = clkappa
    #plt.semilogx(Lprimes, clkappa, ls = '--', color = p[0].get_color())

cls[0,:]=0    

cl_chi = scipy.interpolate.RectBivariateSpline(chimaxs, Lprimes, cls)


#Get M(l,l') matrix
chis, gaussian_weights = gaussxw(0, chistar, Nquadrature)
zs = results.redshift_at_comoving_radial_distance(chis)

win = (1/chis-1/chistar)**2/chis**2
cl = np.zeros(Lprimes.shape)
w = np.ones(chis.shape)
cchi = cl_chi(chis, Lprimes, grid = True)

M = np.zeros((Lprimes.size, Lprimes.size))

for i, l in enumerate(Lprimes):
    k = (l+0.5)/chis
    w[:] = 1
    w[k < 1e-4] = 0
    w[k >= kmax] = 0
    cl = np.dot(gaussian_weights*w*PK.P(zs, k, grid = False)*win/k**4, cchi)
    M[i,:] = cl*l**4 #(l*(l+1))**2

Mf = scipy.interpolate.RectBivariateSpline(Lprimes, Lprimes, np.log(M))
Msp = scipy.interpolate.RectBivariateSpline(Lprimes, Lprimes, M)

def get_angle_cos12(L1, L2, L3):
    return (L1**2+L2**2-L3**2)/(2*L1*L2)

def one_term_bispectrum_born(L1, L2, L3):
    cos12 = get_angle_cos12(L1, L2, L3)
    cos13 = get_angle_cos12(L1, L3, L2)
    cos23 = get_angle_cos12(L2, L3, L1)

    result = 2*cos12/(L1*L2)*(L1*L3*cos13*Msp(L1, L2)+L2*L3*cos23*Msp(L2, L1))
    return result

@np.vectorize
def bispectrum_Born(L1, L2, L3):
    Lslist = [L1, L2, L3]
    options = [[0, 1, 2], [2, 1, 0], [0, 2, 1]]
    return np.sum([one_term_bispectrum_born(Lslist[option[0]], Lslist[option[1]], Lslist[option[2]]) for option in options], axis = 0)


@np.vectorize
def bi_born(l1,l2,l3):
    cos12 = (l3**2-l1**2-l2**2)/2/l1/l2
    cos23 = (l1**2-l2**2-l3**2)/2/l2/l3
    cos31 = (l2**2-l3**2-l1**2)/2/l3/l1
    return  - 2*cos12*((l1/l2+cos12)*Msp(l1,l2,grid=False) + (l2/l1+cos12)*Msp(l2,l1, grid=False) )  \
            - 2*cos23*((l2/l3+cos23)*Msp(l2,l3,grid=False) + (l3/l2+cos23)*Msp(l3,l2, grid=False) )  \
            - 2*cos31*((l3/l1+cos31)*Msp(l3,l1,grid=False) + (l1/l3+cos31)*Msp(l1,l3 ,grid=False) ) 


