import vegas

import numpy as np

import numpy as np

import bispectrum_3D_numba as b3n

import temperatureinfo as ti

from scipy import interpolate

from timeit import timeit

from tqdm import tqdm


@vegas.batchintegrand
class cmb_sky:
    def __init__(self, lmax = 4000, lmin = 10, 
                gradientf = None, Wsinterp = None,
                totalf = None, almcf = None):

        self.lmax = lmax
        self.lmin = lmin

        self.gradientf = gradientf

        self.totalf = totalf
        
        self.Wsinterp = Wsinterp

        self.almcf = almcf

    @staticmethod
    def dotbatch(a, b):
        return a[0, :]*b[0, :]+a[1, :]*b[1, :]

    def filter_batch(self, x):
        return (x >= self.lmin) & (x <= self.lmax)

    def Qbatch(self, a, b, c):
        return 0.5*self.dotbatch(b, c)*self.dotbatch(b, a-b-c)

    def Kbatch(self, a, b):
        return -self.dotbatch(b, a-b)

    def fgradTTbatch(self, l1v, l2v, l1n, l2n, gradientTT):
        Lv = l1v+l2v
        return self.dotbatch(Lv, l1v)*gradientTT(l1n)+self.dotbatch(Lv, l2v)*gradientTT(l2n)

    def gfTTbatch_base(self, lv, Lv, l1n, l2n, gradientTT, totalTT):
        l1v, l2v = lv, Lv-lv
        return self.fgradTTbatch(l1v, l2v, l1n, l2n, gradientTT)/(2*totalTT(l1n)*totalTT(l2n))

    def gfTTbatch(self, lv, Lv, l1n, l2n):
        return self.gfTTbatch_base(lv, Lv, l1n, l2n, self.gradientf, self.totalf)
        



@vegas.batchintegrand
class f_batch(cmb_sky):
    def __init__(self, L, index = 2, itr = 0, lmax = 4000, lmin = 10, **kwargs):
        super().__init__(lmin = lmin, lmax = lmax, **kwargs)
        self.L = L
        self.index = index
        self.itr = itr


@vegas.batchintegrand
class f_n32_base(f_batch):
    def __init__(self, L, index = 2, lmax = 4000, itr = 0, lmin = 10, **kwargs):
        super().__init__(L = L, index = index, itr = itr, lmin = lmin, lmax = lmax, **kwargs)

    def __call__(self, x):

        LL = self.L
        index = self.index

        lmax = self.lmax
        lmin = self.lmin

        l1, l2, theta1, theta2 = x.T
        cos1, cos2 = np.cos(theta1), np.cos(theta2)
        sin1, sin2 = np.sin(theta1), np.sin(theta2)

        l1v = np.array([l1*cos1, l1*sin1])
        l2v = np.array([l2*cos2, l2*sin2])

        l5v = l1v-l2v

        L = np.ones_like(l1)*LL
        Lv = np.c_[L, np.zeros_like(l1)].T

        l3v = Lv-l1v
        l4v = Lv-l2v

        l5 = np.sqrt((l1*cos1-l2*cos2)**2+(l1*sin1-l2*sin2)**2)
        l4 = np.sqrt(LL**2+l2**2-2*LL*l2*cos2)
        l3 = np.sqrt(LL**2+l1**2-2*LL*l1*cos1)

        l2_dot_L = (LL*l2*cos2)
                
        l1_dot_l2 = (l1*cos1*l2*cos2) + (l1*sin1*l2*sin2)
        l2_dot_l5 = l1_dot_l2-l2**2.
        l1_dot_l5 = l1**2-l1_dot_l2

        gXY = self.gfTTbatch(l2v, Lv, l2, l4)*self.filter_batch(l4)
        gYX = gXY

        l1_dot_l2 = (l1*cos1*l2*cos2) + (l1*sin1*l2*sin2)
        L_dot_l1 = LL*l1*cos1
        l1_dot_l3 = L_dot_l1-l1**2
        l2_dot_l3 = l2_dot_L-l1_dot_l2
        l5_dot_l1 = l1**2-l1_dot_l2
        l5_dot_l3 = LL*l1*cos1-l2_dot_L-l1**2+l1_dot_l2

        cl5_XY = self.gradientf(l5)*self.filter_batch(l5)
        Cl2 = self.gradientf(l2)

        hX_l5_l2 = 1.
        hY_l5_l4 = 1.
        hY_l2_l4 = 1.
        hX_l2_l4 = 1.

        productA1 = -l5_dot_l1*l5_dot_l3*hX_l5_l2*hY_l5_l4*cl5_XY*gXY
        productC1 = l2_dot_l3*l1_dot_l2*(gXY*hY_l2_l4+gYX*hX_l2_l4)*Cl2*1/2
        productD1 = -2*l1_dot_l3*l1_dot_l2*gXY*Cl2

        born_term = 0.
        W1 = self.Wsinterp(l1) if self.itr > 0 else 1
        W3 = self.Wsinterp(l3) if self.itr > 0 else 1

        factor = 1. if self.itr == 0 else np.nan_to_num((1-W1)*(1-W3))
        factorC = factor
        factorD = 1

        productA1 *= factor
        productC1 *= factorC
        productD1 *= factorD

        bispectrum_result = b3n.bispec_phi_general(l1, l3, LL, index)
        bispectrum_total = bispectrum_result+born_term

        common = l1*l2*bispectrum_result/(2*np.pi)**4

        A1 = productA1*common
        C1 = productC1*common
        D1 = productD1*common

        result = A1+C1+D1

        return {"B": result, "A1": A1, "C1": C1, "D1": D1}


def main():

    #settings
    bpmodel = "GM"
    indices = {"TR": 0, "SC": 1, "GM": 2}
    index = indices[bpmodel]

    qe_key = "ptt"

    noise, beam = 1., 1.

    delcls_true = np.load(f"delcls_true_{qe_key}.npy", allow_pickle = True)

    pps = [d["pp"] for d in delcls_true]

    N0s_biased = np.load(f"N0s_biased_{qe_key}.npy", allow_pickle = True)
    N1s_biased = np.load(f"N1s_biased_{qe_key}.npy", allow_pickle = True)

    gradients = np.load(f"gradients_{qe_key}.npy", allow_pickle = True)

    ALMCSextended = np.load(f"ALMCSextended_{qe_key}.npy", allow_pickle = True)
    Lsextended = np.load(f"Lsextended_{qe_key}.npy", allow_pickle = True)

    almc_functions = [interpolate.interp1d(Lsextended, A, fill_value = 0., bounds_error = False) for A in ALMCSextended]

    Ntotal = [N0+N1 for N0, N1 in zip(N0s_biased, N1s_biased)]
    Wflist = [np.nan_to_num(pps[0][:N.size]/(pps[0][:N.size]+N)) for N in Ntotal]

    ls = np.arange(0, len(gradients[0]))

    gradientfs = [interpolate.interp1d(ls, gradient, fill_value = 0., bounds_error = False) for gradient in gradients]
        
    Wsinterps = [interpolate.interp1d(np.arange(0, len(W)), W, fill_value = 0., bounds_error = False) for W in Wflist]

    ls = np.arange(0, len(delcls_true[0]["tt"]))
    noise_component = ti.get_noise(ls, noise, beam) #constant among iterations
    totalfs = [interpolate.interp1d(ls, delcls["tt"]+noise_component, fill_value = 1e10, bounds_error = False) for delcls in delcls_true]

    lmin, lmax = 10, 4000

    integ = vegas.Integrator([[lmin, lmax], [lmin, lmax], [0, 2*np.pi], [0, 2*np.pi]], nhcube_batch = 8000, nproc = 8)#6000, 4 |||nhcube_batch = 8000, nproc = 8

    integ_1 = vegas.Integrator([[lmin, lmax], [0, 2*np.pi]], nhcube_batch = 2000, nproc = 4)

    nitn, neval = 4e2, 600
    #nitn, neval = 10, 100

    Ls = np.arange(10, 500, 50)
    Ls = np.concatenate((Ls, np.arange(500, 3500, 150)))

    itr = 1

    #keys = ["B", "A1", "C1", "D1"]
    #NTOT = {k: [] for k in keys}

    def get_result(L):
        #integrand = f_n32_base(L, index = index, itr = itr, lmin = lmin, lmax = lmax,
        #                    gradientf = gradientfs[itr], Wsinterp = Wsinterps[itr-1] if itr > 0 else None, totalf = totalfs[itr])
        #result = integ(integrand, nitn = nitn, neval = neval)

        integrand = f_n32_gradient_GA(L, index = index, itr = itr, lmin = lmin, lmax = lmax,
                            gradientf = gradientfs[itr], Wsinterp = Wsinterps[itr-1] if itr > 0 else None, 
                            totalf = totalfs[itr], almcf = almc_functions[itr])
        result = integ(integrand, nitn = nitn, neval = neval)

        #for k in result.keys():
        #    NTOT[k] += [result[k].mean]
        return [[result[k].mean for k in result.keys()]]

    #results = np.array([get_result(L) for L in tqdm(Ls)])
    results = np.vstack([get_result(L) for L in tqdm(Ls)])

    np.savetxt("base_n32_magn_term_GA.txt", np.c_[Ls, results])

if __name__ == '__main__':
    print(timeit(lambda: main(), number = 1))
    #main()