import vegas

import numpy as np

import numpy as np

from lensbiases import temperatureinfo as ti, bispectrum_3D_numba as b3n
from lensbiases import integrated_pb as pb, products, windows


from scipy import interpolate

from timeit import timeit

from tqdm import tqdm

import pathlib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out_name", type = str, default = "n32")
parser.add_argument("--inputdir", type = str, default = "numbaproducts")
parser.add_argument("--bpmodel", type = str, default = "GM")
parser.add_argument("--qe_key", type = str, default = "ptt")
parser.add_argument("--noise", type = float, default = 1.)
parser.add_argument("--beam", type = float, default = 1.)
parser.add_argument("--lmin", type = int, default = 10)
parser.add_argument("--lmax", type = int, default = 4000)
parser.add_argument("--version", type = str, default = "")

args = parser.parse_args()
out_name = args.out_name
bpmodel = args.bpmodel
qe_key = args.qe_key
noise = args.noise
beam = args.beam
lmin = args.lmin
lmax = args.lmax
version = args.version

path = pathlib.Path(args.inputdir)

P = products.Products(path)

#because of legacy code I need to track windows in chi, for kappa or phi
#the current postborn code wants phi, while the lss code wants kappa
kindow_function, truncated_kwindow_function, phiwindow_function, phiwindow_function_truncated = windows.get_useful_window_functions(P, zoftruncation_low = 3., zoftruncation_high = 1100.)

if version == "":
    window_lss, window_pb = None, None
elif version == "window":
    window_lss, window_pb = kindow_function, phiwindow_function
elif version == "truncated":
    window_lss, window_pb = truncated_kwindow_function, phiwindow_function_truncated


bispec_phi_general = b3n.bispectrum_3D_numba(path, window = window_lss)
bispec_pb = pb.get_pb_bispectrum(H0 = P.H0, ombh2 = P.ombh2, omch2 = P.omch2,
                      As = P.As, ns = P.ns, mnu = P.mnu, num_massive_neutrinos = P.num_massive_neutrinos,
                      window = window_pb)


@vegas.batchintegrand
class cmb_sky:
    def __init__(self, lmax = 4000, lmin = 10, 
                gradientf = None, totalf = None):

        self.lmax = lmax
        self.lmin = lmin

        self.gradientf = gradientf

        self.totalf = totalf


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

        bispectrum_postborn_result = bispec_pb(l1, l3, LL)*8/(l1*l3*LL)**2 #this is in kappa and then you multuply by some factor to get phi

        bispectrum_result = bispec_phi_general(l1, l3, LL, index)

        common = l1*l2*bispectrum_result/(2*np.pi)**4
        common_postborn = l1*l3*bispectrum_postborn_result/(2*np.pi)**4

        A1 = productA1*common
        C1 = productC1*common

        A1_postborn = productA1*common_postborn
        C1_postborn = productC1*common_postborn

        result = A1+C1
        result_postborn = A1_postborn+C1_postborn

        total = result+result_postborn

        return {"TOT": total, "B": result, "PB": result_postborn, "A1": A1, "C1": C1}


def main(out_name: str = "n32", bpmodel: str = "GM", qe_key: str = "ptt", noise: float = 1., beam: float = 1.
         , lmin: int = 10, lmax: int = 4000, Ls = np.concatenate((np.arange(10, 500, 50.), np.arange(500, 3500, 200.))),
         neval:int = 1000, nitn:int = 8e2, nproc:int = 8, nhcube_batch:int = 8000, version:str = ""):

    #settings
    indices = {"TR": 0, "SC": 1, "GM": 2}
    index = indices[bpmodel]

    delcls_true = np.load(f"delcls_true_{qe_key}.npy", allow_pickle = True)

    gradients = np.load(f"gradients_{qe_key}.npy", allow_pickle = True)

    ALMCSextended = np.load(f"ALMCSextended_{qe_key}.npy", allow_pickle = True)
    Lsextended = np.load(f"Lsextended_{qe_key}.npy", allow_pickle = True)

    almc_functions = [interpolate.interp1d(Lsextended, A, fill_value = 0., bounds_error = False) for A in ALMCSextended]

    almc_QE = almc_functions[0]

    ls = np.arange(0, len(gradients[0]))

    gradientfs = [interpolate.interp1d(ls, gradient, fill_value = 0., bounds_error = False) for gradient in gradients]
        
    ls = np.arange(0, len(delcls_true[0]["tt"]))
    noise_component = ti.get_noise(ls, noise, beam) #constant among iterations
    totalfs = [interpolate.interp1d(ls, delcls["tt"]+noise_component, fill_value = 1e10, bounds_error = False) for delcls in delcls_true]

    integ = vegas.Integrator([[lmin, lmax], [lmin, lmax], [0, 2*np.pi], [0, 2*np.pi]], nhcube_batch = 8000, nproc = 8)#6000, 4 |||nhcube_batch = 8000, nproc = 8

    itr = 0

    Ls = Ls.astype(float)

    def get_result(L):
        integrand = f_n32_base(L, index = index, itr = itr, lmin = lmin, lmax = lmax,
                            gradientf = gradientfs[itr], totalf = totalfs[itr])
        result = integ(integrand, nitn = nitn, neval = neval)

        return [[result[k].mean for k in result.keys()]]

    results = np.vstack([get_result(L) for L in tqdm(Ls)])

    version = f"_{version}" if version != "" else version
    np.savetxt(f"../results/{out_name}{version}.txt", np.c_[Ls, results])

if __name__ == '__main__':
    print("Configuration: ", args)
    main(out_name = out_name, bpmodel = bpmodel, qe_key = qe_key, noise = noise, beam = beam, lmin = lmin, lmax = lmax, version = version, neval = 500, nitn = 200)
    #print(timeit(lambda: , number = 1))
    #main()