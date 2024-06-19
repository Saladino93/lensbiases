"""
We have a class to manage products.
"""

import numpy as np
import yaml

from scipy import interpolate

from interpolation.splines import UCGrid, eval_linear

class Products:

    def __init__(self, path, cosmology_file = "cosmology.yaml"):
        self.path = path
        self.cosmology = yaml.safe_load(open(path/cosmology_file, "r"))
        self.cosmology["As"] = float(self.cosmology["As"])

        #setattr all the cosmological parameters
        for key, value in self.cosmology.items():
            setattr(self, key, value)

        self.chikk, self.Wkkarr = np.loadtxt(path / "Wkk.txt", unpack = True)
        self.chi_precalculated, self.z_precalculated = np.loadtxt(path / "zs.txt", unpack = True)

        self.chistar = self.chi_precalculated[-1]
        
        self.chiofz = interpolate.interp1d(self.z_precalculated, self.chi_precalculated)
        self.zofchi = interpolate.interp1d(self.chi_precalculated, self.z_precalculated)

        self.as_precalculated = 1/(1+self.z_precalculated)
        self.aofchi = interpolate.interp1d(self.chi_precalculated, self.as_precalculated)

        _, _, self.Hzs = np.loadtxt(path / "Hzs.txt", unpack = True)
        self.Hz = interpolate.interp1d(self.z_precalculated, self.Hzs)

        self.zKNL, self.kNLz = np.loadtxt(path / "kNL.txt", unpack = True)
        self.zs8, self.s8arr = np.loadtxt(path / "sigma8.txt", unpack = True)
        self.names = ["non_lin", "lin", "neff"]
        self.grid2d = np.loadtxt(path/"z_k_matter_powers.txt")
        minlogz, minlogk = self.grid2d.min(axis = 0)
        maxlogz, maxlogk = self.grid2d.max(axis = 0)
        NN = int(np.sqrt(self.grid2d.shape[0]))
        self.grid = UCGrid((minlogz, maxlogz, NN), (minlogk, maxlogk, NN))
        self.valuesP = np.loadtxt(path/f"{self.names[0]}_matter_power.txt")
        self.valuesPlin = np.loadtxt(path/f"{self.names[1]}_matter_power.txt")
        self.valuesneff = np.loadtxt(path/f"{self.names[2]}_matter_power.txt")
        self.P2D = lambda z, k: eval_linear(self.grid, self.valuesP, np.log10(np.array([z, k])).T, xto.NEAREST)
        self.Plin2D = lambda z, k: eval_linear(self.grid, self.valuesPlin, np.log10(np.array([z, k])).T, xto.NEAREST)
        self.nefff = lambda z, k: eval_linear(self.grid, self.valuesneff, np.log10(np.array([z, k])).T, xto.NEAREST)

