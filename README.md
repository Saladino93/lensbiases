# lensbiases
Calculating biases in CMB lensing analyses.

## Usage

You may want to install https://github.com/Saladino93/angularcls first.

### Bispectrum Calculations

* First, set up your own cosmology. This will be used to pre-calculate several quantities.
* Then, run `ib.generate_products(path, **pars)` using `from lensbiases import integrated_bispectrum as ib`.
* Once this is done, run `bispectrum_3D_numba.b3n.bispectrum_3D_numba(path = path, window = window)`, where `path` is where you stored the results of the previous step, and `window` as a generic window function (the default is `window=None`, giving the CMB lensing window).
