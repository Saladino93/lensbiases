import numpy as np

import sympy as sp


class LensingResponse(object):
    """
    Class to hold the lensing responses.

    For formulae see for example arXiv:1906.08760v2 Table 1.
    """

    def __init__(self):
        pass


    @staticmethod
    def fgeneral():
        pass
    
    @staticmethod
    def fTT(l1v, l2v, l1n, l2n, tfunction, dotoperation):
        """
        Function to compute the lensing response for a TT quadratic estimator.
        """
        Lv = l1v+l2v
        return dotoperation(Lv, l1v)*tfunction(l1n)+dotoperation(Lv, l2v)*tfunction(l2n)
    
    @staticmethod
    def fTE(l1v, l2v, l1n, l2n, cos12, tefunction, dotoperation):
        """
        Function to compute the lensing response for a TE quadratic estimator.
        """
        Lv = l1v+l2v
        return cos12*dotoperation(Lv, l1v)*tefunction(l1n)+dotoperation(Lv, l2v)*tefunction(l2n)

    @staticmethod
    def fEE(l1v, l2v, l1n, l2n, cos12, efunction, dotoperation):
        """
        Function to compute the lensing response for a EE quadratic estimator.

        Parameters
        ----------
        l1v : array
            First lensing vector.
        l2v : array
            Second lensing vector.
        l1n : array
            First lensing modulus.
        l2n : array
            Second lensing modulus.
        cos12 : array
            Cosine of the double of the angle between the two lensing vectors.
        efunction : function
            Function to compute the lensing response for a EE quadratic estimator.

        Returns
        -------
        array
            Lensing response for a EE quadratic estimator.
        """
        Lv = l1v+l2v
        return (dotoperation(Lv, l1v)*efunction(l1n)+dotoperation(Lv, l2v)*efunction(l2n))*cos12
    
    @staticmethod
    def fEB(l1v, l2v, l1n, l2n, cos12, tefunction, dotoperation):
        """
        Function to compute the lensing response for a TE quadratic estimator.
        """
        Lv = l1v+l2v
        return cos12*dotoperation(Lv, l1v)*tefunction(l1n)+dotoperation(Lv, l2v)*tefunction(l2n)


    @staticmethod
    def ftot(l, totfunction):
        """
        Function to return the total power spectrum, signal+noise for given ells.
        """
        return totfunction(l)
    
