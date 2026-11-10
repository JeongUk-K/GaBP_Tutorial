import numpy as np


class NdimGaussian:
    """Gaussian in information form with eta shaped (dim, 1) and lam (dim, dim)."""

    def __init__(self, dimensionality, eta=None, lam=None):
        # dim
        self.dim = dimensionality

        
        if eta is not None and np.asarray(eta).size == self.dim:
            self.eta = np.asarray(eta, dtype=float).reshape(self.dim, 1)
        else:
            self.eta = np.zeros((self.dim, 1))

        if lam is not None and lam.shape == (self.dim, self.dim):
            self.lam = lam
        else:
            self.lam = np.zeros([self.dim, self.dim])
