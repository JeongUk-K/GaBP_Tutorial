import numpy as np

from .manifold import Euclidean


class NdimGaussian:
    """Canonical Gaussian in the tangent coordinates of a reference state."""
    def __init__(self, dimensionality, eta=None, lam=None, *,
                 manifold=None, reference=None):
        self.dim = dimensionality
        self.manifold = Euclidean(self.dim) if manifold is None else manifold

        # The Euclidean origin preserves the existing absolute-coordinate API.
        if reference is None:
            if not isinstance(self.manifold, Euclidean):
                raise ValueError('reference is required for a non-Euclidean manifold')
            reference = np.zeros((self.dim, 1))
        self.reference = np.asarray(reference, dtype=float).copy()

        if eta is not None and np.asarray(eta).size == self.dim:
            self.eta = np.asarray(eta, dtype=float).reshape(self.dim, 1)
        else:
            self.eta = np.zeros((self.dim, 1))

        if lam is not None and lam.shape == (self.dim, self.dim):
            self.lam = lam
        else:
            self.lam = np.zeros([self.dim, self.dim])


    def rebase(self, reference):
        """Return a Gaussian in a new reference chart using a first-order map."""
        reference = np.asarray(reference, dtype=float)
        offset = np.asarray(
            self.manifold.local(self.reference, reference), dtype=float
        ).reshape(self.dim, 1)

        # Map new tangent coordinates into the original reference chart.
        _, local_jac = self.manifold.local_jacobians(self.reference, reference)
        _, retract_jac = self.manifold.retract_jacobians(reference, np.zeros(self.dim))
        jac = np.asarray(local_jac) @ np.asarray(retract_jac)

        eta = jac.T @ (self.eta - self.lam @ offset)
        lam = jac.T @ self.lam @ jac
        return NdimGaussian(self.dim, eta, lam,
                            manifold=self.manifold, reference=reference)
