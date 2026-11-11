import numpy as np
from .gaussian import NdimGaussian
from .manifold import Euclidean

class VariableNode:
    def __init__(self,
                 variable_id,
                 dofs,
                 manifold=None,
                 initial_state=None):

        self.variableID = variable_id
        # Connected factor nodes
        self.adj_factors = []

        # Belief parameters initialised to zero
        self.manifold = Euclidean(dofs) if manifold is None else manifold
        self.mu = (np.zeros((dofs, 1)) if initial_state is None
                   else np.asarray(initial_state, dtype=float).copy())
        self.Sigma = np.zeros([dofs, dofs])


        self.belief = NdimGaussian(dofs, manifold=self.manifold, reference=self.mu)
        self.prior = NdimGaussian(dofs, manifold=self.manifold, reference=self.mu)
        
        self.prior_lambda_end = -1  # -1 flag if the sigma of self.prior is prior_sigma_end
        self.prior_lambda_logdiff = -1

        self.dofs = dofs

    @property
    def mean(self):
        """Current mean state; Euclidean means have shape (dofs, 1)."""
        return self.mu

    @property
    def cov(self):
        """Belief covariance in the tangent space of the current mean."""
        return self.Sigma

    def compute_message(self, target_factor=None, reference=None):
        """Combine the prior and incoming messages, excluding the recipient."""
        reference = self.mu if reference is None else reference
        message = self.prior.rebase(reference)
        for factor in self.adj_factors:
            if factor is target_factor:
                continue
            message_ix = factor.adj_vIDs.index(self.variableID)
            incoming = factor.messages[message_ix].rebase(reference)
            message.eta += incoming.eta
            message.lam += incoming.lam
        return message

    def update_belief(self):
        # Solve in the current chart, then move the belief to its new mean.
        belief = self.compute_message()
        delta = np.linalg.solve(belief.lam, belief.eta)
        self.mu = np.asarray(self.manifold.retract(belief.reference, delta.ravel()), dtype=float)
        self.belief = belief.rebase(self.mu)
        self.Sigma = np.linalg.solve(self.belief.lam, np.eye(self.dofs))

        # Publish independent belief snapshots for adjacent factors.
        for factor in self.adj_factors:
            belief_ix = factor.adj_vIDs.index(self.variableID)
            factor.adj_beliefs[belief_ix] = NdimGaussian(
                self.dofs, self.belief.eta.copy(), self.belief.lam.copy(),
                manifold=self.manifold, reference=self.mu,
            )
