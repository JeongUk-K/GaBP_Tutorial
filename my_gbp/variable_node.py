import numpy as np
from .gaussian import NdimGaussian

class VariableNode:
    def __init__(self,
                 variable_id,
                 dofs):

        self.variableID = variable_id
        # Connected factor nodes
        self.adj_factors = []

        # Belief parameters initialised to zero
        self.mu = np.zeros((dofs, 1))
        self.Sigma = np.zeros([dofs, dofs])


        self.belief = NdimGaussian(dofs)
        self.prior = NdimGaussian(dofs)
        
        self.prior_lambda_end = -1  # -1 flag if the sigma of self.prior is prior_sigma_end
        self.prior_lambda_logdiff = -1

        self.dofs = dofs

    @property
    def mean(self):
        """Current estimate of the belief mean, shaped (dofs, 1)."""
        return self.mu

    @property
    def cov(self):
        """Current estimate of the belief covariance."""
        return self.Sigma

    def update_belief(self):
        """
            Update local belief estimate by taking product of all incoming messages along all edges.
            Then send belief to adjacent factor nodes.
        """
        # Update local belief
        eta = np.asarray(self.prior.eta, dtype=float).reshape(self.dofs, 1).copy()
        lam = self.prior.lam.copy()
        
        # receive factor messages
        for factor in self.adj_factors:
            message_ix = factor.adj_vIDs.index(self.variableID)
            eta_inward, lam_inward = factor.messages[message_ix].eta, factor.messages[message_ix].lam
            eta += np.asarray(eta_inward).reshape(self.dofs, 1)
            lam += lam_inward

        self.belief.eta = eta 
        self.belief.lam = lam
        
        # Update mean and covariance of belief
        self.Sigma = np.linalg.inv(self.belief.lam)
        self.mu = self.Sigma @ self.belief.eta
        
        # Send belief to adjacent factors
        for factor in self.adj_factors:
            belief_ix = factor.adj_vIDs.index(self.variableID)
            factor.adj_beliefs[belief_ix].eta, factor.adj_beliefs[belief_ix].lam = self.belief.eta, self.belief.lam
