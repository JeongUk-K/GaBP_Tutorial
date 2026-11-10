import numpy as np
from .autodiff import make_jac_fn
from .gaussian import NdimGaussian

class Factor:
    def __init__(self,
                 factor_id,
                 adj_var_nodes,
                 measurement,
                 gauss_noise_cov,
                 meas_fn,
                 jac_fn=None,
                 loss=None,
                 mahalanobis_threshold=2,
                 *args,
                 jacobian=None):
        """
            n_stds: number of standard deviations from mean at which loss transitions to robust loss function.

            jac_fn: analytic Jacobian h'(x, *args). If given it is always used.
            jacobian: how to obtain the Jacobian when jac_fn is None. One of
                'numerical' (finite difference, default), 'auto'/'jax' (autodiff),
                or 'analytic' (which then requires jac_fn).
        """

        # Factor ID
        self.factorID = factor_id

        # ?
        self.dofs_conditional_vars = 0
        # Adjacent variable nodes
        self.adj_var_nodes = adj_var_nodes
        self.adj_vIDs = []
        self.adj_beliefs = []
        
        self.messages = []

        # stack up all adjacentn variable nodes
        for adj_var_node in self.adj_var_nodes:
            # sum of dofs of all adjacent variable nodes
            self.dofs_conditional_vars += adj_var_node.dofs
            self.adj_vIDs.append(adj_var_node.variableID)
            self.adj_beliefs.append(NdimGaussian(adj_var_node.dofs))
            self.messages.append(NdimGaussian(adj_var_node.dofs))

        self.factor = NdimGaussian(self.dofs_conditional_vars)
        self.linpoint = np.zeros((self.dofs_conditional_vars, 1))  # linearisation point

        self.measurement = np.asarray(measurement, dtype=float).reshape(-1, 1)

        # Measurement model
        self.gauss_noise_var = gauss_noise_cov
        # Measurement function
        self.meas_fn = meas_fn
        # analytic jacobian function
        if jac_fn is not None:
            # An explicit analytic Jacobian always takes precedence (backwards compatible).
            self.jac_fn = jac_fn
        else:
            self.jac_fn = make_jac_fn(jacobian or 'numerical', meas_fn)
        self.args = args

        # Robust loss function
        self.adaptive_gauss_noise_var = gauss_noise_cov
        self.loss = loss
        self.mahalanobis_threshold = mahalanobis_threshold
        self.robust_flag = False

        # Local relinearisation
        self.eta_damping = 0.
        self.iters_since_relin = 1

    # def compute_residual(self):
    #     """
    #         Calculate the reprojection error vector.
    #     """
    #     adj_belief_means = np.vstack([
    #         np.linalg.solve(belief.lam, belief.eta).reshape(belief.dim, 1)
    #         for belief in self.adj_beliefs
    #     ])
    #     pred = np.asarray(self.meas_fn(adj_belief_means.ravel(), *self.args)).reshape(-1, 1)
    #     d = pred - self.measurement
    #     return d

    # def energy(self):
    #     """
    #         Computes the squared error using the appropriate loss function.
    #     """
    #     return 0.5 * np.linalg.norm(self.compute_residual()) ** 2 / self.adaptive_gauss_noise_var

    def compute_factor(self, linpoint=None, update_self=True):
        """
            Compute the factor given the linearisation point.
            If not given then linearisation point is mean of belief of adjacent nodes.
            If measurement model is linear then factor will always be the same regardless of linearisation point.
            Internal vectors are columns; measurement/Jacobian callbacks receive a 1D state.
        """
        if linpoint is None:
            self.linpoint = np.vstack([
                np.linalg.solve(belief.lam, belief.eta).reshape(belief.dim, 1)
                for belief in self.adj_beliefs
            ])
        else:
            self.linpoint = np.asarray(linpoint, dtype=float).reshape(self.dofs_conditional_vars, 1)

        x_flat = self.linpoint.ravel()
        meas_dim = self.measurement.shape[0]
        # Jacobian of mesurement function
        J = np.asarray(self.jac_fn(x_flat, *self.args)).reshape(meas_dim, self.dofs_conditional_vars)
        pred_measurement = np.asarray(self.meas_fn(x_flat, *self.args)).reshape(meas_dim, 1)

        R = np.asarray(self.adaptive_gauss_noise_var, dtype=float)
        meas_model_lambda = np.linalg.solve(R, np.eye(meas_dim))
        lambda_factor = J.T @ meas_model_lambda @ J
        eta_factor = J.T @ meas_model_lambda @ (J @ self.linpoint + self.measurement - pred_measurement)

        if update_self:
            self.factor.eta, self.factor.lam = eta_factor, lambda_factor

        return eta_factor, lambda_factor

    # def robustify_loss(self):
    #     """
    #         Rescale the variance of the noise in the Gaussian measurement model if necessary and update the factor
    #         correspondingly.
    #     """
    #     old_adaptive_gauss_noise_var = self.adaptive_gauss_noise_var
    #     if self.loss is None:
    #         self.adaptive_gauss_noise_var = self.gauss_noise_var

    #     else:
    #         pred_measurement = np.asarray(
    #             self.meas_fn(self.linpoint.ravel(), *self.args)
    #         ).reshape(self.measurement.shape)

    #         if self.loss == 'huber':  # Loss is linear after Nstds from mean of measurement model
    #             mahalanobis_dist = np.linalg.norm(self.measurement - pred_measurement) / np.sqrt(self.gauss_noise_var)
    #             if mahalanobis_dist > self.mahalanobis_threshold:
    #                 self.adaptive_gauss_noise_var = self.gauss_noise_var * mahalanobis_dist**2 / \
    #                         (2*(self.mahalanobis_threshold * mahalanobis_dist - 0.5 * self.mahalanobis_threshold**2))
    #                 self.robust_flag = True
    #             else:
    #                 self.robust_flag = False
    #                 self.adaptive_gauss_noise_var = self.gauss_noise_var

    #         elif self.loss == 'constant':  # Loss is constant after Nstds from mean of measurement model
    #             mahalanobis_dist = np.linalg.norm(self.measurement - pred_measurement) / np.sqrt(self.gauss_noise_var)
    #             if mahalanobis_dist > self.mahalanobis_threshold:
    #                 self.adaptive_gauss_noise_var = mahalanobis_dist**2
    #                 self.robust_flag = True
    #             else:
    #                 self.robust_flag = False
    #                 self.adaptive_gauss_noise_var = self.gauss_noise_var

    #     # Update factor using existing linearisation point (we are not relinearising).
    #     self.factor.eta *= old_adaptive_gauss_noise_var / self.adaptive_gauss_noise_var
    #     self.factor.lam *= old_adaptive_gauss_noise_var / self.adaptive_gauss_noise_var

    def compute_messages(self, eta_damping):
        """
            Compute all outgoing messages from the factor.
        """
        messages_eta, messages_lam = [], []
        start_dim = 0
        
        for v in range(len(self.adj_vIDs)):
            eta_factor, lam_factor = self.factor.eta.copy(), self.factor.lam.copy()

            # Take product of factor with incoming messages
            mess_start_dim = 0
            for var in range(len(self.adj_vIDs)):
                if var != v:
                    var_dofs = self.adj_var_nodes[var].dofs
                    # adj node belief minus message to that node
                    eta_factor[mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[var].eta - self.messages[var].eta
                    lam_factor[mess_start_dim:mess_start_dim + var_dofs, mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[var].lam - self.messages[var].lam
                mess_start_dim += self.adj_var_nodes[var].dofs

            # Divide up parameters of distribution
            mess_dofs = self.adj_var_nodes[v].dofs
            eo = eta_factor[start_dim:start_dim + mess_dofs]
            eno = np.concatenate((eta_factor[:start_dim], eta_factor[start_dim + mess_dofs:]))
            # lambda
            # [loo, lono
            # lnoo, lnono]
            
            loo = lam_factor[start_dim:start_dim + mess_dofs, start_dim:start_dim + mess_dofs]
            lono = np.hstack((lam_factor[start_dim:start_dim + mess_dofs, :start_dim],
                              lam_factor[start_dim:start_dim + mess_dofs, start_dim + mess_dofs:]))
            # 
            # lnoo = np.vstack((lam_factor[:start_dim, start_dim:start_dim + mess_dofs],
            #                   lam_factor[start_dim + mess_dofs:, start_dim:start_dim + mess_dofs]))
            lnoo = lono.T
            lnono = np.block([[lam_factor[:start_dim, :start_dim], lam_factor[:start_dim, start_dim + mess_dofs:]],
                              [lam_factor[start_dim + mess_dofs:, :start_dim], lam_factor[start_dim + mess_dofs:, start_dim + mess_dofs:]]])

            # Compute outgoing messages
            messages_lam.append(loo - lono @ np.linalg.inv(lnono) @ lnoo)
            new_message_eta = eo - lono @ np.linalg.inv(lnono) @ eno
            messages_eta.append ((1 - eta_damping) * new_message_eta + eta_damping * self.messages[v].eta)
            start_dim += self.adj_var_nodes[v].dofs

        for v in range(len(self.adj_vIDs)):
            self.messages[v].lam = messages_lam[v]
            self.messages[v].eta = messages_eta[v]
