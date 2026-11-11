import numpy as np
from .autodiff import make_jac_fn
from .gaussian import NdimGaussian
from .manifold import Euclidean


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

            jac_fn: analytic dh/dx for flat Euclidean callbacks, or de/d(delta)
                for tuple-state callbacks with e = h - z. If given it is always used.
            jacobian: how to obtain the Jacobian when jac_fn is None. One of
                'numerical' (finite difference, default), 'auto'/'jax' (autodiff),
                or 'analytic' (which then requires jac_fn). Numerical and AD
                differentiate tangent increments applied through retract.
        """

        # Factor ID
        self.factorID = factor_id

        # ?
        self.dofs_conditional_vars = 0
        # Adjacent variable nodes
        self.adj_var_nodes = adj_var_nodes
        self._flat_input = all(isinstance(v.manifold, Euclidean) for v in adj_var_nodes)
        self.adj_vIDs = []
        self.adj_beliefs = []
        
        self.messages = []

        # stack up all adjacentn variable nodes
        for adj_var_node in self.adj_var_nodes:
            # sum of dofs of all adjacent variable nodes
            self.dofs_conditional_vars += adj_var_node.dofs
            self.adj_vIDs.append(adj_var_node.variableID)
            self.adj_beliefs.append(NdimGaussian(
                adj_var_node.dofs, manifold=adj_var_node.manifold,
                reference=adj_var_node.mean,
            ))
            self.messages.append(NdimGaussian(
                adj_var_node.dofs, manifold=adj_var_node.manifold,
                reference=adj_var_node.mean,
            ))

        self.factor = NdimGaussian(self.dofs_conditional_vars)
        self._reference_states = tuple(v.mean.copy() for v in self.adj_var_nodes)
        self.linpoint = self._pack_states(self._reference_states)

        self.measurement = np.asarray(measurement, dtype=float).reshape(-1, 1)

        # Measurement model
        self.gauss_noise_var = gauss_noise_cov
        # Measurement function
        self.meas_fn = meas_fn
        # analytic jacobian function
        self._analytic_jacobian = jac_fn is not None
        self._xp = np
        if jac_fn is not None:
            # An explicit analytic Jacobian always takes precedence (backwards compatible).
            self.jac_fn = jac_fn
        else:
            mode = jacobian or 'numerical'
            self.jac_fn = make_jac_fn(mode, self._residual_delta)
            if mode in ('auto', 'jax'):
                import jax.numpy as jnp
                self._xp = jnp
        self.args = args
        self._manifolds = tuple(
            Euclidean(v.dofs, xp=self._xp) if isinstance(v.manifold, Euclidean)
            else v.manifold for v in self.adj_var_nodes
        )

        # Robust loss function
        self.adaptive_gauss_noise_var = gauss_noise_cov
        self.loss = loss
        self.mahalanobis_threshold = mahalanobis_threshold
        self.robust_flag = False

        # Local relinearisation
        self.eta_damping = 0.
        self.iters_since_relin = 1

    def compute_residual(self):
        """Evaluate h - z at the current variable means."""
        states = self._get_states()
        delta = np.zeros(self.dofs_conditional_vars)
        return np.asarray(self._residual_delta(delta, states)).reshape(-1, 1)

    # def energy(self):
    #     """
    #         Computes the squared error using the appropriate loss function.
    #     """
    #     return 0.5 * np.linalg.norm(self.compute_residual()) ** 2 / self.adaptive_gauss_noise_var

    # Pack states into a flat array or tuple
    def _pack_states(self, states):
        if self._flat_input:
            return np.concatenate([state.ravel() for state in states]).reshape(-1, 1)
        return tuple(state.copy() for state in states)

    # Get the states of the adjacent variable nodes
    def _get_states(self, linpoint=None):
        if linpoint is None:
            return tuple(v.mean.copy() for v in self.adj_var_nodes)
        if isinstance(linpoint, tuple):
            return tuple(np.asarray(state, dtype=float).copy() for state in linpoint)
        flat = np.asarray(linpoint, dtype=float).ravel()
        states, start = [], 0
        for var in self.adj_var_nodes:
            states.append(flat[start:start + var.dofs].reshape(var.mean.shape).copy())
            start += var.dofs
        return tuple(states)

    # perturbed residual function for computing Jacobian
    def _residual_delta(self, delta, reference_states):
        states, start = [], 0
        for var, manifold, reference in zip(self.adj_var_nodes, self._manifolds, reference_states):
            states.append(manifold.retract(
                self._xp.asarray(reference), delta[start:start + var.dofs],
            ))
            start += var.dofs
        callback_input = (self._xp.concatenate([state.ravel() for state in states])
                          if self._flat_input else tuple(states))
        prediction = self._xp.asarray(self.meas_fn(callback_input, *self.args)).reshape(-1)
        return prediction - self._xp.asarray(self.measurement).reshape(-1)

    def compute_factor(self, linpoint=None, update_self=True):
        """Linearise h - z in tangent increments around per-variable states.

        Legacy Euclidean analytic callbacks return dh/dx on a flat state.
        Other analytic callbacks receive a tuple and return de/d(delta).
        """
        states = self._get_states(linpoint)
        zero = np.zeros(self.dofs_conditional_vars)
        meas_dim = self.measurement.shape[0]
        if self._analytic_jacobian:
            callback_input = self._pack_states(states)
            if self._flat_input:
                callback_input = callback_input.ravel()
            J = np.asarray(self.jac_fn(callback_input, *self.args)).reshape(meas_dim, -1)
            if self._flat_input:
                blocks = [var.manifold.retract_jacobians(state, np.zeros(var.dofs))[1]
                          for var, state in zip(self.adj_var_nodes, states)]
                tangent_map = np.zeros((self.dofs_conditional_vars, self.dofs_conditional_vars))
                start = 0
                for var, block in zip(self.adj_var_nodes, blocks):
                    tangent_map[start:start + var.dofs, start:start + var.dofs] = block
                    start += var.dofs
                J = J @ tangent_map
        else:
            J = np.asarray(self.jac_fn(zero, states)).reshape(meas_dim, self.dofs_conditional_vars)
        residual = np.asarray(self._residual_delta(zero, states)).reshape(meas_dim, 1)

        R = np.asarray(self.adaptive_gauss_noise_var, dtype=float)
        meas_model_lambda = np.linalg.solve(R, np.eye(meas_dim))
        # purturbed coordinates of the measurement model in tangent space
        lambda_factor = J.T @ meas_model_lambda @ J
        eta_factor = -J.T @ meas_model_lambda @ residual

        if update_self:
            self._reference_states = states
            self.linpoint = self._pack_states(states)
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

    # prepare messages for all adjacent variable nodes
    # except the one that is factor passed variable node
    def prepare_messages(self):
        """Snapshot variable cavities in the factor's linearisation charts."""
        return [var.compute_message(self, reference=reference)
                for var, reference in zip(self.adj_var_nodes, self._reference_states)]


    def compute_messages(self, eta_damping, incoming_messages=None):
        """Marginalise in tangent coordinates and attach each output reference."""
        if incoming_messages is None:
            incoming_messages = self.prepare_messages()
        messages_eta, messages_lam = [], []
        start_dim = 0
        
        for v in range(len(self.adj_vIDs)):
            eta_factor, lam_factor = self.factor.eta.copy(), self.factor.lam.copy()

            # Take product of factor with incoming messages
            mess_start_dim = 0
            for var in range(len(self.adj_vIDs)):
                if var != v:
                    var_dofs = self.adj_var_nodes[var].dofs
                    # The incoming cavity already excludes this factor.
                    eta_factor[mess_start_dim:mess_start_dim + var_dofs] += incoming_messages[var].eta
                    lam_factor[mess_start_dim:mess_start_dim + var_dofs, mess_start_dim:mess_start_dim + var_dofs] += incoming_messages[var].lam
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
            old_message = self.messages[v].rebase(self._reference_states[v])
            messages_eta.append((1 - eta_damping) * new_message_eta + eta_damping * old_message.eta)
            start_dim += self.adj_var_nodes[v].dofs

        for v in range(len(self.adj_vIDs)):
            var = self.adj_var_nodes[v]
            self.messages[v] = NdimGaussian(
                var.dofs, messages_eta[v], messages_lam[v],
                manifold=var.manifold, reference=self._reference_states[v],
            )
