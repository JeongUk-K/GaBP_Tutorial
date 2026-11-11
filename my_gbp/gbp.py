"""
    Defines classes for variable nodes, factor nodes and edges and factor graph.
"""

import numpy as np
import scipy.linalg

from .gaussian import NdimGaussian
from .autodiff import make_jac_fn
from .factor import Factor
from .variable_node import VariableNode
from .manifold import Euclidean
from .utils.save_history import save_history

from time import perf_counter


class FactorGraph:
    def __init__(self,
                 nonlinear_factors=True,
                 eta_damping=0.0, # Damping factor for messages from factors to variables. 0.0 means no damping, 1.0 means no update.
                 beta=None, # Relinearisation threshold; None means zero.
                 num_undamped_iters=None, # Undamped iterations after relinearisation; None means zero.
                 min_linear_iters=None): # Minimum iterations between relinearisations; None means zero.

        self.var_nodes = []
        self.factors = []
        self.history = []

        self.n_var_nodes = 0
        self.n_factor_nodes = 0
        self.n_edges = 0

        self.nonlinear_factors = nonlinear_factors

        self.eta_damping = eta_damping

        if nonlinear_factors:
            # For linearising nonlinear measurement factors.
            self.beta = 0.0 if beta is None else beta
            self.num_undamped_iters = 0 if num_undamped_iters is None else num_undamped_iters
            self.min_linear_iters = 0 if min_linear_iters is None else min_linear_iters

    def add_variable(self, dofs, prior_mean=None, prior_cov=None, weak_std=1e3, *,
                     manifold=None):
        """
            Declaratively add a variable node to the graph and return it.

            dofs: dimensionality of the variable.
            prior_mean: mean of the Gaussian prior (defaults to zeros). Also used as
                the initial estimate / linearisation point.
            prior_cov: covariance of the prior. If None a weak (near-uninformative)
                isotropic prior with std weak_std is used so the graph stays well posed.
            weak_std: std of the default weak prior when prior_cov is not given.
            manifold: geometry of the state; defaults to Euclidean(dofs).
        """
        manifold = Euclidean(dofs) if manifold is None else manifold
        if prior_mean is None and not isinstance(manifold, Euclidean):
            raise ValueError('prior_mean is required for a non-Euclidean manifold')
        mean = (np.zeros((dofs, 1)) if prior_mean is None
                else np.asarray(prior_mean, dtype=float).copy())
        if isinstance(manifold, Euclidean):
            mean = mean.reshape(dofs, 1)
        var_node = VariableNode(self.n_var_nodes, dofs, manifold=manifold,
                                initial_state=mean)
        if prior_cov is None:
            lam = np.eye(dofs) / (weak_std ** 2)
        else:
            lam = np.linalg.inv(np.asarray(prior_cov, dtype=float))

        # The fixed prior is centred at its own reference state.
        var_node.prior.eta = np.zeros((dofs, 1))
        var_node.prior.lam = lam
        var_node.mu = mean.copy()
        var_node.Sigma = (np.eye(dofs) * (weak_std ** 2) if prior_cov is None
                          else np.asarray(prior_cov, dtype=float))

        self.var_nodes.append(var_node)
        self.n_var_nodes += 1
        return var_node

    def add_factor(self, variables, meas_fn, measurement, noise_cov,
                   jacobian='numerical', jac_fn=None, loss=None,
                   mahalanobis_threshold=2, args=()):
        """
            Declaratively add a factor connecting the given variable nodes and return it.

            variables: list of VariableNode this factor is adjacent to. The measurement
                function receives the stacked values of these variables in order.
            meas_fn: measurement function h(x, *args) predicting the measurement.
            measurement: the observed value z (scalar or vector).
            noise_cov: covariance of the Gaussian measurement noise.
            jacobian: 'numerical' | 'auto'/'jax' | 'analytic'. See Factor / autodiff.
            jac_fn: analytic Jacobian, required when jacobian == 'analytic'.
            loss: robust loss ('huber', 'constant' or None).
            args: extra positional arguments forwarded to meas_fn / jac_fn.

            Edges to the adjacent variables are wired up automatically and the factor
            is linearised at the current variable estimates.
        """
        factor = Factor(self.n_factor_nodes, list(variables), measurement, noise_cov,
                        meas_fn, jac_fn, loss, mahalanobis_threshold, *args,
                        jacobian=jacobian)

        linpoint = tuple(v.mean.copy() for v in variables)
        factor.compute_factor(linpoint)

        for var_node in variables:
            var_node.adj_factors.append(factor)
        self.factors.append(factor)
        self.n_factor_nodes += 1
        self.n_edges += len(variables)
        return factor

    def converged_mean_change(self, prev_means, tol):
        """
            Whether the belief means have moved less than tol (L2) since prev_means.
        """
        if not isinstance(prev_means, tuple):
            return np.linalg.norm(self.get_means() - prev_means) < tol
        changes = [var.manifold.local(previous, var.mean)
                   for var, previous in zip(self.var_nodes, prev_means)]
        mean_change = np.linalg.norm(np.concatenate(changes) if changes else np.array([]))
        return mean_change < tol, mean_change

    def solve(self, max_iters=200, tol=1e-6, local_relin=True, robustify=False, monitoring=False, history=False,
              history_path=None):
        """
            Run synchronous GBP until the belief means stop changing (< tol) or
            max_iters is reached. Returns (converged, n_iters, history).

            Reset self.history on each call. If history=True, store copied means
            and covariances after each completed iteration, in var_nodes order.
            Iterations start at 1; the initial state is not included.
            With history=False, return an empty history list.
            With history=True, also write each iteration to history_path; the
            default is history.csv in the current working directory. Each solve
            starts a new CSV file, retaining completed iterations on later errors.
            Monitoring prints the mean step and factor-plus-prior energy.
        """
        elapsed_time = 0.0
        if monitoring:
            print('Solving factor graph with', self.n_var_nodes, 'variables,', self.n_factor_nodes, 'factors and', self.n_edges, 'edges.')

        self.history = []
        if history:
            variable_ids = [node.variableID for node in self.var_nodes]
            history_csv = save_history(None, variable_ids, history_path, append=False)
        self.update_all_beliefs()
        for i in range(max_iters):
            start_time = perf_counter()
            prev_means = self.get_states()
            self.synchronous_iteration(local_relin=local_relin, robustify=robustify)
            if history:
                self.history.append({
                    'iter': i + 1,
                    'means': [node.mean.copy() for node in self.var_nodes],
                    'covs': [node.cov.copy() for node in self.var_nodes],
                })
                save_history(self.history[-1], variable_ids, history_csv)
            flag, mean_change = self.converged_mean_change(prev_means, tol)
            if monitoring:
                energy = self.energy()
                elapsed_time += perf_counter() - start_time
                print(
                    'Iteration', i + 1, '/', max_iters,
                    '. Elapsed time:', f'{elapsed_time:.3f}', 'seconds.',
                    'Step norm:', f'{mean_change:.6f}',
                    'Energy:', f'{energy:.6f}', flush=True,
                )
            if flag:
                return True, i + 1, self.history
        return False, max_iters, self.history

    def energy(self):
        """Factor and prior quadratic costs, omitting fixed additive constants."""
        energy = 0.0
        for factor in self.factors:
            residual = np.asarray(factor.compute_residual()).reshape(-1)
            covariance = np.asarray(factor.adaptive_gauss_noise_var)
            energy += 0.5 * float(residual @ np.linalg.solve(covariance, residual))

        # Evaluate priors at their original reference, not the updated belief.
        for variable in self.var_nodes:
            prior = variable.prior
            delta = np.asarray(
                prior.manifold.local(prior.reference, variable.mean)
            ).reshape(-1)
            energy += (
                0.5 * float(delta @ prior.lam @ delta)
                - float(prior.eta.ravel() @ delta)
            )
        return energy

    def compute_all_messages(self, local_relin=True):
        # Freeze all incoming cavities before any outgoing message changes.
        incoming = [factor.prepare_messages() for factor in self.factors]
        for factor, messages in zip(self.factors, incoming):
            # If relinearisation is local then damping is also set locally per factor.
            if self.nonlinear_factors and local_relin:
                if factor.iters_since_relin == self.num_undamped_iters:
                    factor.eta_damping = self.eta_damping
                factor.compute_messages(factor.eta_damping, incoming_messages=messages)
            else:
                factor.compute_messages(self.eta_damping, incoming_messages=messages)

    def update_all_beliefs(self):
        for var_node in self.var_nodes:
            var_node.update_belief()

    def compute_all_factors(self):
        for factor in self.factors:
            factor.compute_factor()

    def relinearise_factors(self):
        """
            Compute the factor distribution for all factors for which the local belief mean has deviated a distance
            greater than beta from the current linearisation point.
            Relinearisation is only allowed at a maximum frequency of once every min_linear_iters iterations.
        """
        if self.nonlinear_factors:
            for factor in self.factors:
                states = tuple(var.mean.copy() for var in factor.adj_var_nodes)
                changes = np.concatenate([
                    var.manifold.local(reference, state)
                    for var, reference, state in zip(
                        factor.adj_var_nodes, factor._reference_states, states,
                    )
                ])
                if np.linalg.norm(changes) > self.beta and factor.iters_since_relin >= self.min_linear_iters:
                    factor.compute_factor(linpoint=states)
                    factor.iters_since_relin = 0
                    factor.eta_damping = 0.0
                else:
                    factor.iters_since_relin += 1

    def robustify_all_factors(self):
        for factor in self.factors:
            factor.robustify_loss()

    def synchronous_iteration(self, local_relin=True, robustify=False):
        if robustify:
            self.robustify_all_factors()
        if self.nonlinear_factors and local_relin:
            self.relinearise_factors()
        self.compute_all_messages(local_relin=local_relin)
        self.update_all_beliefs()

    # def joint_distribution_inf(self):
    #     """
    #         Get the joint distribution over all variables in the information form
    #         If nonlinear factors, it is taken at the current linearisation point.
    #     """

    #     eta = np.array([])
    #     lam = np.array([])
    #     var_ix = np.zeros(len(self.var_nodes)).astype(int)
    #     tot_n_vars = 0
    #     for var_node in self.var_nodes:
    #         var_ix[var_node.variableID] = int(tot_n_vars)
    #         tot_n_vars += var_node.dofs
    #         eta = np.concatenate((eta, var_node.prior.eta))
    #         if var_node.variableID == 0:
    #             lam = var_node.prior.lam
    #         else:
    #             lam = scipy.linalg.block_diag(lam, var_node.prior.lam)

    #     for factor in self.factors:
    #         factor_ix = 0
    #         for adj_var_node in factor.adj_var_nodes:
    #             vID = adj_var_node.variableID
    #             # Diagonal contribution of factor
    #             eta[var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
    #                 factor.factor.eta[factor_ix:factor_ix + adj_var_node.dofs]
    #             lam[var_ix[vID]:var_ix[vID] + adj_var_node.dofs, var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
    #                 factor.factor.lam[factor_ix:factor_ix + adj_var_node.dofs, factor_ix:factor_ix + adj_var_node.dofs]
    #             other_factor_ix = 0
    #             for other_adj_var_node in factor.adj_var_nodes:
    #                 if other_adj_var_node.variableID > adj_var_node.variableID:
    #                     other_vID = other_adj_var_node.variableID
    #                     # Off diagonal contributions of factor
    #                     lam[var_ix[vID]:var_ix[vID] + adj_var_node.dofs, var_ix[other_vID]:var_ix[other_vID] + other_adj_var_node.dofs] += \
    #                         factor.factor.lam[factor_ix:factor_ix + adj_var_node.dofs, other_factor_ix:other_factor_ix + other_adj_var_node.dofs]
    #                     lam[var_ix[other_vID]:var_ix[other_vID] + other_adj_var_node.dofs, var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
    #                         factor.factor.lam[other_factor_ix:other_factor_ix + other_adj_var_node.dofs, factor_ix:factor_ix + adj_var_node.dofs]
    #                 other_factor_ix += other_adj_var_node.dofs
    #             factor_ix += adj_var_node.dofs

    #     return eta, lam

    # def joint_distribution_cov(self):
    #     """
    #         Get the joint distribution over all variables in the covariance.
    #         If nonlinear factors, it is taken at the current linearisation point.
    #     """
    #     eta, lam = self.joint_distribution_inf()
    #     sigma = np.linalg.inv(lam)
    #     mu = sigma @ eta
    #     return mu, sigma

    def get_states(self):
        """Return independent state snapshots in variable order."""
        return tuple(var.mean.copy() for var in self.var_nodes)

    def get_means(self):
        """
            Return flat Euclidean means, or a state tuple for manifold graphs.
        """
        if any(not isinstance(var.manifold, Euclidean) for var in self.var_nodes):
            return self.get_states()
        mus = np.array([])
        for var_node in self.var_nodes:
            mus = np.concatenate((mus, var_node.mu.ravel()))
        return mus
