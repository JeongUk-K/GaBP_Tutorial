import numpy as np


class Euclidean:
    def __init__(self, dim, *, xp=np):
        self.tangent_dim = int(dim)
        # Use NumPy or jax.numpy operations.
        self._xp = xp

    def retract(self, state, delta):
        # Add the increment while preserving the state's shape.
        # add purturbation in the tangent space of the reference point
        state = self._xp.asarray(state, dtype=float)
        delta = self._xp.asarray(delta, dtype=float)
        return state + delta.reshape(state.shape)

    def local(self, reference, state):
        # Return the state difference as a flat tangent vector.
        # return purturbation in the tangent space of the reference point
        reference = self._xp.asarray(reference, dtype=float)
        state = self._xp.asarray(state, dtype=float)
        return state.reshape(self.tangent_dim) - reference.reshape(self.tangent_dim)

    def retract_jacobians(self, state, delta):
        # Jacobians with respect to state and delta: (I, I).
        identity = self._xp.eye(self.tangent_dim)
        return identity, identity.copy()

    def local_jacobians(self, reference, state):
        # Jacobians with respect to reference and state: (-I, I).
        identity = self._xp.eye(self.tangent_dim)
        return -identity, identity
