import numpy as np
from evorob.world.robot.controllers.base import Controller


class SmoothedHebbianMLPController(Controller):
    """
    Warm-startable MLP controller with:
    1. evolved/base MLP weights
    2. evolved action smoothing
    3. evolved Hebbian learning rates

    Params:
        MLP weights: 296
        smoothing alphas: 8
        eta1: 27*8 = 216
        eta2: 8*8  = 64

        Total = 584
    """

    def __init__(
        self,
        input_size=27,
        output_size=8,
        hidden_size=8,
        eta_scale=0.002,
        weight_clip=5.0,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.eta_scale = eta_scale
        self.weight_clip = weight_clip

        self.W1_base = np.zeros((hidden_size, input_size), dtype=np.float32)
        self.b1_base = np.zeros(hidden_size, dtype=np.float32)
        self.W2_base = np.zeros((output_size, hidden_size), dtype=np.float32)
        self.b2_base = np.zeros(output_size, dtype=np.float32)

        self.W1 = self.W1_base.copy()
        self.b1 = self.b1_base.copy()
        self.W2 = self.W2_base.copy()
        self.b2 = self.b2_base.copy()

        self.smoothing_alphas = np.full(output_size, 0.7, dtype=np.float32)
        self.prev_action = np.zeros(output_size, dtype=np.float32)

        self.eta1 = np.zeros((hidden_size, input_size), dtype=np.float32)
        self.eta2 = np.zeros((output_size, hidden_size), dtype=np.float32)

        self.n_params = self.get_num_params()

    def get_num_params(self):
        mlp = (
            self.input_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.output_size
            + self.output_size
        )
        smoothing = self.output_size
        hebb = (
            self.input_size * self.hidden_size
            + self.hidden_size * self.output_size
        )
        return mlp + smoothing + hebb

    def reset_controller(self, batch_size=1):
        self.W1 = self.W1_base.copy()
        self.b1 = self.b1_base.copy()
        self.W2 = self.W2_base.copy()
        self.b2 = self.b2_base.copy()
        self.prev_action = np.zeros((batch_size, self.output_size), dtype=np.float32)

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float32)

        single = False
        if state.ndim == 1:
            state = state[None, :]
            single = True

        batch_size = state.shape[0]

        if self.prev_action.ndim == 1 or self.prev_action.shape[0] != batch_size:
            self.prev_action = np.zeros((batch_size, self.output_size), dtype=np.float32)

        hidden = np.tanh(state @ self.W1.T + self.b1)
        raw_action = np.tanh(hidden @ self.W2.T + self.b2)

        action = (
            self.smoothing_alphas[None, :] * self.prev_action
            + (1.0 - self.smoothing_alphas[None, :]) * raw_action
        )
        action = np.clip(action, -1.0, 1.0)

        # Hebbian update gated by action amplitude.
        # This is not a reward signal, only an activity/confidence gate.
        neuromod = np.mean(np.abs(action), axis=1)

        dW1 = np.einsum("b,bh,bi->hi", neuromod, hidden, state) / batch_size
        dW2 = np.einsum("b,bo,bh->oh", neuromod, action, hidden) / batch_size

        self.W1 += self.eta1 * dW1
        self.W2 += self.eta2 * dW2

        self.W1 = np.clip(self.W1, -self.weight_clip, self.weight_clip)
        self.W2 = np.clip(self.W2, -self.weight_clip, self.weight_clip)

        self.prev_action = action.copy()

        return action[0] if single else action

    def set_weights(self, weights):
        weights = np.asarray(weights, dtype=np.float32).ravel()
        expected = self.get_num_params()
        assert len(weights) == expected, (
            f"SmoothedHebbianMLPController expects {expected}, got {len(weights)}"
        )

        idx = 0

        n = self.input_size * self.hidden_size
        self.W1_base = weights[idx:idx+n].reshape(self.hidden_size, self.input_size)
        idx += n

        self.b1_base = weights[idx:idx+self.hidden_size].copy()
        idx += self.hidden_size

        n = self.hidden_size * self.output_size
        self.W2_base = weights[idx:idx+n].reshape(self.output_size, self.hidden_size)
        idx += n

        self.b2_base = weights[idx:idx+self.output_size].copy()
        idx += self.output_size

        raw_alphas = weights[idx:idx+self.output_size]
        idx += self.output_size

        self.smoothing_alphas = np.clip(
            (raw_alphas + 1.0) / 2.0 * 0.95,
            0.0,
            0.95,
        ).astype(np.float32)

        n = self.input_size * self.hidden_size
        self.eta1 = (
            weights[idx:idx+n].reshape(self.hidden_size, self.input_size)
            * self.eta_scale
        )
        idx += n

        n = self.hidden_size * self.output_size
        self.eta2 = (
            weights[idx:idx+n].reshape(self.output_size, self.hidden_size)
            * self.eta_scale
        )

        self.reset_controller(batch_size=1)

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_weights(self):
        raw_alphas = (self.smoothing_alphas / 0.95) * 2.0 - 1.0

        return np.concatenate([
            self.W1_base.ravel(),
            self.b1_base.ravel(),
            self.W2_base.ravel(),
            self.b2_base.ravel(),
            raw_alphas.ravel(),
            self.eta1.ravel() / self.eta_scale,
            self.eta2.ravel() / self.eta_scale,
        ]).astype(np.float32)