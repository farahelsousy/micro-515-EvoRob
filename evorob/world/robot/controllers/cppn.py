import numpy as np
from evorob.world.robot.controllers.base import Controller


class HyperNEATController(Controller):
    def __init__(self, input_size=27, hidden_size=8, output_size=8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Small CPPN: 6 -> 16 -> 1
        self.cppn_in = 6
        self.cppn_hidden = 16
        self.cppn_out = 1

        self.n_cppn_params = (
            self.cppn_in * self.cppn_hidden
            + self.cppn_hidden
            + self.cppn_hidden * self.cppn_out
            + self.cppn_out
        )

        # Optional output biases for final MLP
        self.n_bias_params = hidden_size + output_size

        self.n_params = self.n_cppn_params + self.n_bias_params

        self.W1 = np.zeros((hidden_size, input_size))
        self.W2 = np.zeros((output_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)

    def get_num_params(self):
        return self.n_params

    def reset_controller(self, batch_size=1):
        pass

    def _cppn_forward(self, x):
        h = np.tanh(x @ self.C1 + self.cb1)
        y = np.tanh(h @ self.C2 + self.cb2)
        return y.squeeze(-1)

    def _make_coords(self):
        # Input neurons along x=-1
        input_coords = np.stack([
            -np.ones(self.input_size),
            np.linspace(-1, 1, self.input_size),
        ], axis=1)

        # Hidden neurons along x=0
        hidden_coords = np.stack([
            np.zeros(self.hidden_size),
            np.linspace(-1, 1, self.hidden_size),
        ], axis=1)

        # Output neurons arranged like 8 ant joints
        output_coords = np.array([
            [-1.0,  1.0],  # front-left hip
            [-0.7,  1.0],  # front-left knee
            [ 1.0,  1.0],  # front-right hip
            [ 0.7,  1.0],  # front-right knee
            [-1.0, -1.0],  # back-left hip
            [-0.7, -1.0],  # back-left knee
            [ 1.0, -1.0],  # back-right hip
            [ 0.7, -1.0],  # back-right knee
        ])

        return input_coords, hidden_coords, output_coords

    def _query_cppn_matrix(self, src, tgt):
        rows = []
        for tx, ty in tgt:
            row = []
            for sx, sy in src:
                d = np.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
                row.append([sx, sy, tx, ty, d, 1.0])
            rows.extend(row)

        cppn_inputs = np.asarray(rows, dtype=np.float32)
        weights = self._cppn_forward(cppn_inputs)
        return weights.reshape(len(tgt), len(src))

    def geno2pheno(self, genotype):
        g = np.asarray(genotype, dtype=np.float32).ravel()

        idx = 0

        n = self.cppn_in * self.cppn_hidden
        self.C1 = g[idx:idx+n].reshape(self.cppn_in, self.cppn_hidden)
        idx += n

        self.cb1 = g[idx:idx+self.cppn_hidden]
        idx += self.cppn_hidden

        n = self.cppn_hidden * self.cppn_out
        self.C2 = g[idx:idx+n].reshape(self.cppn_hidden, self.cppn_out)
        idx += n

        self.cb2 = g[idx:idx+self.cppn_out]
        idx += self.cppn_out

        input_coords, hidden_coords, output_coords = self._make_coords()

        self.W1 = self._query_cppn_matrix(input_coords, hidden_coords)
        self.W2 = self._query_cppn_matrix(hidden_coords, output_coords)

        self.b1 = g[idx:idx+self.hidden_size]
        idx += self.hidden_size

        self.b2 = g[idx:idx+self.output_size]

        # scale down for stability
        self.W1 *= 0.5
        self.W2 *= 0.5

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float32)

        if state.ndim == 1:
            hidden = np.tanh(self.W1 @ state + self.b1)
            action = np.tanh(self.W2 @ hidden + self.b2)
            return np.clip(action, -1.0, 1.0)

        hidden = np.tanh(state @ self.W1.T + self.b1)
        action = np.tanh(hidden @ self.W2.T + self.b2)
        return np.clip(action, -1.0, 1.0)