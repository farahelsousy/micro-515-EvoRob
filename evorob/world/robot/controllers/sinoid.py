import numpy as np
from evorob.world.robot.controllers.base import Controller


class OscillatoryController(Controller):
    """
    Oscillatory controller using sine waves with:
        - per-actuator amplitude
        - per-actuator offset
        - one shared global frequency
        - per-actuator phase

    Action for actuator i:
        a_i(t) = offset_i + amplitude_i * sin(2*pi*frequency*t + phase_i)

    Parameters:
        first output_size      -> amplitudes
        next output_size       -> offsets
        next 1                 -> shared frequency
        last output_size       -> phases

    Total params = 3 * output_size + 1
    """

    def __init__(self, output_size: int, dt: float = 0.01):
        self.output_size = int(output_size)
        self.dt = float(dt)
        self.time_step = 0.0
        self.n_params = self.get_num_params()

        # Default decoded parameters
        self.amplitudes = np.ones(self.output_size, dtype=np.float32) * 0.5
        self.offsets = np.zeros(self.output_size, dtype=np.float32)
        self.frequency = 1.0
        self.phases = np.zeros(self.output_size, dtype=np.float32)

    def reset_model(self):
        self.time_step = 0.0

    def reset_controller(self, batch_size=1):
        del batch_size
        self.time_step = 0.0

    def _compute_action_vector(self):
        phase = 2.0 * np.pi * self.frequency * self.time_step + self.phases
        actions = self.offsets + self.amplitudes * np.sin(phase)
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def get_action(self, state):
        """
        Supports both:
        - single observation: shape (obs_dim,) -> returns (output_size,)
        - batched observation: shape (batch_size, obs_dim) -> returns (batch_size, output_size)
        """
        action_vec = self._compute_action_vector()
        self.time_step += self.dt

        if state is None:
            return action_vec

        x = np.asarray(state)

        if x.ndim == 2:
            batch_size = x.shape[0]
            return np.tile(action_vec[None, :], (batch_size, 1))

        return action_vec

    def set_weights(self, weights):
        """
        Flat genotype format:
            first output_size      -> amplitudes
            next output_size       -> offsets
            next 1                 -> shared frequency
            last output_size       -> phases

        Raw parameters are decoded into bounded ranges.
        """
        weights = np.asarray(weights, dtype=np.float32).ravel()
        expected = self.get_num_params()
        if len(weights) != expected:
            raise ValueError(f"Expected {expected} params, got {len(weights)}")

        n = self.output_size

        raw_amp = weights[:n]
        raw_offset = weights[n:2 * n]
        raw_freq = weights[2 * n]
        raw_phase = weights[2 * n + 1:]

        # Amplitudes in [0, 1]
        self.amplitudes = 1.0 / (1.0 + np.exp(-raw_amp))
        self.amplitudes = self.amplitudes.astype(np.float32)

        # Offsets in [-0.75, 0.75]
        # Keeping offsets smaller than full action range avoids constant saturation
        self.offsets = 0.75 * np.tanh(raw_offset)
        self.offsets = self.offsets.astype(np.float32)

        # Shared frequency in [0.2, 2.5]
        self.frequency = float(0.2 + 2.3 * (1.0 / (1.0 + np.exp(-raw_freq))))

        # Phases in [-pi, pi]
        self.phases = (np.pi * np.tanh(raw_phase)).astype(np.float32)

    def get_weights(self):
        """
        Returns decoded controller parameters concatenated as a flat vector.
        Note:
        This is the decoded phenotype, not the original raw genotype.
        """
        return np.concatenate(
            [
                self.amplitudes,
                self.offsets,
                np.array([self.frequency], dtype=np.float32),
                self.phases,
            ]
        ).astype(np.float32)

    def get_num_params(self):
        return 3 * self.output_size + 1

    def geno2pheno(self, genotype):
        self.set_weights(genotype)