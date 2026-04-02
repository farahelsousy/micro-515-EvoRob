import os
import sys
import numpy as np

# Ensure Python can find your modules
sys.path.append(os.getcwd())

from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.controllers.mlp import NeuralNetworkController

# ---------------------------------------------------------------------------
# HEBBIAN HYBRID CONTROLLER
# ---------------------------------------------------------------------------

class HebbianHybridAntController:
    def __init__(self, input_size=27, output_size=8):
        # 1. Frozen PPO base
        self.base_controller = NeuralNetworkController(
            input_size=input_size, output_size=output_size, hidden_size=[256, 256]
        )
        ppo_path = "results/ppo_ckpts/ppo_ant_10000000_steps.zip"
        if os.path.exists(ppo_path):
            from stable_baselines3 import PPO
            model = PPO.load(ppo_path, device="cpu")
            self.base_controller.load_from_ppo_model(model)

        # 2. Hebbian Residual (Inputs: 27 state + 16 phase = 43)
        self.residual_controller = HebbianController(
            input_size=input_size + (2 * output_size), 
            output_size=output_size, 
            hidden_size=16
        )
        
        self.dt = 0.01
        self.time_step = 0.0
        self.frequencies = np.ones(output_size)
        self.phases = np.zeros(output_size)

        # Genotype size for ABCD learning rules
        self.n_params = self.residual_controller.n_params 

    def reset_controller(self, batch_size=1):
        """CRITICAL: Initializes Hebbian weights (lin1 and output)"""
        self.time_step = 0.0
        self.residual_controller.reset_controller(batch_size)

    def get_action(self, state):
        base_action = self.base_controller.get_action(state)
        
        # Oscillator features
        t_phase = 2.0 * np.pi * self.frequencies * self.time_step + self.phases
        phase_feats = np.concatenate([np.sin(t_phase), np.cos(t_phase)])
        
        # Hebbian adaptation occurs inside this call
        aug_state = np.concatenate([state, phase_feats])
        residual_action = self.residual_controller.get_action(aug_state)
        
        action = base_action + 0.15 * residual_action
        self.time_step += self.dt
        return np.clip(action, -1.0, 1.0)

    def set_weights(self, genotype):
        self.residual_controller.geno2pheno(genotype)

# ---------------------------------------------------------------------------
# TESTING FUNCTION
# ---------------------------------------------------------------------------

def test_implementation():
    print("Testing Hebbian Controller...")
    controller = HebbianHybridAntController()
    
    # Must call reset_controller to avoid AttributeError 'lin1'
    controller.reset_controller() 
    
    print(f"Number of evolved parameters (ABCD rules): {controller.n_params}")
    
    dummy_state = np.zeros(27)
    action = controller.get_action(dummy_state)
    print(f"Action generated successfully: {action}")
    print("✅ System Ready for Evolution")

if __name__ == "__main__":
    test_implementation()