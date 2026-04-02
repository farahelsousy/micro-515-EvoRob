import os
import sys
import numpy as np
import imageio
from pathlib import Path

# Add current directory to path so it can find your modules
sys.path.append(os.getcwd())

# macOS-friendly rendering
os.environ["MUJOCO_GL"] = "glfw"

# Corrected Imports based on your file structure
from evorob.world.envs.ant_flat import AntFlatEnvironment

# Import the actual controller class you defined in your main script
# (Assuming your main script is named 'Challenge2.py', change it if different)
try:
    from Challenge2 import CompatibleHybridAntController
except ImportError:
    print("Error: Could not import CompatibleHybridAntController from Challenge2.py")
    print("Make sure this script is in the same folder as Challenge2.py")
    sys.exit(1)

def generate_video(checkpoint_dir, generation=49):
    # Setup paths
    base_path = Path(checkpoint_dir) / str(generation)
    weights_path = base_path / "x.npy"
    fitness_path = base_path / "f.npy"
    
    if not weights_path.exists():
        print(f"Error: Could not find weights at {weights_path}")
        return

    # Load the population and fitnesses
    population = np.load(weights_path)
    fitnesses = np.load(fitness_path)

    # Pick the best for Ice (Objective 2)
    best_ice_idx = np.argmax(fitnesses[:, 1])
    best_weights = population[best_ice_idx]
    
    print(f"Recording best Ice performer (Fitness: {fitnesses[best_ice_idx, 1]:.2f})")

    # Initialize Environment - Use the ICE terrain XML
    env = AntFlatEnvironment(render_mode="rgb_array", robot_path="ant_ice_terrain.xml")
    
    # Initialize the specific controller from your training
    # It needs the input_size (27) and output_size (8)
    controller = CompatibleHybridAntController(input_size=27, output_size=8)
    controller.set_weights(best_weights)

    frames = []
    obs, _ = env.reset(seed=42)
    controller.reset_controller()
    
    done = False
    while not done and len(frames) < 1000:
        frames.append(env.render())
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # Save the file
    output_filename = f"ice_specialist_gen_{generation}.mp4"
    imageio.mimsave(output_filename, frames, fps=20)
    print(f"Video saved as {output_filename}")
    env.close()

if __name__ == "__main__":
    # Ensure this directory name matches your results folder
    MY_CHECKPOINT = "results/20260402_155038_nsga_ckpts"
    generate_video(MY_CHECKPOINT, generation=49)