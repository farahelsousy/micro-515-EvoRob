import os
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory
from PIL import Image
import scipy.ndimage

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# TODO: set for cmaes
from evorob.algorithms.ea_api_sol import EvoAlgAPI
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import (
    get_distinct_filename,
    get_last_checkpoint_dir,
    get_project_root,
)
from evorob.world.base import World
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.controllers.so2 import SO2Controller
from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

"""
Morphology and Controller optimisation: Ant Hill
"""

ROOT_DIR = get_project_root()
ENV_NAME = "AntHill-v0"

MODEL_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_10000000_steps.zip"
STATS_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/ppo_ckpts/ppo_ant_vecnormalize_10000000_steps.pkl"


def infer_policy_architecture(model: PPO):
    hidden_sizes = []
    policy_net = model.policy.mlp_extractor.policy_net
    for layer in policy_net:
        if hasattr(layer, "weight") and hasattr(layer, "bias"):
            hidden_sizes.append(int(layer.weight.shape[0]))
    return hidden_sizes


class AntWorld(World):

    def __init__(self,):
        action_space = 8  # https://gymnasium.farama.org/environments/mujoco/ant/#action-space
        state_space = 27  # https://gymnasium.farama.org/environments/mujoco/ant/#observation-space

        self.controller = SO2Controller(
            input_size=state_space,
            output_size=action_space,
            hidden_size=action_space
        )

        self.n_weights = self.controller.n_params
        self.n_body_params = 8

        self.n_params = self.n_weights + self.n_body_params
        self.temp_dir = TemporaryDirectory()
        self.world_file = join(self.temp_dir.name, "AntHillEnv.xml")
        self.create_terrain_file("terrain.png")
        self.base_xml_path = join(ROOT_DIR, "evorob", "world", "robot", "assets", "hill_world.xml")

        self.joint_limits = [[-30, 30], [30, 70],
                        [-30, 30], [-70, -30],
                        [-30, 30], [-70, -30],
                        [-30, 30], [30, 70], ]
        self.joint_axis = [[0, 0, 1], [-1, 1, 0],
                      [0, 0, 1], [1, 1, 0],
                      [0, 0, 1], [-1, 1, 0],
                      [0, 0, 1], [1, 1, 0],
                      ]

        # whether to normalize observations with PPO VecNormalize stats
        self.use_vecnormalize = False
        self.vecnormalize_stats_path = None

    def update_robot_xml(self, genotype: np.ndarray):
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(points, connectivity_mat, self.joint_limits, self.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)

        #% Defining the Robot environment in MuJoCo
        world = xml.parse(self.base_xml_path)
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding="unicode")
        with open(self.world_file, "w") as f:
            f.write(world_xml)

    def create_env(self, render_mode: str = "rgb_array", n_envs: int = 1, max_episode_steps: int = 1000, reset_noise_scale=0.1, **kwargs):
        # for normal evolutionary runs: same structure as your template
        if not self.use_vecnormalize:
            envs = AsyncVectorEnv(
                [
                    lambda i_env=i_env: gym.make(
                        ENV_NAME,
                        robot_path=self.world_file,
                        reset_noise_scale=reset_noise_scale,
                        max_episode_steps=max_episode_steps,
                        render_mode=render_mode,
                    )
                    for i_env in range(n_envs)
                ]
            )
            return envs

        # for PPO-based controller evaluation: use SB3 VecNormalize wrapper
        def make_env():
            return gym.make(
                ENV_NAME,
                robot_path=self.world_file,
                reset_noise_scale=reset_noise_scale,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode,
            )

        envs = DummyVecEnv([make_env for _ in range(n_envs)])
        envs = VecNormalize.load(self.vecnormalize_stats_path, envs)
        envs.training = False
        envs.norm_reward = False
        return envs

    def geno2pheno(self, genotype):
        control_weights = genotype[:self.n_weights] * 0.1
        body_params = (genotype[self.n_weights:] + 1) / 4 + 0.1
        assert len(body_params) == self.n_body_params
        assert len(control_weights) == self.n_weights
        assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, back_left_leg, back_left_ankle, back_right_leg, back_right_ankle = body_params

        # Define the 3D coordinates of the relative tree structure
        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = np.array([np.sqrt(0.5 * front_left_leg ** 2), np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        front_left_toe_xyz = np.array([np.sqrt(0.5 * front_left_ankle ** 2), np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array([-np.sqrt(0.5 * front_right_leg ** 2), np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        front_right_toe_xyz = np.array([-np.sqrt(0.5 * front_right_ankle ** 2), np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = np.array([-np.sqrt(0.5 * back_left_leg ** 2), -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        back_left_toe_xyz = np.array([-np.sqrt(0.5 * back_left_ankle ** 2), -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = np.array([np.sqrt(0.5 * back_right_leg ** 2), -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        back_right_toe_xyz = np.array([np.sqrt(0.5 * back_right_ankle ** 2), -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        points = np.vstack([front_left_hip_xyz,
                            front_left_knee_xyz,
                            front_left_toe_xyz,
                            front_right_hip_xyz,
                            front_right_knee_xyz,
                            front_right_toe_xyz,
                            back_left_hip_xyz,
                            back_left_knee_xyz,
                            back_left_toe_xyz,
                            back_right_hip_xyz,
                            back_right_knee_xyz,
                            back_right_toe_xyz,
                            ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        )
        return points, connectivity_mat

    def create_terrain_file(self, filename="terrain.png", width=400, depth=400):
        # 1. Create the Slope (Gradient along X)
        # 0.0 at the back, 1.0 at the front
        # TODO: Change the terrain parameters
        slope_deg = 5.0
        bump_scale = 0.1
        sigma = 3.0

        # 1. Create Linear Slope (Gradient along X)
        rise = np.tan(np.deg2rad(slope_deg))
        slope_factor = rise
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        # Use tan to get actual height ratio, but clip to 1.0 to stay within hfield Z-bounds
        # (Assuming max height in XML is defined as the Z-scale)
        slope_map = X * slope_factor

        # 2. Add Bumps (Noise)
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        gentle_bump = np.tanh(X * 10)
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * gentle_bump
        noise_map = noise * bump_scale

        terrain = slope_map + noise_map
        terrain = np.clip(terrain, 0, 1)
        terrain[-1, -1] = 1
        terrain_normalized = (terrain * 255).astype(np.uint8)

        img = Image.fromarray(terrain_normalized, mode='L')
        save_path = os.path.join(self.temp_dir.name, filename)
        img.save(save_path)

    def evaluate_individual(self, genotype, n_repeats=10, n_steps=500):
        self.update_robot_xml(genotype)
        envs = self.create_env(n_envs=n_repeats, max_episode_steps=n_steps)
        self.controller.reset_controller(batch_size=n_repeats)

        rewards_full = np.zeros((n_steps, n_repeats))
        multi_obj_rewards_full = np.zeros((n_steps, n_repeats, 2))

        done_mask = np.zeros(n_repeats, dtype=bool)

        if self.use_vecnormalize:
            observations = envs.reset()
            print("obs shape:", observations.shape)
            print("obs[0][:10]:", observations[0][:10] if observations.ndim > 1 else observations[:10])
            info = None
            for step in range(n_steps):
                actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations))
                observations, rewards, dones, infos = envs.step(actions)

                rewards_full[step, ~done_mask] = rewards[~done_mask]

                if isinstance(infos, list):
                    z_velocity = np.array([info.get("z_velocity", 0.0) for info in infos], dtype=float)
                    ctrl_cost = np.array([info.get("ctrl_cost", 0.0) for info in infos], dtype=float)
                else:
                    z_velocity = np.array(infos.get("z_velocity", np.zeros(n_repeats)), dtype=float)
                    ctrl_cost = np.array(infos.get("ctrl_cost", np.zeros(n_repeats)), dtype=float)

                multi_obj_reward = np.array([z_velocity, -ctrl_cost]).T
                multi_obj_rewards_full[step, ~done_mask] = multi_obj_reward[~done_mask]

                done_mask = done_mask | dones
                if np.all(done_mask):
                    break
        else:
            observations, info = envs.reset()
            for step in range(n_steps):
                actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations))
                observations, rewards, dones, truncated, infos = envs.step(actions)

                rewards_full[step, ~done_mask] = rewards[~done_mask]

                multi_obj_reward = np.array([infos["z_velocity"], -infos["ctrl_cost"]]).T
                multi_obj_rewards_full[step, ~done_mask] = multi_obj_reward[~done_mask]

                done_mask = done_mask | dones | truncated
                if np.all(done_mask):
                    break

        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)


def run_EA_single(ea_single, world):
    for _ in trange(ea_single.n_gen):
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        for index, genotype in enumerate(pop):
            fit_ind, _ = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_single.tell(pop, fitnesses_gen, save_checkpoint=True)


def run_EA_multi(ea_multi, world):
    for _ in trange(ea_multi.n_gen):
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        for index, genotype in enumerate(pop):
            _, fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_multi.tell(pop, fitnesses_gen, save_checkpoint=True)


def main():
    #%% Optimise single-objective
    world = AntWorld()
    n_parameters = world.n_params

    #%% Understanding the world
    genotype = np.random.uniform(-1, 1, n_parameters).astype(np.float32)
    world.update_robot_xml(genotype)
    world.visualise_individual(genotype)

    # TODO Overwrite controller and load best run exercise 1
    model = PPO.load(MODEL_PATH, device="cpu")
    hidden_sizes = infer_policy_architecture(model)

    state_space = 27
    action_space = 8

    world.controller = NeuralNetworkController(
        input_size=state_space,
        output_size=action_space,
        hidden_size=hidden_sizes
    )
    world.controller.load_from_ppo_model(model)

    world.use_vecnormalize = True
    world.vecnormalize_stats_path = STATS_PATH

    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params

    genotype = np.random.uniform(-1, 1, world.n_params).astype(np.float32)

    flat_parts = []
    for W, b in zip(world.controller.weights, world.controller.biases):
        flat_parts.append(W.reshape(-1))
        flat_parts.append(b.reshape(-1))
    ppo_flat = np.concatenate(flat_parts).astype(np.float32)

    # because geno2pheno multiplies by 0.1
    genotype[:world.n_weights] = ppo_flat / 0.1

    # challenge-required lengths:
    genotype[-8::2] = -0.6   # upper legs = 0.2
    genotype[-7::2] = 0.2    # lower legs = 0.4

    world.update_robot_xml(genotype)
    world.visualise_individual(genotype)
    # %% Evolve open-loop so2
    world = AntWorld()
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    n_parameters = world.n_params
    population_size = 150
    opts = CMAES_opts.copy()
    opts["min"] = -1
    opts["max"] = 1
    opts["mutation_sigma"] = 0.3

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "single")
    ea_single = CMAES(population_size, n_parameters, opts, results_dir)

    run_EA_single(ea_single, world)

    #%% visualise
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"))
    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)
    video_name = get_distinct_filename(join(results_dir, "best.mp4"))
    print(f"Finished ES run, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name, n_steps=500)

    #%% Optimise multi-objective
    world = AntWorld()
    state_space = 27
    action_space = 8  # Change controller
    world.controller = NeuralNetworkController(
        input_size=state_space,
        output_size=action_space,
        hidden_size=action_space
    )
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    n_parameters = world.n_params
    print("Number of parameters:", n_parameters)
    print("Number of weights:", world.n_weights)
    population_size = 100

    opts = {}
    opts["min"] = -1
    opts["max"] = 1
    opts["num_parents"] = population_size // 2
    opts["num_generations"] = 50
    opts["mutation_prob"] = 0.2
    opts["crossover_prob"] = 0.5

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "multi")
    ea_multi_obj = NSGAII(population_size,
                          n_parameters,
                          opts["num_parents"],
                          opts["num_generations"],
                          (opts["min"], opts["max"]),
                          opts["mutation_prob"],
                          opts["crossover_prob"])
    ea_multi_obj.directory_name = results_dir
    run_EA_multi(ea_multi_obj, world)

    #%% visualise
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"), allow_pickle=True)
    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)
    video_name = get_distinct_filename(join(results_dir, "best.mp4"))
    print(f"Finished NSGAII run, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name, n_steps=500)


if __name__ == "__main__":
    main()