import os
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory

import gymnasium as gym
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from gymnasium.vector import SyncVectorEnv
from tqdm import trange

from evorob.algorithms.ea_api_sol import EvoAlgAPI
from evorob.utils.filesys import (
    get_distinct_filename,
    get_last_checkpoint_dir,
    get_project_root,
)
from evorob.world.base import World
from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
ENV_NAME = "AntHill-v0"

# =========================================================
# CHANGE THIS TO YOUR ACTUAL STAGE-1 x_best.npy PATH
# =========================================================
STAGE1_BEST_PATH = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/AntHill-v0/hebbian_only copy/35/x_best.npy"


def length_to_genotype(length: float) -> float:
    """Inverse of body mapping:
    length = (g + 1)/4 + 0.1  ->  g = 4*(length - 0.1) - 1
    """
    return 4 * (length - 0.1) - 1


class AntWorld(World):
    def __init__(self):
        action_space = 8
        state_space = 27

        self.controller = HebbianController(
            input_size=state_space,
            output_size=action_space,
            hidden_size=4,
        )

        self.n_weights = self.controller.n_params
        self.n_body_params = 4
        self.n_params = self.n_weights + self.n_body_params

        self.temp_dir = TemporaryDirectory()
        self.world_file = join(self.temp_dir.name, "AntHillEnv.xml")
        self.create_terrain_file("terrain.png")
        self.base_xml_path = join(
            ROOT_DIR, "evorob", "world", "robot", "assets", "hill_world.xml"
        )

        self.joint_limits = [
            [-30, 30], [30, 70],
            [-30, 30], [-70, -30],
            [-30, 30], [-70, -30],
            [-30, 30], [30, 70],
        ]
        self.joint_axis = [
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
        ]

    def update_robot_xml(self, genotype: np.ndarray):
        points, connectivity_mat = self.geno2pheno(genotype)

        robot = AntRobot(
            points,
            connectivity_mat,
            self.joint_limits,
            self.joint_axis,
            verbose=False,
        )
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)

        world = xml.parse(self.base_xml_path)
        robot_env = world.getroot()
        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding="unicode")

        with open(self.world_file, "w") as f:
            f.write(world_xml)

    def create_env(
        self,
        render_mode: str = "rgb_array",
        n_envs: int = 1,
        max_episode_steps: int = 1000,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        return SyncVectorEnv(
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

    def geno2pheno(self, genotype: np.ndarray):
        # Hebbian controller genes are scaled down for stability
        control_weights = genotype[:self.n_weights] * 0.05

        # 4 symmetric body genes: [front_leg, front_ankle, back_leg, back_ankle]
        body_params = (genotype[self.n_weights:] + 1) / 4 + 0.1

        assert len(body_params) == self.n_body_params
        assert len(control_weights) == self.n_weights
        assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        front_leg, front_ankle, back_leg, back_ankle = body_params

    def geno2pheno(self, genotype: np.ndarray):
        # Hebbian controller genes are scaled down for stability
        control_weights = genotype[:self.n_weights] * 0.05

        # 4 symmetric body genes: [front_leg, front_ankle, back_leg, back_ankle]
        body_params = (genotype[self.n_weights:] + 1) / 4 + 0.1

        assert len(body_params) == self.n_body_params
        assert len(control_weights) == self.n_weights
        assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        front_leg, front_ankle, back_leg, back_ankle = body_params

        # left-right symmetry
        front_left_leg = front_leg
        front_left_ankle = front_ankle
        front_right_leg = front_leg
        front_right_ankle = front_ankle

        back_left_leg = back_leg
        back_left_ankle = back_ankle
        back_right_leg = back_leg
        back_right_ankle = back_ankle

        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = (
            np.array([
                np.sqrt(0.5 * front_left_leg**2),
                np.sqrt(0.5 * front_left_leg**2),
                0,
            ]) + front_left_hip_xyz
        )
        front_left_toe_xyz = (
            np.array([
                np.sqrt(0.5 * front_left_ankle**2),
                np.sqrt(0.5 * front_left_ankle**2),
                0,
            ]) + front_left_knee_xyz
        )

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = (
            np.array([
                -np.sqrt(0.5 * front_right_leg**2),
                np.sqrt(0.5 * front_right_leg**2),
                0,
            ]) + front_right_hip_xyz
        )
        front_right_toe_xyz = (
            np.array([
                -np.sqrt(0.5 * front_right_ankle**2),
                np.sqrt(0.5 * front_right_ankle**2),
                0,
            ]) + front_right_knee_xyz
        )

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = (
            np.array([
                -np.sqrt(0.5 * back_left_leg**2),
                -np.sqrt(0.5 * back_left_leg**2),
                0,
            ]) + back_left_hip_xyz
        )
        back_left_toe_xyz = (
            np.array([
                -np.sqrt(0.5 * back_left_ankle**2),
                -np.sqrt(0.5 * back_left_ankle**2),
                0,
            ]) + back_left_knee_xyz
        )

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = (
            np.array([
                np.sqrt(0.5 * back_right_leg**2),
                -np.sqrt(0.5 * back_right_leg**2),
                0,
            ]) + back_right_hip_xyz
        )
        back_right_toe_xyz = (
            np.array([
                np.sqrt(0.5 * back_right_ankle**2),
                -np.sqrt(0.5 * back_right_ankle**2),
                0,
            ]) + back_right_knee_xyz
        )

        points = np.vstack([
            front_left_hip_xyz,
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

        connectivity_mat = np.array([
            [150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        return points, connectivity_mat

    def create_terrain_file(self, filename="terrain.png", width=400, depth=400):
        slope_deg = 5.0
        bump_scale = 0.1
        sigma = 3.0

        rise = np.tan(np.deg2rad(slope_deg))
        slope_factor = rise
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        slope_map = X * slope_factor

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

        img = Image.fromarray(terrain_normalized, mode="L")
        save_path = os.path.join(self.temp_dir.name, filename)
        img.save(save_path)

    def evaluate_individual(self, genotype, n_repeats=2,n_steps=200):
        self.update_robot_xml(genotype)
        envs = self.create_env(n_envs=n_repeats, max_episode_steps=n_steps)
        self.controller.reset_controller(batch_size=n_repeats)

        rewards_full = np.zeros((n_steps, n_repeats))
        observations, _ = envs.reset()
        done_mask = np.zeros(n_repeats, dtype=bool)

        for step in range(n_steps):
            actions = np.where(
                done_mask[:, None],
                0,
                self.controller.get_action(observations),
            )
            observations, rewards, dones, truncated, infos = envs.step(actions)

            rewards_full[step, ~done_mask] = rewards[~done_mask]
            done_mask = done_mask | dones | truncated

            if np.all(done_mask):
                break

        final_rewards = np.sum(rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), None


def run_EA_single(ea, world, injected_best=None, noise_std=0.02, n_inject=8):
    best_so_far = -np.inf

    for gen in trange(ea.n_gen):
        pop = ea.ask()

        # Inject FULL genotype, not just controller weights
        if gen == 0 and injected_best is not None:
            pop[0] = injected_best.copy()

            n_extra = min(n_inject, len(pop) - 1)
            for i in range(1, n_extra + 1):
                candidate = injected_best + np.random.normal(0, noise_std, size=injected_best.shape)
                candidate = np.clip(candidate, -1, 1)
                pop[i] = candidate

            print("Injected full Stage-1 genotype into generation 0.")

        fitnesses = np.empty(len(pop))

        for i, genotype in enumerate(pop):
            fit, _ = world.evaluate_individual(genotype)
            fitnesses[i] = fit

        ea.tell(pop, fitnesses, save_checkpoint=True)

        gen_best = np.max(fitnesses)
        gen_mean = np.mean(fitnesses)
        best_so_far = max(best_so_far, gen_best)

        print(
            f"\nGen {gen:03d} | "
            f"Best: {gen_best:.2f} | "
            f"Mean: {gen_mean:.2f} | "
            f"Best so far: {best_so_far:.2f}"
        )


def main():
    world = AntWorld()

    print("Hebbian weights:", world.n_weights)
    print("Total search dim:", world.n_params)

    if not os.path.isfile(STAGE1_BEST_PATH):
        raise FileNotFoundError(f"Could not find Stage-1 best file:\n{STAGE1_BEST_PATH}")

    best_stage1 = np.load(STAGE1_BEST_PATH)
    best_stage1 = np.asarray(best_stage1).squeeze()

    print("Loaded from:", STAGE1_BEST_PATH)
    print("Loaded shape:", best_stage1.shape)
    print("Expected params:", world.n_params)

    if best_stage1.shape[0] != world.n_params:
        raise ValueError(
            f"Stage-1 best has shape {best_stage1.shape}, "
            f"but current world expects ({world.n_params},). "
            f"Make sure Stage 1 used Hebbian hidden_size=4 and n_body_params=4."
        )

    # Evaluate loaded Stage-1 best directly before refinement
    direct_fit, _ = world.evaluate_individual(best_stage1)
    print("Direct evaluation of loaded Stage-1 best:", direct_fit)

    # Optional visualization of the loaded best
    world.update_robot_xml(best_stage1)
    world.visualise_individual(best_stage1)

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "hebbian_refine")

    ea_single = EvoAlgAPI(
        n_params=world.n_params,
        population_size=130,
        num_generations=100,
        sigma=0.1,
        bounds=(-1, 1),
        output_dir=results_dir,
    )

    run_EA_single(
        ea_single,
        world,
        injected_best=best_stage1,
        noise_std=0.02,
        n_inject=8,
    )

    checkpoint = get_last_checkpoint_dir(results_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {results_dir}")

    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"))
    best_individual = np.asarray(best_individual).squeeze()

    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)

    video_name = get_distinct_filename(join(results_dir, "best_hebbian_refined.mp4"))
    print(f"Finished Hebbian refinement, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name,n_steps=200)
    env.close()


if __name__ == "__main__":
    main()