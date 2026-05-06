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
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import (
    get_distinct_filename,
    get_last_checkpoint_dir,
    get_project_root,
)
from evorob.world.base import World
from evorob.world.robot.controllers.mlp_sol import NeuralNetworkController
from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.controllers.so2 import SO2Controller
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

"""
Morphology and Controller optimisation: Ant Hill
"""

ROOT_DIR = get_project_root()
ENV_NAME = "AntHill-v0"


def length_to_genotype(length: float) -> float:
    """Inverse of body mapping:
    length = (g + 1)/4 + 0.1  ->  g = 4*(length - 0.1) - 1
    """
    return 4 * (length - 0.1) - 1


class AntWorld(World):
    def __init__(self):
        action_space = 8
        state_space = 27

        self.controller = SO2Controller(
            input_size=state_space,
            output_size=action_space,
            hidden_size=action_space,
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
        reset_noise_scale=0.1,
        **kwargs,
    ):
        envs = SyncVectorEnv(
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

    def geno2pheno(self, genotype: np.ndarray):
        # Correct controller scaling:
        # - MLP: full weights
        # - Hebbian: scaled down
        # - SO2 / others: mildly scaled
        if isinstance(self.controller, NeuralNetworkController):
            control_weights = genotype[:self.n_weights]
        elif isinstance(self.controller, HebbianController):
            control_weights = genotype[:self.n_weights] * 0.05
        else:
            control_weights = genotype[:self.n_weights] * 0.1

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

    def evaluate_individual(self, genotype, n_repeats=2, n_steps=300):
        self.update_robot_xml(genotype)
        envs = self.create_env(n_envs=n_repeats, max_episode_steps=n_steps)
        self.controller.reset_controller(batch_size=n_repeats)

        rewards_full = np.zeros((n_steps, n_repeats))
        multi_obj_rewards_full = np.zeros((n_steps, n_repeats, 2))

        observations, info = envs.reset()
        done_mask = np.zeros(n_repeats, dtype=bool)

        for step in range(n_steps):
            actions = np.where(
                done_mask[:, None],
                0,
                self.controller.get_action(observations),
            )
            observations, rewards, dones, truncated, infos = envs.step(actions)

            rewards_full[step, ~done_mask] = rewards[~done_mask]

            multi_obj_reward = np.array([
                infos["reward_forward"]
    + infos["healthy_reward"]
    - 0.1 * np.abs(infos["y_velocity"])
    - 0.2 * np.maximum(0.0, -infos["x_velocity"]),
    -infos["ctrl_cost"] - infos["cfrc_cost"]
            ]).T
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


def run_EA_multi(ea, world):
    for gen in trange(ea.n_gen):
        pop = ea.ask()

        # mild diversity injection
        n_reset = max(1, int(0.02 * len(pop)))
        idx = np.random.choice(len(pop), size=n_reset, replace=False)
        pop[idx] = np.random.uniform(-1, 1, size=(n_reset, pop.shape[1]))

        fitnesses = np.empty((len(pop), 2))

        for i, genotype in enumerate(pop):
            _, fit = world.evaluate_individual(genotype)
            fitnesses[i] = fit

        ea.tell(pop, fitnesses, save_checkpoint=True)

        obj1 = fitnesses[:, 0]
        obj2 = fitnesses[:, 1]

        print(
            f"\nGen {gen:03d} | "
            f"Obj1 best: {np.max(obj1):.2f} | "
            f"Obj1 mean: {np.mean(obj1):.2f} || "
            f"Obj2 best: {np.max(obj2):.2f} | "
            f"Obj2 mean: {np.mean(obj2):.2f}"
        )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_episodes_hill(world, genotype, n_episodes, max_episode_steps, seed):
    world.update_robot_xml(genotype)
    env = gym.make(
        ENV_NAME,
        robot_path=world.world_file,
        max_episode_steps=max_episode_steps,
    )

    rng = np.random.default_rng(seed)
    episode_rewards, episode_obj1, episode_obj2 = [], [], []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        world.controller.reset_controller(batch_size=1)

        total_reward = total_obj1 = total_obj2 = 0.0
        for _ in range(max_episode_steps):
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_obj1 += (
                infos["reward_forward"]
    + infos["healthy_reward"]
    - 0.1 * np.abs(infos["y_velocity"])
    - 0.2 * np.maximum(0.0, -infos["x_velocity"]),
    -infos["ctrl_cost"]-infos["cfrc_cost"])
            total_obj2 += -float(info.get("ctrl_cost", 0.0))
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_obj1.append(total_obj1)
        episode_obj2.append(total_obj2)

    env.close()
    return episode_rewards, episode_obj1, episode_obj2


def _record_video_hill(world, genotype, max_steps, seed, out_path):
    try:
        world.update_robot_xml(genotype)
        env = gym.make(
            ENV_NAME,
            robot_path=world.world_file,
            max_episode_steps=max_steps,
            render_mode="rgb_array",
        )
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=seed)
        frames, video_reward = [], 0.0

        for _ in range(max_steps):
            frames.append(env.render())
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action)
            video_reward += reward
            if terminated or truncated:
                break

        env.close()
        imageio.mimwrite(out_path, frames, fps=20)
        print(f"  Video saved: {out_path}")
        return video_reward
    except Exception as e:
        print(f"  Warning: video skipped ({e})")
        return None


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "values": list(values),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "best": float(np.max(a)),
        "worst": float(np.min(a)),
        "median": float(np.median(a)),
    }


def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 256,
):
    max_episode_steps: int = 1000
    seed: int = 0

    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _try_load(filename):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = os.path.join(d, filename)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _try_load("x_best.npy")
    if x_best is None:
        print(f"ERROR: Could not find x_best.npy in '{checkpoint_dir}'.")
        return None

    population = _try_load("x.npy")
    fitness = _try_load("f.npy")
    print(f"Loaded x_best  (shape: {x_best.shape})")

    if population is not None and fitness is not None and fitness.ndim == 2 and fitness.shape[1] >= 2:
        spec1_idx = int(np.argmax(fitness[:, 0]))
        spec2_idx = int(np.argmax(fitness[:, 1]))
        gen_idx = int(np.argmax(np.sum(fitness, axis=1)))
        spec1_g, spec2_g, gen_g = population[spec1_idx], population[spec2_idx], population[gen_idx]
        print(f"Specialist obj1 (forward): idx={spec1_idx}  f={fitness[spec1_idx]}")
        print(f"Specialist obj2 (effic.) : idx={spec2_idx}  f={fitness[spec2_idx]}")
        print(f"Generalist (best sum)    : idx={gen_idx}    f={fitness[gen_idx]}")
    else:
        print("Warning: population/fitness not found — using x_best for all three roles.")
        spec1_g = spec2_g = gen_g = x_best

    world = AntWorld()
    controller_name = type(world.controller).__name__
    print(
        f"Controller: {controller_name}  |  params={world.controller.n_params}  |  genotype size={world.n_params}\n"
    )

    individuals = {
        "specialist_obj1": spec1_g,
        "specialist_obj2": spec2_g,
        "generalist": gen_g,
    }

    results = {}
    for label, genotype in individuals.items():
        print(f"Evaluating {label} ({n_episodes} episodes)...")
        rewards, obj1_vals, obj2_vals = _run_episodes_hill(
            world, genotype, n_episodes, max_episode_steps, seed
        )
        results[label] = {
            "reward": _stats(rewards),
            "obj1": _stats(obj1_vals),
            "obj2": _stats(obj2_vals),
        }
        r = results[label]
        print(
            f"  reward: {r['reward']['mean']:.2f} +/- {r['reward']['std']:.2f}  "
            f"obj1: {r['obj1']['mean']:.2f}  obj2: {r['obj2']['mean']:.2f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    video_names = {
        "specialist_obj1": "specialist_forward",
        "specialist_obj2": "specialist_efficiency",
        "generalist": "generalist",
    }
    for label, genotype in individuals.items():
        vpath = os.path.join(output_dir, f"evaluation_{video_names[label]}.mp4")
        _record_video_hill(world, genotype, max_episode_steps, seed, vpath)

    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Challenge 3 - Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Controller      : {controller_name} ({world.controller.n_params} params)\n")
        f.write(f"Genotype size   : {world.n_params}  (weights={world.controller.n_params}, body={world.n_body_params})\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/indiv. : {n_episodes}\n")
        f.write("Objectives      : [reward_forward+healthy_reward, -ctrl_cost]\n\n")

    print(f"\nScore saved to: {score_path}")
    return results


def main():
    # =========================================================
    # 1) NSGA-II with MLP + symmetric body
    # =========================================================
    world = AntWorld()
    world.controller = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=8,
    )
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params

    print("Number of parameters:", world.n_params)
    print("Number of weights:", world.n_weights)

    population_size = 100
    num_parents = 60
    num_generations = 200
    mutation_prob = 0.2
    crossover_prob = 0.7

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "nsga_only")

    ea_multi = NSGAII(
        population_size,
        world.n_params,
        num_parents,
        num_generations,
        (-1, 1),
        mutation_prob,
        crossover_prob,
    )
    ea_multi.directory_name = results_dir

    run_EA_multi(ea_multi, world)

    # =========================================================
    # 2) Visualize x_best
    # =========================================================
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"), allow_pickle=True)
    best_individual = np.asarray(best_individual).squeeze()

    print("x_best shape:", best_individual.shape)
    print("world.n_params:", world.n_params)
    print("world.n_weights:", world.n_weights)

    assert best_individual.shape[0] == world.n_params, (
        f"Mismatch: genotype has {best_individual.shape[0]} params "
        f"but world expects {world.n_params}"
    )

    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)

    video_name = get_distinct_filename(join(results_dir, "best_nsga.mp4"))
    print(f"Finished NSGA-II run, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name, n_steps=500)
    env.close()


if __name__ == "__main__":
    main()