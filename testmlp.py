import os
import json
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory

import gymnasium as gym
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image

from evorob.utils.filesys import get_project_root
from evorob.world.base import World
from evorob.world.robot.controllers.mlp_sol import NeuralNetworkController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
ENV_NAME = "AntHill-v0"

# ============================================================
# CHANGE THIS TO YOUR CHECKPOINT FOLDER
# Example:
# /Users/.../results/AntHill-v0/single_mlp_1000/100
# ============================================================
RESULTS_DIR = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/AntHill-v0/single_mlp copy/100/"


def decode_body(genotype: np.ndarray, n_body_params: int) -> dict:
    body = (genotype[-n_body_params:] + 1) / 4 + 0.1

    if n_body_params == 8:
        return {
            "front_left_leg": float(body[0]),
            "front_left_ankle": float(body[1]),
            "front_right_leg": float(body[2]),
            "front_right_ankle": float(body[3]),
            "back_left_leg": float(body[4]),
            "back_left_ankle": float(body[5]),
            "back_right_leg": float(body[6]),
            "back_right_ankle": float(body[7]),
        }
    elif n_body_params == 4:
        return {
            "front_left_leg": float(body[0]),
            "front_left_ankle": float(body[1]),
            "front_right_leg": float(body[0]),
            "front_right_ankle": float(body[1]),
            "back_left_leg": float(body[2]),
            "back_left_ankle": float(body[3]),
            "back_right_leg": float(body[2]),
            "back_right_ankle": float(body[3]),
        }
    else:
        raise ValueError(f"Unsupported n_body_params={n_body_params}")


def stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "values": [float(v) for v in arr],
    }


class AntWorld(World):
    def __init__(self, n_body_params: int):
        action_space = 8
        state_space = 27

        self.controller = NeuralNetworkController(
            input_size=state_space,
            output_size=action_space,
            hidden_size=8,
        )

        self.n_weights = self.controller.n_params
        self.n_body_params = n_body_params
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
        render_mode="rgb_array",
        max_episode_steps=1000,
        reset_noise_scale=0.1,
    ):
        return gym.make(
            ENV_NAME,
            robot_path=self.world_file,
            reset_noise_scale=reset_noise_scale,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )

    def geno2pheno(self, genotype: np.ndarray):
        assert len(genotype) == self.n_params, (
            f"Genotype has {len(genotype)} params, expected {self.n_params}"
        )

        control_weights = genotype[:self.n_weights]
        body_params = (genotype[self.n_weights:] + 1) / 4 + 0.1

        assert len(control_weights) == self.n_weights
        assert len(body_params) == self.n_body_params
        assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        if self.n_body_params == 8:
            (
                front_left_leg,
                front_left_ankle,
                front_right_leg,
                front_right_ankle,
                back_left_leg,
                back_left_ankle,
                back_right_leg,
                back_right_ankle,
            ) = body_params
        elif self.n_body_params == 4:
            front_leg, front_ankle, back_leg, back_ankle = body_params
            front_left_leg = front_leg
            front_left_ankle = front_ankle
            front_right_leg = front_leg
            front_right_ankle = front_ankle
            back_left_leg = back_leg
            back_left_ankle = back_ankle
            back_right_leg = back_leg
            back_right_ankle = back_ankle
        else:
            raise ValueError(f"Unsupported n_body_params={self.n_body_params}")

        front_left_hip_xyz = np.array([0.2, 0.2, 0.0])
        front_left_knee_xyz = np.array([
            np.sqrt(0.5 * front_left_leg**2),
            np.sqrt(0.5 * front_left_leg**2),
            0.0,
        ]) + front_left_hip_xyz
        front_left_toe_xyz = np.array([
            np.sqrt(0.5 * front_left_ankle**2),
            np.sqrt(0.5 * front_left_ankle**2),
            0.0,
        ]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0.0])
        front_right_knee_xyz = np.array([
            -np.sqrt(0.5 * front_right_leg**2),
            np.sqrt(0.5 * front_right_leg**2),
            0.0,
        ]) + front_right_hip_xyz
        front_right_toe_xyz = np.array([
            -np.sqrt(0.5 * front_right_ankle**2),
            np.sqrt(0.5 * front_right_ankle**2),
            0.0,
        ]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0.0])
        back_left_knee_xyz = np.array([
            -np.sqrt(0.5 * back_left_leg**2),
            -np.sqrt(0.5 * back_left_leg**2),
            0.0,
        ]) + back_left_hip_xyz
        back_left_toe_xyz = np.array([
            -np.sqrt(0.5 * back_left_ankle**2),
            -np.sqrt(0.5 * back_left_ankle**2),
            0.0,
        ]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0.0])
        back_right_knee_xyz = np.array([
            np.sqrt(0.5 * back_right_leg**2),
            -np.sqrt(0.5 * back_right_leg**2),
            0.0,
        ]) + back_right_hip_xyz
        back_right_toe_xyz = np.array([
            np.sqrt(0.5 * back_right_ankle**2),
            -np.sqrt(0.5 * back_right_ankle**2),
            0.0,
        ]) + back_right_knee_xyz

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
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        slope_map = X * rise

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
        img.save(os.path.join(self.temp_dir.name, filename))

    def evaluate_individual(self, genotype, n_repeats=1, n_steps=500):
        rewards = evaluate_genotype(self, genotype, n_episodes=n_repeats, max_steps=n_steps)
        return float(np.mean(rewards)), None


def infer_body_params(genotype: np.ndarray, n_weights: int = 280) -> int:
    n_body_params = len(genotype) - n_weights
    if n_body_params not in (4, 8):
        raise ValueError(
            f"Loaded genotype has {len(genotype)} params. "
            f"Expected 280 weights + 4 or 8 body genes."
        )
    return n_body_params


def evaluate_genotype(world: AntWorld, genotype: np.ndarray, n_episodes=5, max_steps=500):
    rewards = []

    for ep in range(n_episodes):
        world.update_robot_xml(genotype)
        env = world.create_env(render_mode="rgb_array", max_episode_steps=max_steps)
        obs, _ = env.reset(seed=ep)
        world.controller.reset_controller(batch_size=1)

        total_reward = 0.0
        for _ in range(max_steps):
            action = world.controller.get_action(obs)
            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action.squeeze(0)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        env.close()
        rewards.append(total_reward)

    return rewards


def save_video(world: AntWorld, genotype: np.ndarray, out_path: str, n_steps: int = 500):
    world.update_robot_xml(genotype)
    env = world.create_env(render_mode="rgb_array", max_episode_steps=n_steps)
    obs, _ = env.reset(seed=0)
    world.controller.reset_controller(batch_size=1)

    frames = []
    total_reward = 0.0

    for _ in range(n_steps):
        frames.append(env.render())

        action = world.controller.get_action(obs)
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.squeeze(0)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    imageio.mimwrite(out_path, frames, fps=20)
    return total_reward


def main():
    xbest_path = os.path.join(RESULTS_DIR, "x_best.npy")
    if not os.path.isfile(xbest_path):
        raise FileNotFoundError(f"Missing file: {xbest_path}")

    genotype = np.load(xbest_path, allow_pickle=True)
    genotype = np.asarray(genotype).squeeze()

    n_body_params = infer_body_params(genotype, n_weights=280)
    world = AntWorld(n_body_params=n_body_params)

    print("Loaded genotype shape:", genotype.shape)
    print("Detected n_weights:", world.n_weights)
    print("Detected n_body_params:", world.n_body_params)
    print("Expected total n_params:", world.n_params)

    if genotype.shape[0] != world.n_params:
        raise ValueError(
            f"Mismatch after detection: loaded genotype has {genotype.shape[0]} params, "
            f"but world expects {world.n_params}."
        )

    body = decode_body(genotype, world.n_body_params)
    print("\nDecoded morphology:")
    for k, v in body.items():
        print(f"{k}: {v:.4f}")

    rewards = evaluate_genotype(world, genotype, n_episodes=5, max_steps=500)
    reward_stats = stats(rewards)

    print("\nReward stats over 5 episodes:")
    print(f"mean   : {reward_stats['mean']:.4f}")
    print(f"std    : {reward_stats['std']:.4f}")
    print(f"min    : {reward_stats['min']:.4f}")
    print(f"max    : {reward_stats['max']:.4f}")
    print(f"median : {reward_stats['median']:.4f}")

    video_path = os.path.join(RESULTS_DIR, "test_mlp_video.mp4")
    video_reward = save_video(world, genotype, video_path, n_steps=500)
    print(f"\nSaved video to: {video_path}")
    print(f"Video episode reward: {video_reward:.4f}")

    report = {
        "checkpoint_dir": RESULTS_DIR,
        "genotype_shape": int(genotype.shape[0]),
        "n_weights": int(world.n_weights),
        "n_body_params": int(world.n_body_params),
        "decoded_body": body,
        "reward_stats_5eps": reward_stats,
        "video_reward": float(video_reward),
        "video_path": video_path,
    }

    report_path = os.path.join(RESULTS_DIR, "test_mlp_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()