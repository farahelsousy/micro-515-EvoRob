"""
MICRO-515 Final Project — Multi-task Robot Evolution
=====================================================
Clean corrected version:
- SmoothedHebbianMLPController
- 4-parameter symmetric body representation
- SyncVectorEnv for macOS stability
- NSGA-II multi-objective optimization over [flat, ice, hill]
- Proper checkpoint XML copying
- No warm-start by default
"""

import os
import shutil
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import scipy.ndimage
from PIL import Image
from gymnasium.vector import SyncVectorEnv

import evorob.world  # registers environments
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.base import World
from evorob.world.robot.controllers.smoothedhebbian import SmoothedHebbianMLPController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
_ASSETS = join(ROOT_DIR, "evorob", "world", "robot", "assets")
MAX_EPISODE_STEPS = 1000


class FinalWorld(World):
    """Body + controller co-evolution across flat, ice, and hill terrains."""

    def __init__(self):
        self.controller = SmoothedHebbianMLPController(
            input_size=27,
            output_size=8,
            hidden_size=8,
        )

        self.n_weights = self.controller.n_params
        self.n_body_params = 4  # front upper, front lower, rear upper, rear lower
        self.n_params = self.n_weights + self.n_body_params

        self.temp_dir = TemporaryDirectory()
        self.flat_world_file = join(self.temp_dir.name, "WorldFlat.xml")
        self.ice_world_file = join(self.temp_dir.name, "WorldIce.xml")
        self.hill_world_file = join(self.temp_dir.name, "WorldHill.xml")
        self.world_file = self.hill_world_file

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

        # Keep full 27-dim proprioceptive observation.
        # If you change this, also change controller input_size and EvalWorld.
        self.sensor_fn = None

        self._create_terrain_file("terrain.png")

    def geno2pheno(self, genotype: np.ndarray):
        """Decode genotype into controller parameters and symmetric body geometry."""
        control_params = genotype[:self.n_weights] * 0.1
        body_params = (genotype[self.n_weights:] + 1) / 4 + 0.1
        self.controller.geno2pheno(control_params)

        body_raw = genotype[self.n_weights:]

        front_upper = 0.2 + 0.1 * body_raw[0]
        front_lower = 0.4 + 0.1 * body_raw[1]
        rear_upper  = 0.2 + 0.1 * body_raw[2]
        rear_lower  = 0.4 + 0.1 * body_raw[3]

        front_left_leg = front_right_leg = front_upper
        front_left_ankle = front_right_ankle = front_lower
        back_left_leg = back_right_leg = rear_upper
        back_left_ankle = back_right_ankle = rear_lower

        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = np.array([
            np.sqrt(0.5 * front_left_leg ** 2),
            np.sqrt(0.5 * front_left_leg ** 2),
            0,
        ]) + front_left_hip_xyz
        front_left_toe_xyz = np.array([
            np.sqrt(0.5 * front_left_ankle ** 2),
            np.sqrt(0.5 * front_left_ankle ** 2),
            0,
        ]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array([
            -np.sqrt(0.5 * front_right_leg ** 2),
            np.sqrt(0.5 * front_right_leg ** 2),
            0,
        ]) + front_right_hip_xyz
        front_right_toe_xyz = np.array([
            -np.sqrt(0.5 * front_right_ankle ** 2),
            np.sqrt(0.5 * front_right_ankle ** 2),
            0,
        ]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = np.array([
            -np.sqrt(0.5 * back_left_leg ** 2),
            -np.sqrt(0.5 * back_left_leg ** 2),
            0,
        ]) + back_left_hip_xyz
        back_left_toe_xyz = np.array([
            -np.sqrt(0.5 * back_left_ankle ** 2),
            -np.sqrt(0.5 * back_left_ankle ** 2),
            0,
        ]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = np.array([
            np.sqrt(0.5 * back_right_leg ** 2),
            -np.sqrt(0.5 * back_right_leg ** 2),
            0,
        ]) + back_right_hip_xyz
        back_right_toe_xyz = np.array([
            np.sqrt(0.5 * back_right_ankle ** 2),
            -np.sqrt(0.5 * back_right_ankle ** 2),
            0,
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
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=float,
        )
        return points, connectivity_mat

    def update_robot_xml(self, genotype: np.ndarray) -> None:
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(
            points,
            connectivity_mat,
            self.joint_limits,
            self.joint_axis,
            name="Robot",
            verbose=False,
        )
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)

        for template, world_file in [
            (join(_ASSETS, "flat_world.xml"), self.flat_world_file),
            (join(_ASSETS, "ice_world.xml"), self.ice_world_file),
            (join(_ASSETS, "hill_world.xml"), self.hill_world_file),
        ]:
            tree = xml.parse(template)
            root = tree.getroot()
            root.append(xml.Element("include", attrib={"file": "Robot.xml"}))
            with open(world_file, "w") as f:
                f.write(xml.tostring(root, encoding="unicode"))

    def _create_terrain_file(self, filename: str, width: int = 200, depth: int = 400):
        slope_deg = 5.0
        bump_scale = 0.08
        sigma = 4.0

        rise = np.tan(np.deg2rad(slope_deg))
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, _ = np.meshgrid(x, y)

        slope_map = np.clip(X * rise, 0, 1)
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        bump_envelope = np.sin(np.pi * X)
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * bump_envelope
        noise_map = noise * bump_scale

        terrain = np.clip(slope_map + noise_map, 0, 1)
        terrain[-1, -1] = 1
        img = Image.fromarray((terrain * 255).astype(np.uint8), mode="L")
        img.save(join(self.temp_dir.name, filename))

    def _run_env(self, env_id: str, world_file: str, n_repeats: int, n_steps: int) -> float:
        envs = SyncVectorEnv([
            (lambda eid, wf: lambda: gym.make(
                eid,
                robot_path=wf,
                max_episode_steps=n_steps,
            ))(env_id, world_file)
            for _ in range(n_repeats)
        ])

        self.controller.reset_controller(batch_size=n_repeats)
        rewards = np.zeros((n_steps, n_repeats), dtype=float)
        obs, _ = envs.reset()

        if self.sensor_fn is not None:
            obs = self.sensor_fn(obs)

        done = np.zeros(n_repeats, dtype=bool)
        for t in range(n_steps):
            actions = self.controller.get_action(obs)
            actions = np.where(done[:, None], 0.0, actions)

            obs, r, terminated, truncated, _ = envs.step(actions)

            if self.sensor_fn is not None:
                obs = self.sensor_fn(obs)

            rewards[t, ~done] = r[~done]
            done |= terminated | truncated
            if done.all():
                break

        envs.close()
        return float(rewards.sum(axis=0).mean())

    def _eval_flat(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("FlatEnv-v0", self.flat_world_file, n_repeats, n_steps)

    def _eval_ice(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("IceEnv-v0", self.ice_world_file, n_repeats, n_steps)

    def _eval_hill(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("HillEnv-v0", self.hill_world_file, n_repeats, n_steps)

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        return gym.make(
            "HillEnv-v0",
            robot_path=self.hill_world_file,
            render_mode=render_mode,
            **kwargs,
        )

    def evaluate_individual(
        self,
        genotype: np.ndarray,
        n_repeats: int = 4,
        n_steps: int = 500,
    ) -> np.ndarray:
        self.update_robot_xml(genotype)
        return np.array([
            self._eval_flat(n_repeats, n_steps),
            self._eval_ice(n_repeats, n_steps),
            self._eval_hill(n_repeats, n_steps),
        ])


# ---------------------------------------------------------------------------
# Optional local evaluation on training terrains
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 32,
) -> dict | None:
    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _load(fname):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = join(d, fname)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _load("x_best.npy")
    if x_best is None:
        print(f"ERROR: x_best.npy not found in '{checkpoint_dir}'.")
        return None

    world = FinalWorld()
    world.update_robot_xml(x_best)

    terrains = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice": ("IceEnv-v0", world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }

    results = {}
    for name, (env_id, world_file) in terrains.items():
        scores = []
        for _ in range(n_episodes):
            scores.append(world._run_env(env_id, world_file, n_repeats=1, n_steps=MAX_EPISODE_STEPS))
        arr = np.asarray(scores, dtype=float)
        results[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "best": float(arr.max()),
            "worst": float(arr.min()),
        }
        print(f"{name}: {arr.mean():.2f} ± {arr.std():.2f}")

    os.makedirs(output_dir, exist_ok=True)
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def make_warmstart_genotype(old_path: str, world: FinalWorld) -> np.ndarray:
    """Convert old trained genotype to current 588-param format.

    Current format:
      584 controller params + 4 symmetric body params = 588

    Supported old formats:
      304 = old smoothed MLP controller only
      312 = old smoothed MLP controller + 8 body params
      584 = current controller only
      588 = current full genotype
    """
    old = np.load(old_path, allow_pickle=True).astype(np.float32).ravel()
    warm = np.zeros(world.n_params, dtype=np.float32)

    if old.shape[0] == world.n_params:
        warm[:] = old
        return np.clip(warm, -1.0, 1.0)

    if old.shape[0] == 584 and world.n_params == 588:
        warm[:584] = old
        warm[584:588] = 0.0
        return np.clip(warm, -1.0, 1.0)

    if old.shape[0] == 312 and world.n_params == 588:
        warm[:304] = old[:304]
        warm[304:584] = 0.0

        old_body8 = old[304:312]
        front_upper = 0.5 * (old_body8[0] + old_body8[2])
        front_lower = 0.5 * (old_body8[1] + old_body8[3])
        rear_upper = 0.5 * (old_body8[4] + old_body8[6])
        rear_lower = 0.5 * (old_body8[5] + old_body8[7])
        warm[584:588] = np.array(
            [front_upper, front_lower, rear_upper, rear_lower],
            dtype=np.float32,
        )
        return np.clip(warm, -1.0, 1.0)

    if old.shape[0] == 304 and world.n_params == 588:
        warm[:304] = old
        warm[304:584] = 0.0
        warm[584:588] = 0.0
        return np.clip(warm, -1.0, 1.0)

    raise ValueError(
        f"Unsupported warmstart shape {old.shape}. "
        f"Expected 304, 312, 584, or {world.n_params}."
    )


def run_multi_task_evolution(
    num_generations: int = 100,
    population_size: int = 48,
    n_parents: int = 16,
    n_repeats: int = 1,
    n_steps: int = 500,
    mutation_prob: float = 0.7,
    crossover_prob: float = 0.6,
    bounds: tuple = (-1, 1),
    ckpt_interval: int = 5,
    results_dir: str = None,
    random_seed: int = 42,
    warmstart_path: str = None,
    warmstart_noise: float = 0.15,
) -> None:
    np.random.seed(random_seed)

    world = FinalWorld()
    print(
        f"Genotype : {world.n_params} params "
        f"(controller={world.n_weights}, body={world.n_body_params})"
    )

    if results_dir is None:
        results_dir = join(ROOT_DIR, "results", "final_project")

    ea = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )

    warm = None
    if warmstart_path is not None:
        warm = make_warmstart_genotype(warmstart_path, world)
        print(f"Warm-start enabled from: {warmstart_path}")
        print(f"Warm-start noise: {warmstart_noise}")
        print(f"Warm-start body params: {warm[-4:]}")

    n_obj = 3
    print(f"\nRunning {num_generations} generations  pop={population_size}")
    print("Objectives : [flat, ice, hill]")
    print(f"Checkpoints: {results_dir}\n")

    os.makedirs(results_dir, exist_ok=True)
    best_xml_stage = join(results_dir, "_best_robot.xml")
    best_scalar = -np.inf

    for gen in range(num_generations):
        pop = ea.ask()

        if warm is not None and gen == 0:
            pop[0] = warm.copy()
            for i in range(1, len(pop)):
                pop[i] = np.clip(
                    warm + np.random.normal(0.0, warmstart_noise, size=world.n_params),
                    bounds[0],
                    bounds[1],
                )
            print("Injected warm-start population at generation 0")
            print("distance pop0-warm:", np.linalg.norm(pop[0] - warm))
            print("distance pop1-warm:", np.linalg.norm(pop[1] - warm))

        fitnesses = np.empty((len(pop), n_obj), dtype=float)

        for idx, genotype in enumerate(pop):
            fitnesses[idx] = world.evaluate_individual(
                genotype,
                n_repeats=n_repeats,
                n_steps=n_steps,
            )

            scalar = float(fitnesses[idx].sum())
            if scalar > best_scalar:
                best_scalar = scalar
                shutil.copy2(join(world.temp_dir.name, "Robot.xml"), best_xml_stage)

        save_ckpt = (gen % ckpt_interval == 0) or (gen == num_generations - 1)
        ea.tell(pop, fitnesses, save_checkpoint=save_ckpt)

        if save_ckpt:
            gen_dir = join(results_dir, str(gen))
            if os.path.isdir(gen_dir) and os.path.isfile(best_xml_stage):
                shutil.copy2(best_xml_stage, join(gen_dir, "Robot.xml"))

    best_f = ea.f_best_so_far
    score_path = join(results_dir, "training_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generations     : {num_generations}\n")
        f.write(f"Population size : {population_size}\n")
        f.write(f"Parents         : {n_parents}\n")
        f.write(f"Repeats         : {n_repeats}\n")
        f.write(f"Steps           : {n_steps}\n")
        f.write(f"Mutation prob   : {mutation_prob}\n")
        f.write(f"Crossover prob  : {crossover_prob}\n")
        f.write(f"Controller      : {type(world.controller).__name__} ({world.n_weights} params)\n")
        f.write(f"Genotype size   : {world.n_params} (controller={world.n_weights}, body={world.n_body_params})\n\n")
        f.write("Best individual:\n")
        labels = ["flat", "ice", "hill"]
        for label, val in zip(labels, best_f):
            f.write(f"  {label:<6}: {float(val):10.2f}\n")
        f.write(f"  {'sum':<6}: {float(best_f.sum()):10.2f}\n")

    print(f"\nTraining summary saved to: {score_path}")


if __name__ == "__main__":
    run_multi_task_evolution(
        num_generations=80,
        population_size=48,
        n_parents=16,
        n_repeats=1,
        n_steps=500,
        mutation_prob=0.7,
        crossover_prob=0.6,
        ckpt_interval=5,
        results_dir=join(ROOT_DIR, "results", "warmstart_sweep_explore_final"),
        random_seed=42,
        warmstart_path=join(
            ROOT_DIR,
            "results",
            "20260510_170905_smoothed_mlp_ckpts/999",
            "x_best.npy",
        ),
        warmstart_noise=0.15,
    )
