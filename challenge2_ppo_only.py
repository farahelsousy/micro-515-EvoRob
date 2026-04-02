import os

# macOS-friendly; on Linux headless you may prefer "egl" or "osmesa"
os.environ.setdefault("MUJOCO_GL", "glfw")
import matplotlib
matplotlib.use("Agg")

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np

from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir
from evorob.world.ant_multi_world import AntMultiWorld
from evorob.world.ant_world import AntFlatWorld
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController

"""
    Multi-objective optimisation: Ant two-terrains
    PPO-only baseline inside the SAME NSGA-II pipeline
"""


# ---------------------------------------------------------------------------
# PPO-only controller
# ---------------------------------------------------------------------------

class PPOOnlyAntController:
    """
    Frozen PPO-only controller.

    Keeps the same style as your current CompatibleHybridAntController:
    - build a NeuralNetworkController
    - load PPO weights into it with load_from_ppo_model()

    No residual, no oscillator, no trainable control parameters.

    To keep the NSGA-II pipeline unchanged, we expose 1 dummy parameter
    that is ignored completely.
    """

    BASE_HIDDEN = [256, 256]
    PPO_PATH = "micro-515-EvoRob/results/ppo_ckpts_both/ppo_ant_both_4000000_steps.zip"

    def __init__(self, input_size, output_size):
        obs_dim = int(input_size)
        action_dim = int(output_size)

        # --------------------------------------------------
        # Frozen PPO base controller
        # --------------------------------------------------
        base_controller = NeuralNetworkController(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_size=self.BASE_HIDDEN,
        )

        if os.path.isfile(self.PPO_PATH):
            try:
                from stable_baselines3 import PPO
                model = PPO.load(self.PPO_PATH, device="cpu")
                base_controller.load_from_ppo_model(model)
                print(f"[PPOOnlyController] Loaded PPO weights from: {self.PPO_PATH}")
            except Exception as e:
                print(
                    f"[PPOOnlyController] Warning: failed to load PPO model from "
                    f"'{self.PPO_PATH}'. Using random frozen base controller instead. "
                    f"Error: {e}"
                )
        else:
            print(
                f"[PPOOnlyController] Warning: PPO checkpoint not found at "
                f"'{self.PPO_PATH}'. Using random frozen base controller instead."
            )

        self.base_controller = base_controller
        self.input_size = obs_dim
        self.output_size = action_dim

        # Keep 1 dummy parameter so NSGA-II code stays unchanged.
        # This parameter has NO effect.
        self.n_params = 1

    def reset_controller(self, batch_size=1):
        if hasattr(self.base_controller, "reset_controller"):
            self.base_controller.reset_controller(batch_size=batch_size)

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float32)
        action = self.base_controller.get_action(state)
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        return action

    def set_weights(self, weights):
        """
        Ignore genotype completely.
        We only keep this for compatibility with AntMultiWorld / NSGA-II.
        """
        if weights is None:
            return

        weights = np.asarray(weights, dtype=np.float32).ravel()
        if len(weights) != self.n_params:
            raise ValueError(
                f"[PPOOnlyController] Expected {self.n_params} dummy param, "
                f"got {len(weights)}"
            )

        # Intentionally ignored

    def get_num_params(self):
        return self.n_params

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_phase_info(self):
        return {}


def test_exercise_implementation():
    """Test NSGA-II implementation components."""
    print("\n" + "=" * 60)
    print("EXERCISE 2: Testing NSGA-II Components")
    print("=" * 60)

    # Test 1: Dominance Function
    print("\n[1/5] Testing Pareto Dominance...")
    try:
        nsga = NSGAII(population_size=10, n_opt_params=5)

        assert nsga.dominates([5, 3], [4, 2]) is True, (
            "[5,3] should dominate [4,2] (better in both)"
        )

        assert nsga.dominates([5, 2], [4, 3]) is False, (
            "[5,2] should NOT dominate [4,3] (trade-off)"
        )

        assert nsga.dominates([5, 3], [5, 2]) is True, (
            "[5,3] should dominate [5,2] (equal in first, better in second)"
        )

        assert nsga.dominates([4, 3], [4, 3]) is False, (
            "[4,3] should NOT dominate [4,3] (identical)"
        )

        print("✅ Dominance function works correctly!")
    except NotImplementedError as e:
        print(f"❌ Not implemented: {str(e)}")
        print("   👉 Implement dominates() in nsga.py")
        print("   See Exercise 2a in challenge2.md for guidance")
        exit(1)
    except AssertionError as e:
        print(f"❌ Assertion failed: {str(e)}")
        print("   👉 Check your dominance logic")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        exit(1)

    # Test 2: Fast Non-Dominated Sorting
    print("\n[2/5] Testing Fast Non-Dominated Sorting...")
    try:
        nsga = NSGAII(population_size=10, n_opt_params=5)

        test_fitness = np.array(
            [
                [5, 5],
                [6, 4],
                [4, 6],
                [5, 3],
                [3, 5],
                [3, 3],
            ]
        )

        fronts, ranks = nsga.fast_nondominated_sort(test_fitness)

        assert len(fronts[0]) == 3, (
            f"Front 0 should have 3 solutions, got {len(fronts[0])}"
        )
        assert all(ranks[i] == 0 for i in fronts[0]), (
            "Front 0 solutions should have rank 0"
        )

        assert len(fronts[1]) == 2, (
            f"Front 1 should have 2 solutions, got {len(fronts[1])}"
        )
        assert all(ranks[i] == 1 for i in fronts[1]), (
            "Front 1 solutions should have rank 1"
        )

        assert len(fronts[2]) == 1, (
            f"Front 2 should have 1 solution, got {len(fronts[2])}"
        )
        assert ranks[5] == 2, "Last solution should be in Front 2"

        print(f"✅ Sorting correctly identified {len(fronts)} fronts!")
        print(f"   Front 0: {len(fronts[0])} solutions (non-dominated)")
        print(f"   Front 1: {len(fronts[1])} solutions")
        print(f"   Front 2: {len(fronts[2])} solutions")

    except NotImplementedError as e:
        print(f"❌ Not implemented: {str(e)}")
        print("   👉 Complete the TODOs in fast_nondominated_sort()")
        print("   See Exercise 2b in challenge2.md for guidance")
        exit(1)
    except AssertionError as e:
        print(f"❌ Assertion failed: {str(e)}")
        print("   👉 Check your sorting logic")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        exit(1)

    print("\n" + "-" * 60)
    print("Testing Enhanced Diversity (Crowding Distance)")
    print("-" * 60)

    print("\n[3/5] Testing Crowding Distance...")
    try:
        nsga = NSGAII(population_size=10, n_opt_params=5)

        test_fitness = np.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
            ]
        )
        front_indices = [0, 1, 2, 3, 4]

        distances = nsga.compute_crowding_distance(test_fitness, front_indices)

        assert distances[0] == np.inf, "First solution should have infinite distance"
        assert distances[4] == np.inf, "Last solution should have infinite distance"

        assert distances[1] > 0 and np.isfinite(distances[1]), (
            "Interior solution should have finite positive distance"
        )
        assert distances[2] > 0 and np.isfinite(distances[2]), (
            "Interior solution should have finite positive distance"
        )
        assert distances[3] > 0 and np.isfinite(distances[3]), (
            "Interior solution should have finite positive distance"
        )

        print("✅ Crowding distance works correctly!")
        print(f"   Boundary distances: {distances[0]}, {distances[4]} (should be ∞)")
        print(
            f"   Interior distances: {distances[1]:.3f}, {distances[2]:.3f}, {distances[3]:.3f}"
        )

    except NotImplementedError:
        print("⏭️  Skipped (not implemented)")
        print("   This is optional. See challenge2.md for details if interested.")
    except AssertionError as e:
        print(f"⚠️  Implementation issue: {str(e)}")
    except Exception as e:
        print(f"⚠️  Error: {type(e).__name__}: {str(e)}")

    print("\n[4/5] Testing Crowding Operator...")
    try:
        nsga = NSGAII(population_size=10, n_opt_params=5)

        ranks = [0, 0, 1, 1]
        crowding_dists = np.array([2.0, 3.0, 5.0, 1.0])

        winner = nsga.crowding_operator(0, 2, ranks, crowding_dists)
        assert winner == 0, "Individual with rank 0 should beat rank 1"

        winner = nsga.crowding_operator(0, 1, ranks, crowding_dists)
        assert winner == 1, "Individual with crowding distance 3.0 should beat 2.0"

        print("✅ Crowding operator works correctly!")
        print("   ✓ Prefers lower rank (better front)")
        print("   ✓ Prefers larger crowding distance (more diverse)")

    except NotImplementedError:
        print("⏭️  Skipped (not implemented)")
        print("   This is optional. See challenge2.md for details if interested.")
    except AssertionError as e:
        print(f"⚠️  Implementation issue: {str(e)}")
    except Exception as e:
        print(f"⚠️  Error: {type(e).__name__}: {str(e)}")

    print("\n[5/5] Testing Enhanced Parent Selection...")

    try:
        nsga = NSGAII(population_size=10, n_opt_params=5, n_parents=5)

        test_population = np.random.uniform(-1, 1, (10, 5))
        test_fitness = np.random.uniform(0, 10, (10, 2))

        parents, parent_fitness = nsga.sort_and_select_parents(
            test_population, test_fitness, n_parents=5
        )

        assert parents.shape == (5, 5), (
            f"Should select 5 parents with 5 params, got shape {parents.shape}"
        )
        assert parent_fitness.shape == (5, 2), (
            f"Parent fitness should be (5, 2), got {parent_fitness.shape}"
        )

        print("✅ Parent selection works!")
        print(
            f"   Selected {len(parents)} parents from population of {len(test_population)}"
        )
        print("   Note: This uses rank-based selection (crowding distance is optional)")

    except AssertionError as e:
        print(f"⚠️  Implementation issue: {str(e)}")
    except Exception as e:
        print(f"⚠️  Error: {type(e).__name__}: {str(e)}")

    print("\n" + "=" * 60)
    print("🎉 NSGA-II tests passed!")
    print("=" * 60)
    print("\nYour implementation is ready to run multi-objective evolution.")
    print("Uncomment the evolution code below to start training.\n")


def inspect_ant_multi_world():
    """Test the AntMultiWorld environment."""
    world = AntMultiWorld(controller_cls=PPOOnlyAntController)
    print(f"Observation space: {world.obs_size}")
    print(f"Action space: {world.action_size}")
    print(f"Controller parameters: {world.n_params}")

    random_genotype = np.random.uniform(-1, 1, world.n_params)
    fitness = world.evaluate_individual(random_genotype)
    print(f"Fitness of random individual: {fitness}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_fitness(full_f, output_dir):
    """Save a fitness-over-generations plot for both objectives."""
    fitness_array = np.array(full_f)
    generations = np.arange(1, len(fitness_array) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    obj_labels = ["Objective 1 (Flat)", "Objective 2 (Ice)"]

    for obj_idx, (ax, label) in enumerate(zip(axes, obj_labels)):
        obj_fitness = fitness_array[:, :, obj_idx]
        best_per_gen = np.max(obj_fitness, axis=1)
        mean_per_gen = np.mean(obj_fitness, axis=1)
        std_per_gen = np.std(obj_fitness, axis=1)

        ax.plot(
            generations, best_per_gen,
            label="Best", color="#B51F1F", linewidth=2, linestyle="--",
        )
        ax.plot(
            generations, mean_per_gen,
            label="Mean", color="#007480", linewidth=2,
        )
        ax.fill_between(
            generations,
            mean_per_gen - std_per_gen,
            mean_per_gen + std_per_gen,
            alpha=0.2, color="#007480", label="Mean +/- 1 std",
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fitness over Generations", fontsize=14)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "fitness_plot.pdf")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Fitness plot saved to: {plot_path}")


def plot_pareto_fronts(fitness, output_dir, num_generations=None, population_size=None):
    """Plot Pareto fronts for a 2-objective fitness array."""
    dummy_nsga = NSGAII(population_size=fitness.shape[0], n_opt_params=1)
    fronts, _ = dummy_nsga.fast_nondominated_sort(fitness)

    fig, ax = plt.subplots(figsize=(9, 6))
    n_fronts = len(fronts)

    top_colors = ["#B51F1F", "#007480", "#4B0082"]
    n_top = min(3, n_fronts)
    for i in range(n_top):
        fi = fitness[fronts[i]]
        si = np.argsort(fi[:, 0])
        fi_sorted = fi[si]
        ax.plot(
            fi_sorted[:, 0], fi_sorted[:, 1],
            color=top_colors[i], alpha=0.5, linewidth=1.2, zorder=3,
        )
        ax.scatter(
            fi[:, 0], fi[:, 1],
            label=f"Front {i}",
            color=top_colors[i],
            s=50,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )

    if n_fronts > 3:
        remaining_cmap = plt.cm.coolwarm
        for i in range(3, n_fronts):
            fi = fitness[fronts[i]]
            t = (i - 3) / max(n_fronts - 4, 1)
            ax.scatter(
                fi[:, 0], fi[:, 1],
                label=f"Front {i}" if i <= 6 else None,
                color=remaining_cmap(t),
                s=25,
                alpha=0.5,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )

    ax.set_xlabel("Fitness — Flat Terrain", fontsize=11)
    ax.set_ylabel("Fitness — Ice Terrain", fontsize=11)
    info = [f"{n_fronts} front{'s' if n_fronts > 1 else ''}"]
    if num_generations is not None:
        info.insert(0, f"gen {num_generations}")
    if population_size is not None:
        info.insert(1 if num_generations else 0, f"pop {population_size}")
    ax.set_title(f"Pareto Fronts  ({',  '.join(info)})", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    pareto_path = os.path.join(output_dir, "pareto_fronts.pdf")
    fig.savefig(pareto_path, dpi=150)
    plt.close(fig)
    print(f"Pareto front plot saved to: {pareto_path}")


def plot_pareto_fronts_from_checkpoint(checkpoint_dir: str):
    """Load fitness data from a checkpoint directory and plot Pareto fronts."""
    fitness_path = f"{checkpoint_dir}/full_f.npy"
    try:
        all_fitness = np.load(fitness_path)
    except Exception as e:
        print(f"Could not load fitness data from {fitness_path}: {e}")
        return

    fitness = all_fitness[-1] if all_fitness.ndim == 3 else all_fitness
    plot_pareto_fronts(fitness, checkpoint_dir)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_episodes(env, controller, genotype, n_episodes, max_episode_steps, seed):
    """Run n_episodes on a single environment, return list of episode rewards."""
    rng = np.random.default_rng(seed)
    controller.geno2pheno(genotype)
    episode_rewards = []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        controller.reset_controller(batch_size=1)

        total_reward = 0.0
        for _ in range(max_episode_steps):
            action = controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

    return episode_rewards


def _record_video(env_cls, robot_path, controller, genotype, max_steps, seed, out_path):
    """Record a single-episode video. Returns episode reward or None on failure."""
    try:
        env = env_cls(render_mode="rgb_array", robot_path=robot_path)
        controller.geno2pheno(genotype)
        obs, _ = env.reset(seed=seed)
        controller.reset_controller(batch_size=1)
        frames = []
        video_reward = 0.0

        for _ in range(max_steps):
            frames.append(env.render())
            action = controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action)
            video_reward += reward
            if terminated or truncated:
                break

        env.close()
        imageio.mimwrite(out_path, frames, fps=20)
        print(f"  Video saved to: {out_path}")
        return video_reward
    except Exception as e:
        print(f"  Warning: Video recording skipped (rendering unavailable): {e}")
        return None


def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
):
    """Evaluate a checkpoint on both custom environments (flat + ice terrain)."""
    n_episodes: int = 256
    max_episode_steps: int = 1000
    seed: int = 0

    last_gen = get_last_checkpoint_dir(checkpoint_dir)
    x_best_path = os.path.join(last_gen, "x_best.npy") if last_gen else ""

    if not os.path.isfile(x_best_path):
        x_best_path = os.path.join(checkpoint_dir, "x_best.npy")

    if not os.path.isfile(x_best_path):
        print(f"ERROR: Could not find x_best.npy in '{checkpoint_dir}'.")
        print("Make sure the path points to your checkpoint folder.")
        return None

    genotype = np.load(x_best_path)
    print(f"Loaded genotype from: {x_best_path}  (shape: {genotype.shape})")

    controller = PPOOnlyAntController(input_size=27, output_size=8)
    print(
        f"Controller: PPOOnlyAntController  |  Parameters: {controller.n_params}\n"
    )

    terrains = {
        "flat": "ant_flat_terrain.xml",
        "ice": "ant_ice_terrain.xml",
    }
    results = {}

    for terrain_name, robot_path in terrains.items():
        print(f"Evaluating on {terrain_name} terrain ({n_episodes} episodes)...")
        env = AntFlatEnvironment(robot_path=robot_path)
        rewards = _run_episodes(
            env, controller, genotype, n_episodes, max_episode_steps, seed
        )
        env.close()

        results[terrain_name] = {
            "rewards": rewards,
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "best": float(np.max(rewards)),
            "worst": float(np.min(rewards)),
            "median": float(np.median(rewards)),
        }
        r = results[terrain_name]
        print(
            f"  {terrain_name.capitalize():5s}: mean={r['mean']:.2f} +/- {r['std']:.2f}  "
            f"best={r['best']:.2f}  worst={r['worst']:.2f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    for terrain_name, robot_path in terrains.items():
        video_path = os.path.join(output_dir, f"evaluation_{terrain_name}.mp4")
        _record_video(
            AntFlatEnvironment, robot_path, controller, genotype,
            max_episode_steps, seed, video_path,
        )

    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("MICRO-515 Challenge 2 - Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Controller type : PPOOnlyAntController\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/terrain: {n_episodes}\n\n")

        f.write("=" * 60 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Terrain':<8s} {'Mean':>8s} {'Std':>8s} {'Best':>8s} {'Worst':>8s} {'Median':>8s}\n")
        f.write("-" * 60 + "\n")
        for terrain_name in terrains:
            r = results[terrain_name]
            f.write(
                f"{terrain_name.upper():<8s} {r['mean']:8.2f} {r['std']:8.2f} "
                f"{r['best']:8.2f} {r['worst']:8.2f} {r['median']:8.2f}\n"
            )
        f.write("\n")

        for terrain_name in terrains:
            r = results[terrain_name]
            f.write("-" * 50 + "\n")
            f.write(f"{terrain_name.upper()} TERRAIN — Per-episode rewards\n")
            f.write("-" * 50 + "\n")
            for i, rew in enumerate(r["rewards"]):
                f.write(f"  Episode {i + 1:3d}: {rew:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print(f"\n{'=' * 50}")
    print(f"  FLAT : {results['flat']['mean']:.2f} +/- {results['flat']['std']:.2f}  (best: {results['flat']['best']:.2f})")
    print(f"  ICE  : {results['ice']['mean']:.2f} +/- {results['ice']['std']:.2f}  (best: {results['ice']['best']:.2f})")
    print(f"{'=' * 50}")

    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_evolution_nsga(
    num_generations: int,
    population_size: int,
    ckpt_interval: int,
    run_evaluation: bool,
    compute_score: bool,
    random_seed: int,
    n_repeats: int,
    mutation_prob: float,
    crossover_prob: float,
    bounds: Tuple[float, float],
    n_parents: int,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Run NSGA-II multi-objective evolutionary optimization."""
    np.random.seed(random_seed)

    world = AntMultiWorld(
        controller_cls=PPOOnlyAntController,
        n_repeats=n_repeats,
    )

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_path is None:
        checkpoint_path = f"results/{dt_str}_nsga_ckpts"
    else:
        checkpoint_path = str(
            Path(checkpoint_path).parent / f"{dt_str}_{Path(checkpoint_path).name}"
        )

    ckpt_dir = Path(checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    nsga_kwargs = dict(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
    )
    nsga = NSGAII(**nsga_kwargs, output_dir=ckpt_dir)

    metadata_path = ckpt_dir / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("NSGA-II Training Metadata\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date            : {dt_str}\n")
        f.write(f"Generations     : {num_generations}\n")
        f.write(f"Population size : {population_size}\n")
        f.write(f"Random seed     : {random_seed}\n")
        f.write(f"Num parameters  : {world.n_params}\n")
        f.write(f"Repeats         : {world.n_repeats}\n")
        f.write(f"Episode steps   : {world.max_episode_steps}\n")
        f.write(f"Checkpoint dir  : {ckpt_dir}\n\n")
        f.write("-" * 60 + "\n")
        f.write("NSGA-II Hyperparameters\n")
        f.write("-" * 60 + "\n")
        for key, val in nsga_kwargs.items():
            f.write(f"  {key:<20s}: {val}\n")
        f.write("\n")
    print(f"Metadata saved to: {metadata_path}")

    print("\n" + "=" * 70)
    print(f"{'MULTI-OBJECTIVE EVOLUTION - NSGA-II':^70}")
    print("=" * 70)
    print(
        f"Population: {population_size} | Generations: {num_generations} | "
        f"Parents: {nsga.n_parents}"
    )
    print(f"Objective 1: Flat Terrain Speed | Objective 2: Ice Terrain Speed")
    print("=" * 70 + "\n")

    for generation in range(num_generations):
        population = nsga.ask()
        multi_fitness = np.empty((len(population), 2))

        for i, individual in enumerate(population):
            multi_fitness[i] = world.evaluate_individual(individual)

        save_checkpoint = (
            (generation % ckpt_interval == 0) or (generation == num_generations - 1)
        )
        nsga.tell(population, multi_fitness, save_checkpoint=save_checkpoint)

        mean_obj1 = np.mean(multi_fitness[:, 0])
        mean_obj2 = np.mean(multi_fitness[:, 1])
        best_obj1 = np.max(multi_fitness[:, 0])
        best_obj2 = np.max(multi_fitness[:, 1])

        progress = (generation + 1) / num_generations
        bar_length = 50
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(
            f"Gen {generation + 1:4d}/{num_generations} [{bar}] {progress * 100:5.1f}%"
        )
        print(
            f"     Best:     Objective 1={best_obj1:7.2f}  Objective 2={best_obj2:7.2f}"
        )
        print(
            f"     Mean:     Objective 1={mean_obj1:7.2f}  Objective 2={mean_obj2:7.2f}"
        )
        print()

    plot_fitness(nsga.full_f, ckpt_dir)

    final_fitness = np.array(nsga.full_f)[-1]
    plot_pareto_fronts(
        final_fitness, ckpt_dir,
        num_generations=num_generations,
        population_size=population_size,
    )

    eval_results = None
    if compute_score:
        try:
            eval_results = evaluate_checkpoint(
                checkpoint_dir=str(ckpt_dir),
                output_dir=str(ckpt_dir),
            )
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")

    if run_evaluation:
        best_population = nsga.x
        best_fitness = nsga.f
        best_flat_idx = np.argmax(best_fitness[:, 0])
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            try:
                evaluation_env = world.create_env(render_mode="human")
                evaluation_controller = world.controller
                evaluation_controller.geno2pheno(best_population[best_flat_idx])

                obs, _ = evaluation_env.reset()
                trial_reward = 0.0
                trial_count = 0

                print("\nPress Ctrl+C to stop the evaluation...")
                try:
                    while True:
                        action = evaluation_controller.get_action(obs)
                        obs, reward, terminated, truncated, _ = evaluation_env.step(action)
                        trial_reward += reward

                        if np.logical_or(terminated, truncated):
                            trial_count += 1
                            print(f"Trial {trial_count} reward: {float(trial_reward):.2f}")
                            trial_reward = 0.0
                            obs, _ = evaluation_env.reset()
                except KeyboardInterrupt:
                    print(f"\n\nEvaluation stopped by user after {trial_count} trials.")
                finally:
                    evaluation_env.close()
            except Exception as e:
                print(f"Warning: Interactive evaluation skipped (rendering unavailable): {e}")
        else:
            print("Skipping interactive evaluation (no display available).")


# ---------------------------------------------------------------------------
# Standalone checkpoint utilities
# ---------------------------------------------------------------------------

def replay_checkpoint(checkpoint_path: str):
    """Re-evaluate a checkpoint and generate videos."""
    np.random.seed(31)

    x_path = os.path.join(checkpoint_path, "x.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Checkpoint file not found: {x_path}")

    population = np.load(x_path)
    world = AntMultiWorld(controller_cls=PPOOnlyAntController)

    multi_fitness = np.empty((len(population), 2))
    for i, individual in enumerate(population):
        multi_fitness[i] = world.evaluate_individual(individual)

    best_flat_idx = np.argmax(multi_fitness[:, 0])
    best_ice_idx = np.argmax(multi_fitness[:, 1])

    print(
        f"Best Flat Terrain Individual: {best_flat_idx} "
        f"with fitness {multi_fitness[best_flat_idx]}"
    )
    print(
        f"Best Ice Terrain Individual: {best_ice_idx} "
        f"with fitness {multi_fitness[best_ice_idx]}"
    )

    plt.scatter(multi_fitness[:, 0], multi_fitness[:, 1], alpha=0.5)
    plt.xlabel("Fitness Objective 1")
    plt.ylabel("Fitness Objective 2")
    plt.title("Multi-Objective Fitness Scatter Plot")
    plt.savefig("fitness_scatter.png")
    plt.close()

    n_evals = 5
    for idx_eval in range(n_evals):
        ant_ice_world = AntFlatWorld(controller_cls=PPOOnlyAntController)
        ant_ice_world.generate_best_individual_video(
            env=ant_ice_world.create_env(
                robot_path="ant_ice_terrain.xml", width=800, height=608
            ),
            video_name=f"best_ice_individual_{idx_eval}.mp4",
            controller=ant_ice_world.geno2pheno(population[best_ice_idx]),
        )

        ant_flat_world = AntFlatWorld(controller_cls=PPOOnlyAntController)
        ant_flat_world.generate_best_individual_video(
            env=ant_flat_world.create_env(
                robot_path="ant_flat_terrain.xml", width=800, height=608
            ),
            video_name=f"best_flat_individual_{idx_eval}.mp4",
            controller=ant_flat_world.geno2pheno(population[best_flat_idx]),
        )
        print(f"Generated videos for iteration {idx_eval + 1}/{n_evals}")


if __name__ == "__main__":
    test_exercise_implementation()
    run_evolution_nsga(
        num_generations=50,
        population_size=120,
        n_parents=50,
        n_repeats=6,
        mutation_prob=0.02,
        crossover_prob=0.8,
        bounds=(-0.05, 0.05),
        compute_score=True,
        run_evaluation=True,
        random_seed=42,
        ckpt_interval=20,
    )

"""
    # Uncomment to replay your checkpoint
    replay_checkpoint(
         checkpoint_path="./results/nsga_multi_terrain_ckpt/99"
     )

    # Uncomment to plot Pareto fronts from checkpoint
    plot_pareto_fronts_from_checkpoint(
         checkpoint_dir="./results/nsga_multi_terrain_ckpt/99"
     )
"""