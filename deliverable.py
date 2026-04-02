import os
import json
from pathlib import Path
from typing import Optional, Tuple

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir
from evorob.world.ant_multi_world import AntMultiWorld
from evorob.world.ant_world import AntFlatWorld
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController

from Challenge2 import CompatibleHybridAntController
# ---------------------------------------------------------------------------
# YOUR EXISTING CONTROLLER(S)
# Keep your own controller definitions here
# Example below assumes CompatibleHybridAntController already exists above.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Utility: load final population / fitness from checkpoint
# ---------------------------------------------------------------------------

def load_final_population_and_fitness(checkpoint_dir: str):
    """
    Try to load final population and fitness from a checkpoint directory.

    Supported patterns:
    - x.npy + f.npy
    - full_x.npy + full_f.npy (takes last generation)
    """
    x_path = os.path.join(checkpoint_dir, "x.npy")
    f_path = os.path.join(checkpoint_dir, "f.npy")

    if os.path.isfile(x_path) and os.path.isfile(f_path):
        population = np.load(x_path, allow_pickle=True)
        fitness = np.load(f_path, allow_pickle=True)
        return population, fitness

    full_x_path = os.path.join(checkpoint_dir, "full_x.npy")
    full_f_path = os.path.join(checkpoint_dir, "full_f.npy")

    if os.path.isfile(full_x_path) and os.path.isfile(full_f_path):
        full_x = np.load(full_x_path, allow_pickle=True)
        full_f = np.load(full_f_path, allow_pickle=True)
        population = full_x[-1]
        fitness = full_f[-1]
        return population, fitness

    # fallback: last generation folder may contain x.npy / f.npy
    last_gen = get_last_checkpoint_dir(checkpoint_dir)
    if last_gen is not None:
        x_path = os.path.join(last_gen, "x.npy")
        f_path = os.path.join(last_gen, "f.npy")
        if os.path.isfile(x_path) and os.path.isfile(f_path):
            population = np.load(x_path, allow_pickle=True)
            fitness = np.load(f_path, allow_pickle=True)
            return population, fitness

    raise FileNotFoundError(
        f"Could not find final population/fitness files in '{checkpoint_dir}'. "
        f"Expected x.npy + f.npy or full_x.npy + full_f.npy."
    )


# ---------------------------------------------------------------------------
# Select specialist + generalist from Pareto front
# ---------------------------------------------------------------------------

def extract_specialist_and_generalist(
    checkpoint_dir: str,
    save_dir: Optional[str] = None,
):
    """
    Extract 3 representative controllers from the final Pareto front:
    - flat specialist = max flat score on Front 0
    - ice specialist  = max ice score on Front 0
    - generalist      = point nearest center of normalized Front 0

    Returns a dict containing genotypes and metadata.
    """
    population, fitness = load_final_population_and_fitness(checkpoint_dir)

    population = np.asarray(population)
    fitness = np.asarray(fitness)

    if population.ndim != 2:
        raise ValueError(f"Expected population shape (N, D), got {population.shape}")
    if fitness.ndim != 2 or fitness.shape[1] != 2:
        raise ValueError(f"Expected fitness shape (N, 2), got {fitness.shape}")

    nsga = NSGAII(
        population_size=len(population),
        n_opt_params=population.shape[1],
    )
    fronts, ranks = nsga.fast_nondominated_sort(fitness)

    if len(fronts) == 0 or len(fronts[0]) == 0:
        raise RuntimeError("Pareto front is empty.")

    pareto_indices = np.array(fronts[0], dtype=int)
    pareto_population = population[pareto_indices]
    pareto_fitness = fitness[pareto_indices].astype(np.float32)

    # Specialists
    flat_idx = int(np.argmax(pareto_fitness[:, 0]))
    ice_idx = int(np.argmax(pareto_fitness[:, 1]))

    # Generalist: closest to center of normalized Pareto front
    fmin = pareto_fitness.min(axis=0)
    fmax = pareto_fitness.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-8, (fmax - fmin), 1.0)
    fit_norm = (pareto_fitness - fmin) / denom

    target = np.array([0.5, 0.5], dtype=np.float32)
    dists = np.linalg.norm(fit_norm - target[None, :], axis=1)
    gen_idx = int(np.argmin(dists))

    selected = {
        "pareto_indices": pareto_indices.tolist(),

        "flat_specialist": {
            "population_index": int(pareto_indices[flat_idx]),
            "pareto_local_index": flat_idx,
            "fitness": pareto_fitness[flat_idx].tolist(),
            "genotype": pareto_population[flat_idx],
        },

        "ice_specialist": {
            "population_index": int(pareto_indices[ice_idx]),
            "pareto_local_index": ice_idx,
            "fitness": pareto_fitness[ice_idx].tolist(),
            "genotype": pareto_population[ice_idx],
        },

        "generalist": {
            "population_index": int(pareto_indices[gen_idx]),
            "pareto_local_index": gen_idx,
            "fitness": pareto_fitness[gen_idx].tolist(),
            "genotype": pareto_population[gen_idx],
        },
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        np.save(
            os.path.join(save_dir, "flat_specialist.npy"),
            selected["flat_specialist"]["genotype"],
        )
        np.save(
            os.path.join(save_dir, "ice_specialist.npy"),
            selected["ice_specialist"]["genotype"],
        )
        np.save(
            os.path.join(save_dir, "generalist.npy"),
            selected["generalist"]["genotype"],
        )

        metadata = {
            "pareto_indices": selected["pareto_indices"],
            "flat_specialist": {
                "population_index": selected["flat_specialist"]["population_index"],
                "pareto_local_index": selected["flat_specialist"]["pareto_local_index"],
                "fitness": selected["flat_specialist"]["fitness"],
            },
            "ice_specialist": {
                "population_index": selected["ice_specialist"]["population_index"],
                "pareto_local_index": selected["ice_specialist"]["pareto_local_index"],
                "fitness": selected["ice_specialist"]["fitness"],
            },
            "generalist": {
                "population_index": selected["generalist"]["population_index"],
                "pareto_local_index": selected["generalist"]["pareto_local_index"],
                "fitness": selected["generalist"]["fitness"],
            },
        }

        with open(os.path.join(save_dir, "selected_controllers.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return selected


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_genotype_on_terrain(
    controller_cls,
    genotype,
    robot_path: str,
    n_episodes: int = 30,
    max_episode_steps: int = 1000,
    seed: int = 0,
):
    """
    Evaluate one genotype on one terrain.
    """
    env = AntFlatEnvironment(robot_path=robot_path)
    controller = controller_cls(input_size=27, output_size=8)

    rng = np.random.default_rng(seed)
    rewards = []

    controller.geno2pheno(genotype)

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=ep_seed)
        controller.reset_controller(batch_size=1)

        total_reward = 0.0
        for _ in range(max_episode_steps):
            action = controller.get_action(obs)
            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action.squeeze(0)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()

    rewards = np.asarray(rewards, dtype=np.float32)
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "median": float(np.median(rewards)),
        "best": float(np.max(rewards)),
        "worst": float(np.min(rewards)),
        "rewards": rewards.tolist(),
    }


def evaluate_selected_controllers(
    checkpoint_dir: str,
    controller_cls,
    output_dir: str,
    n_episodes: int = 30,
):
    """
    Reevaluate flat specialist, ice specialist, and generalist on both terrains.
    Saves a comparison table.
    """
    os.makedirs(output_dir, exist_ok=True)

    selected = extract_specialist_and_generalist(
        checkpoint_dir=checkpoint_dir,
        save_dir=output_dir,
    )

    terrains = {
        "flat": "ant_flat_terrain.xml",
        "ice": "ant_ice_terrain.xml",
    }

    results = {}
    for ctrl_name in ["flat_specialist", "ice_specialist", "generalist"]:
        genotype = selected[ctrl_name]["genotype"]
        results[ctrl_name] = {}

        for terrain_name, robot_path in terrains.items():
            stats = evaluate_genotype_on_terrain(
                controller_cls=controller_cls,
                genotype=genotype,
                robot_path=robot_path,
                n_episodes=n_episodes,
                max_episode_steps=1000,
                seed=0,
            )
            results[ctrl_name][terrain_name] = stats

    # Save text table
    table_path = os.path.join(output_dir, "controller_comparison.txt")
    with open(table_path, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("Controller Comparison: Specialists vs Generalist\n")
        f.write("=" * 90 + "\n\n")
        f.write(
            f"{'Controller':<20}"
            f"{'Flat mean':>12}"
            f"{'Flat std':>12}"
            f"{'Ice mean':>12}"
            f"{'Ice std':>12}\n"
        )
        f.write("-" * 90 + "\n")

        for ctrl_name in ["flat_specialist", "ice_specialist", "generalist"]:
            flat_stats = results[ctrl_name]["flat"]
            ice_stats = results[ctrl_name]["ice"]
            f.write(
                f"{ctrl_name:<20}"
                f"{flat_stats['mean']:12.2f}"
                f"{flat_stats['std']:12.2f}"
                f"{ice_stats['mean']:12.2f}"
                f"{ice_stats['std']:12.2f}\n"
            )

        f.write("\n")
        f.write("=" * 90 + "\n")
        f.write("Selected controller fitness from Pareto front\n")
        f.write("=" * 90 + "\n")
        for ctrl_name in ["flat_specialist", "ice_specialist", "generalist"]:
            fit = selected[ctrl_name]["fitness"]
            f.write(f"{ctrl_name:<20} flat={fit[0]:.2f}, ice={fit[1]:.2f}\n")

    print(f"Controller comparison saved to: {table_path}")

    # Save JSON too
    json_path = os.path.join(output_dir, "controller_comparison.json")
    serializable_results = {}
    for ctrl_name, ctrl_res in results.items():
        serializable_results[ctrl_name] = {}
        for terrain_name, stats in ctrl_res.items():
            serializable_results[ctrl_name][terrain_name] = stats

    with open(json_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Controller comparison JSON saved to: {json_path}")
    return results


# ---------------------------------------------------------------------------
# Pareto plot with selected controllers
# ---------------------------------------------------------------------------

def plot_pareto_with_selected(
    checkpoint_dir: str,
    output_dir: str,
):
    """
    Plot final Pareto fronts and mark:
    - flat specialist
    - ice specialist
    - generalist
    """
    os.makedirs(output_dir, exist_ok=True)

    population, fitness = load_final_population_and_fitness(checkpoint_dir)
    population = np.asarray(population)
    fitness = np.asarray(fitness)

    nsga = NSGAII(
        population_size=len(population),
        n_opt_params=population.shape[1],
    )
    fronts, ranks = nsga.fast_nondominated_sort(fitness)

    selected = extract_specialist_and_generalist(checkpoint_dir)

    fig, ax = plt.subplots(figsize=(9, 6))
    n_fronts = len(fronts)

    top_colors = ["#B51F1F", "#007480", "#4B0082"]
    n_top = min(3, n_fronts)

    for i in range(n_top):
        front = np.array(fronts[i], dtype=int)
        fi = fitness[front]
        sort_idx = np.argsort(fi[:, 0])
        fi_sorted = fi[sort_idx]

        ax.plot(
            fi_sorted[:, 0], fi_sorted[:, 1],
            color=top_colors[i], alpha=0.5, linewidth=1.2, zorder=2,
        )
        ax.scatter(
            fi[:, 0], fi[:, 1],
            color=top_colors[i],
            s=50,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label=f"Front {i}",
        )

    if n_fronts > 3:
        remaining_cmap = plt.cm.coolwarm
        for i in range(3, n_fronts):
            front = np.array(fronts[i], dtype=int)
            fi = fitness[front]
            t = (i - 3) / max(n_fronts - 4, 1)
            ax.scatter(
                fi[:, 0], fi[:, 1],
                color=remaining_cmap(t),
                s=25,
                alpha=0.5,
                edgecolors="white",
                linewidths=0.3,
                zorder=1,
                label=f"Front {i}" if i <= 6 else None,
            )

    fs = np.array(selected["flat_specialist"]["fitness"])
    ispec = np.array(selected["ice_specialist"]["fitness"])
    gen = np.array(selected["generalist"]["fitness"])

    ax.scatter(
        fs[0], fs[1],
        s=180, marker="*",
        label="Flat specialist",
        zorder=5,
    )
    ax.scatter(
        ispec[0], ispec[1],
        s=180, marker="*",
        label="Ice specialist",
        zorder=5,
    )
    ax.scatter(
        gen[0], gen[1],
        s=180, marker="*",
        label="Generalist",
        zorder=5,
    )

    ax.annotate("Flat specialist", (fs[0], fs[1]), xytext=(8, 8), textcoords="offset points")
    ax.annotate("Ice specialist", (ispec[0], ispec[1]), xytext=(8, 8), textcoords="offset points")
    ax.annotate("Generalist", (gen[0], gen[1]), xytext=(8, 8), textcoords="offset points")

    ax.set_xlabel("Fitness — Flat Terrain")
    ax.set_ylabel("Fitness — Ice Terrain")
    ax.set_title("Pareto Front with Selected Controllers")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "pareto_front_selected.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Pareto plot with selected controllers saved to: {out_path}")


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def record_controller_video(
    controller_cls,
    genotype,
    robot_path: str,
    out_path: str,
    max_steps: int = 1000,
    seed: int = 0,
):
    """
    Record one controller on one terrain to MP4.
    """
    env = AntFlatEnvironment(render_mode="rgb_array", robot_path=robot_path)
    controller = controller_cls(input_size=27, output_size=8)
    controller.geno2pheno(genotype)

    obs, _ = env.reset(seed=seed)
    controller.reset_controller(batch_size=1)

    frames = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = controller.get_action(obs)
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.squeeze(0)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    env.close()

    if len(frames) == 0:
        print(f"Warning: no frames captured for {out_path}")
        return

    imageio.mimwrite(out_path, frames, fps=20)
    print(f"Saved video: {out_path} | reward={total_reward:.2f}")


def generate_required_videos(
    checkpoint_dir: str,
    controller_cls,
    output_dir: str,
):
    """
    Generate the required videos:
    - flat specialist on flat
    - ice specialist on ice
    - generalist on flat
    - generalist on ice
    """
    os.makedirs(output_dir, exist_ok=True)
    selected = extract_specialist_and_generalist(checkpoint_dir)

    record_controller_video(
        controller_cls=controller_cls,
        genotype=selected["flat_specialist"]["genotype"],
        robot_path="ant_flat_terrain.xml",
        out_path=os.path.join(output_dir, "flat_specialist_on_flat.mp4"),
    )

    record_controller_video(
        controller_cls=controller_cls,
        genotype=selected["ice_specialist"]["genotype"],
        robot_path="ant_ice_terrain.xml",
        out_path=os.path.join(output_dir, "ice_specialist_on_ice.mp4"),
    )

    record_controller_video(
        controller_cls=controller_cls,
        genotype=selected["generalist"]["genotype"],
        robot_path="ant_flat_terrain.xml",
        out_path=os.path.join(output_dir, "generalist_on_flat.mp4"),
    )

    record_controller_video(
        controller_cls=controller_cls,
        genotype=selected["generalist"]["genotype"],
        robot_path="ant_ice_terrain.xml",
        out_path=os.path.join(output_dir, "generalist_on_ice.mp4"),
    )


# ---------------------------------------------------------------------------
# One-call submission output generator
# ---------------------------------------------------------------------------

def make_submission_outputs(
    checkpoint_dir: str,
    controller_cls,
    output_dir: Optional[str] = None,
    n_eval_episodes: int = 30,
):
    """
    Build all Challenge 2 deliverables from one finished checkpoint.

    Outputs:
    - selected_controllers.json
    - flat_specialist.npy
    - ice_specialist.npy
    - generalist.npy
    - controller_comparison.txt
    - controller_comparison.json
    - pareto_front_selected.pdf
    - flat_specialist_on_flat.mp4
    - ice_specialist_on_ice.mp4
    - generalist_on_flat.mp4
    - generalist_on_ice.mp4
    """
    if output_dir is None:
        output_dir = os.path.join(checkpoint_dir, "submission_outputs")

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Generating Challenge 2 submission outputs")
    print("=" * 80)

    print("\n[1/4] Extracting specialist and generalist controllers...")
    extract_specialist_and_generalist(
        checkpoint_dir=checkpoint_dir,
        save_dir=output_dir,
    )

    print("\n[2/4] Reevaluating selected controllers...")
    evaluate_selected_controllers(
        checkpoint_dir=checkpoint_dir,
        controller_cls=controller_cls,
        output_dir=output_dir,
        n_episodes=n_eval_episodes,
    )

    print("\n[3/4] Creating Pareto front plot with selected controllers...")
    plot_pareto_with_selected(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )

    print("\n[4/4] Generating required videos...")
    generate_required_videos(
        checkpoint_dir=checkpoint_dir,
        controller_cls=controller_cls,
        output_dir=output_dir,
    )

    print("\nDone.")
    print(f"Submission outputs saved in: {output_dir}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace with your final checkpoint directory
    checkpoint_dir = "./results/20260402_184707_nsga_ckpts"

    # Replace with your actual controller class
    # e.g. CompatibleHybridAntController
    make_submission_outputs(
        checkpoint_dir=checkpoint_dir,
        controller_cls=CompatibleHybridAntController,
        output_dir=None,           # defaults to checkpoint_dir/submission_outputs
        n_eval_episodes=30,
    )