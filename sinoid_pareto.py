import os
import json
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np

from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.sinoid import OscillatoryController


# ---------------------------------------------------------------------------
# Compatibility wrapper
# ---------------------------------------------------------------------------

class CompatibleOscillatoryAntController:
    """
    Thin compatibility wrapper so evaluation code can use the same interface
    as the other controllers.
    """

    def __init__(self, input_size=None, output_size=8):
        del input_size
        self.output_size = int(output_size)
        self.controller = OscillatoryController(output_size=self.output_size)
        self.n_params = self.controller.get_num_params()

    def get_action(self, state):
        return self.controller.get_action(state)

    def set_weights(self, weights):
        self.controller.set_weights(weights)

    def get_num_params(self):
        return self.controller.get_num_params()

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)

    def reset_controller(self, batch_size=1):
        if hasattr(self.controller, "reset_controller"):
            self.controller.reset_controller(batch_size=batch_size)
        elif hasattr(self.controller, "reset_model"):
            self.controller.reset_model()


# ---------------------------------------------------------------------------
# Load final population / fitness
# ---------------------------------------------------------------------------

def load_final_population_and_fitness(checkpoint_dir: str):
    """
    Load final population and fitness from an NSGA-II checkpoint directory.

    Supported:
    - full_x.npy + full_f.npy -> uses last generation
    - x.npy + f.npy
    - fallback to last generation subfolder
    """
    full_x_path = os.path.join(checkpoint_dir, "full_x.npy")
    full_f_path = os.path.join(checkpoint_dir, "full_f.npy")

    if os.path.isfile(full_x_path) and os.path.isfile(full_f_path):
        full_x = np.load(full_x_path, allow_pickle=True)
        full_f = np.load(full_f_path, allow_pickle=True)

        population = full_x[-1] if full_x.ndim == 3 else full_x
        fitness = full_f[-1] if full_f.ndim == 3 else full_f
        return np.asarray(population), np.asarray(fitness)

    x_path = os.path.join(checkpoint_dir, "x.npy")
    f_path = os.path.join(checkpoint_dir, "f.npy")

    if os.path.isfile(x_path) and os.path.isfile(f_path):
        population = np.load(x_path, allow_pickle=True)
        fitness = np.load(f_path, allow_pickle=True)
        return np.asarray(population), np.asarray(fitness)

    last_gen = get_last_checkpoint_dir(checkpoint_dir)
    if last_gen is not None:
        x_path = os.path.join(last_gen, "x.npy")
        f_path = os.path.join(last_gen, "f.npy")
        if os.path.isfile(x_path) and os.path.isfile(f_path):
            population = np.load(x_path, allow_pickle=True)
            fitness = np.load(f_path, allow_pickle=True)
            return np.asarray(population), np.asarray(fitness)

    raise FileNotFoundError(
        f"Could not find final population/fitness in '{checkpoint_dir}'."
    )


# ---------------------------------------------------------------------------
# Select flat specialist, ice specialist, generalist from Pareto front
# ---------------------------------------------------------------------------

def extract_specialist_and_generalist(
    checkpoint_dir: str,
    save_dir: Optional[str] = None,
):
    population, fitness = load_final_population_and_fitness(checkpoint_dir)

    if population.ndim != 2:
        raise ValueError(f"Expected population shape (N, D), got {population.shape}")
    if fitness.ndim != 2 or fitness.shape[1] != 2:
        raise ValueError(f"Expected fitness shape (N, 2), got {fitness.shape}")

    nsga = NSGAII(
        population_size=len(population),
        n_opt_params=population.shape[1],
    )
    fronts, _ = nsga.fast_nondominated_sort(fitness)

    if len(fronts) == 0 or len(fronts[0]) == 0:
        raise RuntimeError("Pareto front is empty.")

    pareto_indices = np.array(fronts[0], dtype=int)
    pareto_population = population[pareto_indices]
    pareto_fitness = fitness[pareto_indices].astype(np.float32)

    # Specialists: extremes on Pareto front
    flat_idx = int(np.argmax(pareto_fitness[:, 0]))
    ice_idx = int(np.argmax(pareto_fitness[:, 1]))

    # Generalist: maximin on normalized objectives
    fmin = pareto_fitness.min(axis=0)
    fmax = pareto_fitness.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-8, (fmax - fmin), 1.0)
    fit_norm = (pareto_fitness - fmin) / denom

    balanced_score = np.minimum(fit_norm[:, 0], fit_norm[:, 1])
    candidate_idxs = np.where(
        np.isclose(balanced_score, np.max(balanced_score))
    )[0]

    if len(candidate_idxs) > 1:
        avg_score = np.mean(fit_norm[candidate_idxs], axis=1)
        gen_idx = int(candidate_idxs[np.argmax(avg_score)])
    else:
        gen_idx = int(candidate_idxs[0])

    selected = {
        "pareto_indices": pareto_indices.tolist(),
        "flat_specialist": {
            "population_index": int(pareto_indices[flat_idx]),
            "pareto_local_index": int(flat_idx),
            "fitness": pareto_fitness[flat_idx].tolist(),
            "normalized_fitness": fit_norm[flat_idx].tolist(),
            "genotype": pareto_population[flat_idx],
        },
        "ice_specialist": {
            "population_index": int(pareto_indices[ice_idx]),
            "pareto_local_index": int(ice_idx),
            "fitness": pareto_fitness[ice_idx].tolist(),
            "normalized_fitness": fit_norm[ice_idx].tolist(),
            "genotype": pareto_population[ice_idx],
        },
        "generalist": {
            "population_index": int(pareto_indices[gen_idx]),
            "pareto_local_index": int(gen_idx),
            "fitness": pareto_fitness[gen_idx].tolist(),
            "normalized_fitness": fit_norm[gen_idx].tolist(),
            "balanced_score": float(balanced_score[gen_idx]),
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
                "normalized_fitness": selected["flat_specialist"]["normalized_fitness"],
            },
            "ice_specialist": {
                "population_index": selected["ice_specialist"]["population_index"],
                "pareto_local_index": selected["ice_specialist"]["pareto_local_index"],
                "fitness": selected["ice_specialist"]["fitness"],
                "normalized_fitness": selected["ice_specialist"]["normalized_fitness"],
            },
            "generalist": {
                "population_index": selected["generalist"]["population_index"],
                "pareto_local_index": selected["generalist"]["pareto_local_index"],
                "fitness": selected["generalist"]["fitness"],
                "normalized_fitness": selected["generalist"]["normalized_fitness"],
                "balanced_score": selected["generalist"]["balanced_score"],
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
    Returns stats + per-episode rewards + episode seeds.
    """
    env = AntFlatEnvironment(robot_path=robot_path)
    controller = controller_cls(input_size=27, output_size=8)

    rng = np.random.default_rng(seed)
    rewards = []
    episode_seeds = []

    controller.geno2pheno(genotype)

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        episode_seeds.append(ep_seed)

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
        "episode_seeds": episode_seeds,
    }


def record_controller_video_for_seed(
    controller_cls,
    genotype,
    robot_path: str,
    out_path: str,
    max_steps: int = 1000,
    seed: int = 0,
):
    """
    Record one controller on one terrain using a fixed seed.
    Returns rollout reward.
    """
    env = AntFlatEnvironment(render_mode="rgb_array", robot_path=robot_path)
    controller = controller_cls(input_size=27, output_size=8)
    controller.geno2pheno(genotype)

    obs, _ = env.reset(seed=seed)
    controller.reset_controller(batch_size=1)

    frames = []
    total_reward = 0.0

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    for _ in range(max_steps):
        action = controller.get_action(obs)
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.squeeze(0)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    env.close()

    if len(frames) == 0:
        print(f"Warning: no frames captured for {out_path}")
        return None

    imageio.mimwrite(out_path, frames, fps=20)
    print(f"Saved video: {out_path} | reward={total_reward:.2f} | seed={seed}")
    return float(total_reward)


# ---------------------------------------------------------------------------
# Comparison outputs
# ---------------------------------------------------------------------------

def evaluate_selected_controllers(
    checkpoint_dir: str,
    controller_cls,
    output_dir: str,
    n_episodes: int = 30,
):
    """
    Reevaluate flat specialist, ice specialist, and generalist on both terrains.
    Saves comparison table and JSON.
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
        results[ctrl_name] = {
            "population_index": selected[ctrl_name]["population_index"],
            "pareto_fitness": selected[ctrl_name]["fitness"],
            "normalized_pareto_fitness": selected[ctrl_name].get("normalized_fitness"),
        }

        if ctrl_name == "generalist":
            results[ctrl_name]["balanced_score"] = selected[ctrl_name].get("balanced_score")

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

        flat_mean = results[ctrl_name]["flat"]["mean"]
        ice_mean = results[ctrl_name]["ice"]["mean"]
        results[ctrl_name]["comparison_metrics"] = {
            "mean_of_means": float((flat_mean + ice_mean) / 2.0),
            "worst_case_mean": float(min(flat_mean, ice_mean)),
            "gap_between_means": float(abs(flat_mean - ice_mean)),
        }

    table_path = os.path.join(output_dir, "controller_comparison.txt")
    with open(table_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write("Controller Comparison: Specialists vs Generalist\n")
        f.write("=" * 120 + "\n\n")
        f.write(
            f"{'Controller':<20}"
            f"{'Flat mean':>12}"
            f"{'Flat std':>12}"
            f"{'Ice mean':>12}"
            f"{'Ice std':>12}"
            f"{'Mean':>12}"
            f"{'Min':>12}"
            f"{'Gap':>12}\n"
        )
        f.write("-" * 120 + "\n")

        for ctrl_name in ["flat_specialist", "ice_specialist", "generalist"]:
            flat_stats = results[ctrl_name]["flat"]
            ice_stats = results[ctrl_name]["ice"]
            cm = results[ctrl_name]["comparison_metrics"]

            f.write(
                f"{ctrl_name:<20}"
                f"{flat_stats['mean']:12.2f}"
                f"{flat_stats['std']:12.2f}"
                f"{ice_stats['mean']:12.2f}"
                f"{ice_stats['std']:12.2f}"
                f"{cm['mean_of_means']:12.2f}"
                f"{cm['worst_case_mean']:12.2f}"
                f"{cm['gap_between_means']:12.2f}\n"
            )

        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("Selected controller fitness from final Pareto front\n")
        f.write("=" * 120 + "\n")
        for ctrl_name in ["flat_specialist", "ice_specialist", "generalist"]:
            fit = results[ctrl_name]["pareto_fitness"]
            nfit = results[ctrl_name]["normalized_pareto_fitness"]
            f.write(
                f"{ctrl_name:<20} "
                f"flat={fit[0]:.2f}, ice={fit[1]:.2f}, "
                f"norm_flat={nfit[0]:.3f}, norm_ice={nfit[1]:.3f}"
            )
            if ctrl_name == "generalist":
                f.write(f", balanced_score={results[ctrl_name]['balanced_score']:.3f}")
            f.write("\n")

    print(f"Controller comparison saved to: {table_path}")

    json_path = os.path.join(output_dir, "controller_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

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

    nsga = NSGAII(
        population_size=len(population),
        n_opt_params=population.shape[1],
    )
    fronts, _ = nsga.fast_nondominated_sort(fitness)

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

    fs = np.array(selected["flat_specialist"]["fitness"], dtype=np.float32)
    ispec = np.array(selected["ice_specialist"]["fitness"], dtype=np.float32)
    gen = np.array(selected["generalist"]["fitness"], dtype=np.float32)

    ax.scatter(
        fs[0], fs[1],
        s=220, marker="*", label="Flat specialist",
        zorder=5, edgecolors="black"
    )
    ax.scatter(
        ispec[0], ispec[1],
        s=220, marker="*", label="Ice specialist",
        zorder=5, edgecolors="black"
    )
    ax.scatter(
        gen[0], gen[1],
        s=220, marker="*", label="Generalist (maximin)",
        zorder=5, edgecolors="black"
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

def generate_required_videos(
    checkpoint_dir: str,
    controller_cls,
    output_dir: str,
    n_video_eval_episodes: int = 20,
):
    """
    Generate required videos using the best rollout seed found over multiple episodes:
    - flat specialist on flat
    - ice specialist on ice
    - generalist on flat
    - generalist on ice
    """
    os.makedirs(output_dir, exist_ok=True)
    selected = extract_specialist_and_generalist(checkpoint_dir)

    video_jobs = [
        ("flat_specialist", "flat", "ant_flat_terrain.xml", "flat_specialist_on_flat.mp4"),
        ("ice_specialist", "ice", "ant_ice_terrain.xml", "ice_specialist_on_ice.mp4"),
        ("generalist", "flat", "ant_flat_terrain.xml", "generalist_on_flat.mp4"),
        ("generalist", "ice", "ant_ice_terrain.xml", "generalist_on_ice.mp4"),
    ]

    for ctrl_name, terrain_name, robot_path, filename in video_jobs:
        genotype = selected[ctrl_name]["genotype"]

        stats = evaluate_genotype_on_terrain(
            controller_cls=controller_cls,
            genotype=genotype,
            robot_path=robot_path,
            n_episodes=n_video_eval_episodes,
            max_episode_steps=1000,
            seed=0,
        )

        rewards = np.asarray(stats["rewards"], dtype=np.float32)
        seeds = stats["episode_seeds"]
        best_idx = int(np.argmax(rewards))
        best_seed = int(seeds[best_idx])
        best_reward = float(rewards[best_idx])

        print(
            f"[Video selection] {ctrl_name} on {terrain_name}: "
            f"best reward={best_reward:.2f}, seed={best_seed}"
        )

        record_controller_video_for_seed(
            controller_cls=controller_cls,
            genotype=genotype,
            robot_path=robot_path,
            out_path=os.path.join(output_dir, filename),
            max_steps=1000,
            seed=best_seed,
        )


# ---------------------------------------------------------------------------
# One-call submission output generator
# ---------------------------------------------------------------------------

def make_submission_outputs(
    checkpoint_dir: str,
    controller_cls,
    output_dir: Optional[str] = None,
    n_eval_episodes: int = 30,
    n_video_eval_episodes: int = 20,
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

    print("\n[4/4] Generating required videos with best-rollout selection...")
    generate_required_videos(
        checkpoint_dir=checkpoint_dir,
        controller_cls=controller_cls,
        output_dir=output_dir,
        n_video_eval_episodes=n_video_eval_episodes,
    )

    print("\nDone.")
    print(f"Submission outputs saved in: {output_dir}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    checkpoint_dir = "results/20260402_234428_oscillatory_nsga_ckpts"

    make_submission_outputs(
        checkpoint_dir=checkpoint_dir,
        controller_cls=CompatibleOscillatoryAntController,
        output_dir=None,
        n_eval_episodes=30,
        n_video_eval_episodes=20,
    )