import os
import json
import numpy as np
import matplotlib.pyplot as plt

from Challenge3 import AntWorld
from evorob.world.robot.controllers.mlp_sol import NeuralNetworkController


# ============================================================
# SET THIS TO YOUR FINAL NSGA-II FOLDER
# ============================================================
RESULTS_DIR = "/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/AntHill-v0/multi/100"


# ============================================================
# Helpers
# ============================================================
def decode_body(genotype: np.ndarray) -> np.ndarray:
    """Convert last 8 genes to physical leg lengths."""
    return (genotype[-8:] + 1) / 4 + 0.1


def morphology_dict(genotype: np.ndarray) -> dict:
    body = decode_body(genotype)
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


def select_individuals(X: np.ndarray, F: np.ndarray):
    """
    Select:
    - specialist 1: best objective 1
    - specialist 2: best objective 2
    - generalist: best normalized sum
    """
    spec1_idx = int(np.argmax(F[:, 0]))
    spec2_idx = int(np.argmax(F[:, 1]))

    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-8)
    gen_idx = int(np.argmax(F_norm[:, 0] + F_norm[:, 1]))

    return spec1_idx, spec2_idx, gen_idx


def save_video(world: AntWorld, genotype: np.ndarray, out_path: str, n_steps: int = 500):
    world.update_robot_xml(genotype)
    env = world.create_env(max_episode_steps=-1)
    try:
        world.generate_best_individual_video(env, video_name=out_path, n_steps=n_steps)
    finally:
        env.close()


def save_pareto_plot(F: np.ndarray, spec1_idx: int, spec2_idx: int, gen_idx: int, out_path: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(F[:, 0], F[:, 1], alpha=0.7, label="Population")

    plt.scatter(
        F[spec1_idx, 0], F[spec1_idx, 1],
        color="red", s=120, marker="o", label="Specialist obj1"
    )
    plt.scatter(
        F[spec2_idx, 0], F[spec2_idx, 1],
        color="green", s=120, marker="o", label="Specialist obj2"
    )
    plt.scatter(
        F[gen_idx, 0], F[gen_idx, 1],
        color="blue", s=120, marker="o", label="Generalist"
    )

    plt.xlabel("Objective 1: forward + healthy")
    plt.ylabel("Objective 2: efficiency (-ctrl_cost)")
    plt.title("Final Pareto Front")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_morphology_report(
    X: np.ndarray,
    F: np.ndarray,
    spec1_idx: int,
    spec2_idx: int,
    gen_idx: int,
    out_txt: str,
    out_json: str,
):
    labels = {
        "specialist_obj1": spec1_idx,
        "specialist_obj2": spec2_idx,
        "generalist": gen_idx,
    }

    report = {}
    for name, idx in labels.items():
        report[name] = {
            "index": int(idx),
            "objectives": {
                "obj1_forward_healthy": float(F[idx, 0]),
                "obj2_efficiency": float(F[idx, 1]),
            },
            "body_lengths": morphology_dict(X[idx]),
        }

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    with open(out_txt, "w") as f:
        f.write("Morphology Comparison\n")
        f.write("=" * 60 + "\n\n")
        for name, item in report.items():
            f.write(f"{name}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Index: {item['index']}\n")
            f.write(
                f"Objectives: obj1={item['objectives']['obj1_forward_healthy']:.4f}, "
                f"obj2={item['objectives']['obj2_efficiency']:.4f}\n"
            )
            for k, v in item["body_lengths"].items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")


# ============================================================
# Main
# ============================================================
def main():
    x_path = os.path.join(RESULTS_DIR, "x.npy")
    f_path = os.path.join(RESULTS_DIR, "f.npy")
    xbest_path = os.path.join(RESULTS_DIR, "x_best.npy")

    if not os.path.isfile(x_path):
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not os.path.isfile(f_path):
        raise FileNotFoundError(f"Missing file: {f_path}")
    if not os.path.isfile(xbest_path):
        print(f"Warning: x_best.npy not found at {xbest_path}")

    X = np.load(x_path, allow_pickle=True)
    F = np.load(f_path, allow_pickle=True)

    X = np.asarray(X)
    F = np.asarray(F)

    print("Loaded:")
    print("X shape:", X.shape)
    print("F shape:", F.shape)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")
    if F.ndim != 2 or F.shape[1] < 2:
        raise ValueError(f"Expected F to have shape (N,2), got {F.shape}")

    spec1_idx, spec2_idx, gen_idx = select_individuals(X, F)

    print("\nSelected individuals:")
    print(f"Specialist obj1 index: {spec1_idx}, objectives: {F[spec1_idx]}")
    print(f"Specialist obj2 index: {spec2_idx}, objectives: {F[spec2_idx]}")
    print(f"Generalist index:      {gen_idx}, objectives: {F[gen_idx]}")

    spec1 = X[spec1_idx]
    spec2 = X[spec2_idx]
    generalist = X[gen_idx]

    # Save selected genotypes
    np.save(os.path.join(RESULTS_DIR, "specialist_obj1.npy"), spec1)
    np.save(os.path.join(RESULTS_DIR, "specialist_obj2.npy"), spec2)
    np.save(os.path.join(RESULTS_DIR, "generalist.npy"), generalist)

    # Save Pareto plot
    pareto_path = os.path.join(RESULTS_DIR, "pareto_front.png")
    save_pareto_plot(F, spec1_idx, spec2_idx, gen_idx, pareto_path)
    print(f"\nSaved Pareto front plot to: {pareto_path}")

    # Save morphology comparison
    morph_txt = os.path.join(RESULTS_DIR, "morphology_comparison.txt")
    morph_json = os.path.join(RESULTS_DIR, "morphology_comparison.json")
    save_morphology_report(X, F, spec1_idx, spec2_idx, gen_idx, morph_txt, morph_json)
    print(f"Saved morphology report to: {morph_txt}")
    print(f"Saved morphology JSON to:   {morph_json}")

    # Print quick morphology summary
    print("\nDecoded body lengths:")
    print("Specialist obj1:", morphology_dict(spec1))
    print("Specialist obj2:", morphology_dict(spec2))
    print("Generalist:     ", morphology_dict(generalist))

    # IMPORTANT: use same controller type as NSGA-II run
    world = AntWorld()
    world.controller = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=8,
    )
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params

    print("\nGenerating videos...")
    save_video(world, spec1, os.path.join(RESULTS_DIR, "specialist_forward.mp4"))
    save_video(world, spec2, os.path.join(RESULTS_DIR, "specialist_efficiency.mp4"))
    save_video(world, generalist, os.path.join(RESULTS_DIR, "generalist.mp4"))
    print("Done.")

    print("\nFiles ready:")
    print("- pareto_front.png")
    print("- specialist_forward.mp4")
    print("- specialist_efficiency.mp4")
    print("- generalist.mp4")
    print("- morphology_comparison.txt")
    print("- morphology_comparison.json")
    print("- specialist_obj1.npy")
    print("- specialist_obj2.npy")
    print("- generalist.npy")


if __name__ == "__main__":
    main()