import numpy as np

from evorob.algorithms.nsga import NSGAII
from evorob.world.robot.controllers.mlp_hebbian import HebbianController

# Reuse your existing world/evaluation helpers from Challenge2
from Challenge2 import AntMultiWorld, test_exercise_implementation


class CompatibleHebbianAntController(HebbianController):
    """
    Standalone Hebbian controller wrapper for the Ant task.

    No PPO base.
    No oscillator.
    No residual branch.

    This is a pure Hebbian/plastic controller evolved by NSGA-II.
    """

    HIDDEN_SIZE = 16

    def __init__(self, input_size, output_size):
        super().__init__(
            input_size=int(input_size),
            output_size=int(output_size),
            hidden_size=int(self.HIDDEN_SIZE),
        )
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n_params = self.get_num_params()


def run_evolution_nsga_hebbian(
    num_generations=50,
    population_size=80,
    n_parents=30,
    n_repeats=6,
    mutation_prob=0.01,
    crossover_prob=0.75,
    bounds=(-0.03, 0.03),
    ckpt_interval=10,
    compute_score=True,
    random_seed=42,
):
    """
    Run a multi-objective NSGA-II experiment using the Hebbian controller.
    """

    print("\n" + "=" * 70)
    print("           MULTI-OBJECTIVE EVOLUTION - NSGA-II (HEBBIAN)           ")
    print("=" * 70)
    print(
        f"Population: {population_size} | Generations: {num_generations} | Parents: {n_parents}"
    )
    print("Objective 1: Flat Terrain Speed | Objective 2: Ice Terrain Speed")
    print("=" * 70)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Build the evaluation world using the Hebbian controller
    world = AntMultiWorld(
        controller_cls=CompatibleHebbianAntController,
        n_repeats=n_repeats,
    )

    # Print controller info once
    demo_ctrl = CompatibleHebbianAntController(input_size=27, output_size=8)
    print(
        f"Controller: CompatibleHebbianAntController | Parameters: {demo_ctrl.n_params}"
    )
    print()

    # Create NSGA-II optimizer
    nsga = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
    )

    # Initial population
    population = nsga.ask()

    for generation in range(num_generations):
        fitness = np.zeros((population_size, 2), dtype=np.float32)

        # Evaluate all individuals
        for i, individual in enumerate(population):
            fitness[i] = world.evaluate_individual(individual)

        save_ckpt = ((generation + 1) % ckpt_interval == 0)
        nsga.tell(population, fitness, save_checkpoint=save_ckpt)

        # Logging
        best_obj1 = float(np.max(fitness[:, 0]))
        best_obj2 = float(np.max(fitness[:, 1]))
        mean_obj1 = float(np.mean(fitness[:, 0]))
        mean_obj2 = float(np.mean(fitness[:, 1]))
        std_obj1 = float(np.std(fitness[:, 0]))
        std_obj2 = float(np.std(fitness[:, 1]))
        mean_total = float(np.mean(np.sum(fitness, axis=1)))
        std_total = float(np.std(np.sum(fitness, axis=1)))

        if generation == 0 or (generation + 1) % 5 == 0:
            print(f"Generation {generation}:   {nsga.f_best_so_far}")
            print(f"Mean fitness:   {mean_total:.2f} +- {std_total:.2f}")
            print(
                f"Mean fitness per obj: "
                f"['{mean_obj1:.2f} +- {std_obj1:.2f}', "
                f"'{mean_obj2:.2f} +- {std_obj2:.2f}']"
            )

        progress = 100.0 * (generation + 1) / num_generations
        bar_len = 50
        filled = int(bar_len * (generation + 1) / num_generations)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"Gen {generation + 1:4d}/{num_generations} [{bar}] {progress:5.1f}%\n"
            f"     Best:     Objective 1={best_obj1:7.2f}  Objective 2={best_obj2:7.2f}\n"
            f"     Mean:     Objective 1={mean_obj1:7.2f}  Objective 2={mean_obj2:7.2f}\n"
        )

        if generation < num_generations - 1:
            population = nsga.ask()

    if compute_score:
        print("\nFinished Hebbian NSGA-II run.")

    return nsga


if __name__ == "__main__":
    test_exercise_implementation()

    run_evolution_nsga_hebbian(
        num_generations=50,
        population_size=80,
        n_parents=30,
        n_repeats=6,
        mutation_prob=0.01,
        crossover_prob=0.75,
        bounds=(-0.03, 0.03),
        ckpt_interval=10,
        compute_score=True,
        random_seed=42,
    )