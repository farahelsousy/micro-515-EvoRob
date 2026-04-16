# Multi-Objective Evolutionary Robotics (NSGA-II)

## NSGA-II Implementation Choices

We implemented NSGA-II to optimize locomotion of an Ant robot on two terrains: flat and ice. The controller was modified to a **sinusoidal oscillatory controller**, parameterized by amplitudes, phases, and a shared frequency, enabling periodic gait generation.

The algorithm used a population size of 100 over 100 generations. Elitism was applied by combining parent and offspring populations and selecting the best individuals using fast non-dominated sorting. Individuals were ranked into Pareto fronts, and diversity was maintained using crowding distance.

Parent selection was performed via tournament selection based on rank and crowding distance. Offspring were generated using a differential evolution-style operator, where individuals were perturbed using the difference of two randomly selected parents. Mutation probability was set to 1/n_params and crossover probability to 0.9.

## Observed Trade-offs in the Pareto Front

The resulting Pareto front shows a clear trade-off between flat and ice performance. Solutions optimized for flat terrain achieve higher forward speeds but tend to be less stable on ice. Conversely, solutions that perform well on ice exhibit more conservative and stable gaits but achieve lower speeds on flat terrain.

This demonstrates that improving performance in one environment generally leads to reduced performance in the other, resulting in a well-defined Pareto front with diverse trade-off solutions.


## Performance Comparison: Specialists vs Generalists

Flat specialists achieve the highest speed on flat terrain but show reduced robustness and higher variance on ice. Ice specialists are more stable and consistent on slippery terrain but sacrifice speed on flat surfaces.

The generalist, selected using a maximin criterion on normalized objectives, achieves balanced performance across both terrains. Although it does not outperform specialists on either terrain individually, it provides the best worst-case performance, making it the most robust solution overall.