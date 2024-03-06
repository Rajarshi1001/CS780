

# CS780 Assignments

> Assignment 1
- Implementation of Bernoulli and Gaussian Bandit environment using **Gymnasium** library and simulating them for different combinations of hyper parameters
- Implementation of different learning strategies like `pureExploitation`, `pureExploration`, `epsilonGreedyExploration`, `decayingEpsilonGreedyExploration`, `softmaxExploration` and `UCBExploration` methods and their corresponding simulations on both environments along with tuning hyper parameters for different environments.
- Implementation of Random Walk Environment, creation of trajectory using `generateTrajectory` function for simulation
- Implementation of `MonteCarloPrediction` (both FVMC and EVMC) and `TemporalDifferencePrediction` for calculation of state values in the environment
- Plotting the evolution of state values over episodes, log scale episodes, seed averaged plots for effective noise removal
- Analysing the variation of target values for a particular state for the case of both environments

> Assignment 2
- Implementation of control algorithms like `MonteCarloControl`, `SARSAControl`, `Q learning`, `double Q learning`, `SARSA`($\lambda$) with eligibility traces, `Q`($\lambda$) with traces
- Implementation of model based algorithms like `Dyna-Q` and `Trajectory Sampling` for optimal policy calculation and values for each of the states in Random Maze Environment
- Comparison between different off-policy and on-policy control algorithms for this environment

> Midsem
- Implementation of Random Maze Environment and its simulations
- Implementation of `Policy Iteration` and `Value Iteration` for optimal policy calculation and values for each of the states in the environment and its comparative analyses.
- Implementation of `Monte Carlo`, `Temporal Difference-n step`, `TD`($\lambda$) algorithm for calculation of values for each states using optimal policies and its comparative analyses.


