# DD-DR-CVaR Motion Planning

This package contains code for solving a Data-Driven Distributionally-Robust CVaR-based Motion Planning problem.
The package uses CVXPy to model and solve the problem.

## Installation

- Download/Clone the package
- (Optional) Create a conda environment by running the following in terminal:
    ```
    conda create --name dr_cvar_safety_filtering python=3.10 pip -y
    conda activate dr_cvar_safety_filtering
    ```
- Install dependencies by running the following in terminal:
    ```
    pip install numpy==1.23.5 
    pip install scipy==1.9.3 
    pip install matplotlib==3.6.1
    pip install cvxpy==1.2.1
    pip install polytope==0.2.3
    pip install colorama==0.4.6
    ```
- Install the statistics package 
    ```
    cd statistics
    pip install -e .
    cd ..
    ```

## Architecture
We highlight the following files:
- `dynamic_vehicles.py`: defines classes for dynamic vehicles; i.e. vehicles with both geometry and a dynamics models
- `dynamics.py`: defines classes for the vehicle dynamics
- `geometry_vehicles.py`: defines classes for geometric vehicles
- `ref_traj_generation.py`: defines classes to generate reference trajectories for the ego vehicle
- `safe_halfspaces.py`: defines classes to obtain safe halfspaces
- `safety_filters.py`: defines classes for MPC-based safety filters

## Running the Scripts
The following scripts are useful for running the experiments:
- `drone_simulations.py`: runs the simulation
- `experiment_setup.py`: setup for experiment scenarios (intersection, head-on, multi-obstacles, ...)
- `multi_exp_sim.py`: runs Monte Carlo simulations and plots the results
- `paper_figures.py`: generates the halfspace comparison plots and the halfspace compute time plots
- `simulation_functions.py`: functions used to keep `drone_simulations.py` cleaner. Mostly includes plotting functions.

### Running simulations and generating experiment data:
- To plot paper supporting figures (non-experiment figures), run `paper_figures.py`.
- To set up a new type of experiment, edit the `experiment_setup.py` file
- To run a single simulation, edit the setting of the `reach_avoid` function in `drone_simulations.py` and run the script
- To run Monte carlo simulations, edit the toggles and ranges at the top of `multi_exp_sim.py` and run the script
- To plot experiment data
  - For single experiments, set the plotting variables in `drone_simulations.py` to `True`
  - For multiple experiments, set the plotting variable in `multi_exp_sim.py` to `True`
