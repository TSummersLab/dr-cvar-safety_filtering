from drone_simulations import reach_avoid
from simulation_functions import *

# set the value of the chosen setting to True to generate the Monte Carlo simulations
monte_carlo_gauss = True
monte_carlo_guass_seed_range = [1, 301]  # if [s, e] --> s, s+1, ..., e-1
monte_carlo_lap = False
monte_carlo_lap_seed_range = [1001, 1501]
# plotting settings
monte_carlo_plots = True
plots_seed_range = [1, 301]
# plots_seed_range = [1001, 1501]

# Gaussian noise samples, single obstacle: seed range 1->1000
if monte_carlo_gauss:
    for seed in range(*monte_carlo_guass_seed_range):
        reach_avoid(seed=seed, exp_type='ego_intersect', metric='drcvar', filter_slack=False,
                    samp_dist='norm', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_intersect', metric='cvar', filter_slack=False,
                    samp_dist='norm', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_intersect', metric='mean', filter_slack=False,
                    samp_dist='norm', realize_dist='lap')

    for seed in range(*monte_carlo_guass_seed_range):
        reach_avoid(seed=seed, exp_type='ego_overtaking', metric='drcvar', filter_slack=False,
                    samp_dist='norm', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_overtaking', metric='cvar', filter_slack=False,
                    samp_dist='norm', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_overtaking', metric='mean', filter_slack=False,
                    samp_dist='norm', realize_dist='lap')

    for seed in range(*monte_carlo_guass_seed_range):
        reach_avoid(seed=seed, exp_type='ego_headon', metric='drcvar', filter_slack=False,
                    samp_dist='norm', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_headon', metric='cvar', filter_slack=False,
                    samp_dist='norm', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_headon', metric='mean', filter_slack=False,
                    samp_dist='norm', realize_dist='lap')


# Gaussian noise samples, single obstacle: seed range 1001->2000
if monte_carlo_lap:
    for seed in range(*monte_carlo_lap_seed_range):
        reach_avoid(seed=seed, exp_type='ego_intersect', metric='drcvar', filter_slack=False,
                    samp_dist='lap', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_intersect', metric='cvar', filter_slack=False,
                    samp_dist='lap', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_intersect', metric='mean', filter_slack=False,
                    samp_dist='lap', realize_dist='lap')

    for seed in range(*monte_carlo_lap_seed_range):
        reach_avoid(seed=seed, exp_type='ego_overtaking', metric='drcvar', filter_slack=False,
                    samp_dist='lap', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_overtaking', metric='cvar', filter_slack=False,
                    samp_dist='lap', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_overtaking', metric='mean', filter_slack=False,
                    samp_dist='lap', realize_dist='lap')

    for seed in range(*monte_carlo_lap_seed_range):
        reach_avoid(seed=seed, exp_type='ego_headon', metric='drcvar', filter_slack=False,
                    samp_dist='lap', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_headon', metric='cvar', filter_slack=False,
                    samp_dist='lap', realize_dist='lap')
        reach_avoid(seed=seed, exp_type='ego_headon', metric='mean', filter_slack=False,
                    samp_dist='lap', realize_dist='lap')


if monte_carlo_plots:
    plot_saved_dist_to_col(exp_type='ego_headon', filter_slack=False, seed_range=plots_seed_range, start=6, end=9, add_legend=False)
    plot_saved_dist_to_col(exp_type='ego_overtaking', filter_slack=False, seed_range=plots_seed_range, start=3, end=6, add_legend=False)
    plot_saved_dist_to_col(exp_type='ego_intersect', filter_slack=False, seed_range=plots_seed_range, start=6, end=9, add_legend=True)

    # plot_saved_traj(exp_type='ego_headon', filter_slack=False, seed=plots_seed_range[0])
    # plot_saved_traj(exp_type='ego_overtaking', filter_slack=False, seed=plots_seed_range[0])
    # plot_saved_traj(exp_type='ego_intersect', filter_slack=False, seed=plots_seed_range[0])

    plot_saved_traj_overlayed(exp_type='ego_headon', filter_slack=False, seed=plots_seed_range[0], show=False,
                              xlim=(-5.2, 4.5), ylim=(-2, 2), add_legend=False, figsize=None)
    plot_saved_traj_overlayed(exp_type='ego_overtaking', filter_slack=False, seed=plots_seed_range[0], show=False,
                              xlim=(-5.2, 4.5), ylim=(-2, 2), add_legend=False, figsize=None)
    plot_saved_traj_overlayed(exp_type='ego_intersect', filter_slack=False, seed=plots_seed_range[0], show=False,
                              xlim=(-5, 3), ylim=(-4, 2), add_legend=False, figsize=None)

