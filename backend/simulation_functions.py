"""
Helper functions for the simulations

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
July 24, 2023
"""
import os

import numpy as np
from matplotlib import pyplot as plt

from printing_fxns import print_colored
from paper_figures import MAIN_TITLE, SUBPLOT_TITLE, LEGEND, XYAXIS
import pickle


def plot_sim_trajs(num_obst: int, env_limit: float, safe_polys_all: list,
                   ego_color: str, ego_pos_all: list, ego_patches_all: list,
                   obst_colors_list: list, obst_pos_all: list, obst_patches_all: list,
                   show=True, save=True, save_file_name='',
                   xlim=None, ylim=None, figsize=(8, 8)):
    """
    Plot the simulation trajectories
    :param num_obst: number of obstacles
    :param env_limit: environment limit which defines the square x-y bounds between -env_limit and env_limit
    :param safe_polys_all: list of polytope.Polytope polytopes representing the safe space
    :param ego_color: ego vehicle color
    :param ego_pos_all: list of ego vehicle positions
    :param ego_patches_all: list of matplotlib patches representing the ego vehicle
    :param obst_colors_list: list of obstacle vehicle colors
    :param obst_pos_all: list of obstacle vehicle positions
    :param obst_patches_all: list of matplotlib patches representing the obstacle vehicles
    :param show: True --> show the plot. False --> don't show it
    :param save: True --> save the plot. False --> don't save it
    :param save_file_name: save file name (.png appended to it)
    :param xlim: x limits
    :param ylim: y limits
    :param figsize: figure size tuple
    :return:
    """

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    # plot the safe halfspaces
    for p in safe_polys_all:
        p.plot(ax=ax, color='g', alpha=0.05, linestyle='solid', linewidth=1)

    # plot the ego polytopes
    for pe in ego_patches_all:
        ax.add_patch(pe)

    # plot all obstacle polytopes
    for obst_polys_t in obst_patches_all:
        for po in obst_polys_t:
            ax.add_patch(po)

    # scatter plot of the ego trajectory
    ego_pos_all_array = np.array(ego_pos_all).T
    e = plt.scatter(ego_pos_all_array[0, :], ego_pos_all_array[1, :], color=ego_color)
    for t in range(len(obst_pos_all)):  # add timestep text
        plt.text(ego_pos_all_array[0, t], ego_pos_all_array[1, t], str(t), color=ego_color, fontsize=12)

    # scatter plot of all obstacle trajectories
    o = []
    for ob_num in range(num_obst):
        ob_traj = np.zeros([len(obst_pos_all), 2])
        color = obst_colors_list[ob_num % len(obst_colors_list)]
        for t in range(len(obst_pos_all)):
            ob_traj[t] = obst_pos_all[t][ob_num]
        o.append(plt.scatter(ob_traj[:, 0], ob_traj[:, 1], color=color))
        for t in range(len(obst_pos_all)):  # add timestep text
            plt.text(ob_traj[t, 0], ob_traj[t, 1], str(t), color=color, fontsize=12)

    # legend
    legend = [e]
    legend.extend(o)
    labels = ['ego']
    labels.extend(['ob{}'.format(i) for i in range(num_obst)])
    plt.legend(legend, labels, ncol=2, loc='upper center', fontsize=16)

    # figure size
    ax.axis('equal')
    if xlim is None:
        plt.xlim([-(env_limit + 0.5), (env_limit + 0.5)])
    else:
        plt.xlim(xlim)
    if ylim is None:
        plt.ylim([-(env_limit + 0.5), (env_limit + 0.5)])
    else:
        plt.ylim(ylim)
    ax.tick_params(axis='both', labelsize=16)

    if save:
        plt.savefig(save_file_name)
    if show:
        plt.show()
    plt.close()


def plot_sim_cvx_pb_data(mpc_filter_time_info: list, mpc_filter_status: list, safe_hs_time_info: list,
                         obst_colors_list: list, show=True, save=True, save_file_name=''):
    sim_steps = len(mpc_filter_time_info)
    sim_time = range(sim_steps)
    horizon, num_obst = safe_hs_time_info[0].shape

    fig, axs = plt.subplots(3, figsize=(8, 8), sharex=True)  # define the figure
    axs[0].set_title('MPC Call Time', fontsize=SUBPLOT_TITLE)
    axs[0].set_ylabel('Time (ms)', fontsize=XYAXIS)
    axs[0].tick_params(axis='y', labelsize=XYAXIS)
    axs[1].set_title('MPC Status', fontsize=SUBPLOT_TITLE)
    axs[1].set_ylabel('Status', fontsize=XYAXIS)
    axs[1].tick_params(axis='y', labelsize=XYAXIS)
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['failed', 'solved'], rotation=40)
    axs[2].set_title('Safe Halfspace Call Time (Cumulative)', fontsize=SUBPLOT_TITLE)
    # axs[2].set_xticks(sim_time)
    # axs[2].set_xticklabels(sim_time)
    axs[2].set_ylabel('Time (ms)', fontsize=XYAXIS)
    axs[2].tick_params(axis='y', labelsize=XYAXIS)
    axs[2].set_xlabel('Time Step', fontsize=XYAXIS)

    mpc_filter_time_info = np.squeeze(mpc_filter_time_info)
    axs[0].plot(sim_time, mpc_filter_time_info)
    axs[0].fill_between(sim_time, mpc_filter_time_info, alpha=0.2)
    print('MPC Filter Call time (ms) (min, mean, max): ({}, {}, {}) times'.format(
        np.min(mpc_filter_time_info), np.mean(mpc_filter_time_info), np.max(mpc_filter_time_info)))

    mpc_filter_status = np.squeeze(mpc_filter_status)
    axs[1].plot(sim_time, mpc_filter_status)
    axs[1].fill_between(sim_time, mpc_filter_status, alpha=0.2)
    print('MPC Filter Failed: {}/{} times'.format(len(mpc_filter_status) - np.count_nonzero(mpc_filter_status), len(mpc_filter_status)))

    cumulative_times = [np.zeros([sim_steps, 1]) for _ in range(num_obst)]
    for sim_step in range(sim_steps):
        for ob_num in range(num_obst):
            t = np.sum(safe_hs_time_info[sim_step][:, ob_num])
            if ob_num == 0:
                cumulative_times[ob_num][sim_step] = t
            else:
                cumulative_times[ob_num][sim_step] = cumulative_times[ob_num-1][sim_step] + t
    for ob_num in range(num_obst):
        color = obst_colors_list[ob_num % len(obst_colors_list)]
        axs[2].plot(sim_time, cumulative_times[ob_num], color=color)
        if ob_num == 0:
            axs[2].fill_between(sim_time, np.squeeze(cumulative_times[ob_num]), color=color, alpha=0.2)
        else:
            axs[2].fill_between(sim_time, np.squeeze(cumulative_times[ob_num]), np.squeeze(cumulative_times[ob_num-1]),
                                color=color, alpha=0.2)
        print('Halfspace Cumulative Call Time (ms) with {} obstacles (min, mean, max) = ({}, {}, {})'.format(
            ob_num+1, np.min(np.squeeze(cumulative_times[ob_num])),
            np.mean(np.squeeze(cumulative_times[ob_num])),
            np.max(np.squeeze(cumulative_times[ob_num]))
        ))

    if save:
        plt.savefig(save_file_name)
    if show:
        plt.show()
    plt.close()


def plot_ego_obst_dist_to_collision(ego_obst_dist_to_col: list, obst_colors_list: list,
                                    show=True, save=True, save_file_name=''):
    ego_obst_dist_to_col = np.array(ego_obst_dist_to_col)  # sim_steps x num_obst
    sim_steps, num_obst = ego_obst_dist_to_col.shape
    sim_time = range(sim_steps)
    plt.plot([0, sim_steps], [0, 0], color='k')
    for ob_num in range(num_obst):
        color = obst_colors_list[ob_num % len(obst_colors_list)]
        plt.plot(sim_time, ego_obst_dist_to_col[:, ob_num], color=color)
    plt.ylabel('Distance to collision', fontsize=XYAXIS)
    plt.xlabel('Timestep', fontsize=XYAXIS)
    plt.yticks(fontsize=XYAXIS)
    plt.xticks(fontsize=XYAXIS)
    plt.ylim([min(0, np.min(ego_obst_dist_to_col))-0.1, np.max(ego_obst_dist_to_col)+0.1])
    # plt.title('Distance to collision', fontsize=MAIN_TITLE)
    if save:
        plt.savefig(save_file_name)
    if show:
        plt.show()
    plt.close()


def get_obst_pos_and_patches(obst_vehicles: list, obst_colors_list: list, obst_alpha):
    obst_pos_now, obst_patches_now = [], []
    for i, obst_veh in enumerate(obst_vehicles):
        obst_pos_now.append(obst_veh.get_pos)
        obst_patches_now.append(obst_veh.get_patch(obst_colors_list[i % len(obst_colors_list)], obst_alpha))
    return obst_pos_now, obst_patches_now


def apply_ego_control(mpc_filter, ego_ref_traj, ego_veh, verbose=False):
    x_opt = mpc_filter.x_mpc
    ctrl = mpc_filter.u_mpc[:, 0]
    ego_veh.step(ctrl)
    if verbose:
        diff = (x_opt - ego_ref_traj)
        correction = sum([np.linalg.norm(diff[:, t + 1]) for t in range(mpc_filter.horizon)])
        print_colored('Ctrl: ' + str(ctrl), 'c')
        print_colored('Error reaching next step: ' + str(ego_veh.get_pos - mpc_filter.dyn_model.C @ x_opt[:, 1]), 'c')
        print_colored('Final state error: ' + str(np.linalg.norm(ego_ref_traj[:, -1] - x_opt[:, -1])), 'c')
        print_colored('Horizon state correction: ' + str(correction), 'g')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def plot_saved_dist_to_col(exp_type, filter_slack, seed_range, start, end, add_legend=True):
    start -= 1
    colors = ["tab:red", "tab:blue", "tab:green"]
    metrics = ['mean', 'cvar', 'drcvar']
    min_dist = [[] for _ in metrics]
    max_dist = [[] for _ in metrics]
    all_dist = [[] for _ in metrics]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        path = os.path.join("exp_data", exp_type, metric, "slack={}".format(filter_slack))
        print('Metric {}. Seed where dist to col < 0:'.format(metric))
        for seed in range(seed_range[0], seed_range[1]):
            data_dict = pickle.load(open(os.path.join(path, 'seed={}.pkl'.format(seed)), 'rb'))
            ego_obst_dist_to_col = data_dict["ego_obst_dist_to_col"]
            ego_obst_dist_to_col = np.array(ego_obst_dist_to_col)  # sim_steps x num_obst
            sim_steps, num_obst = ego_obst_dist_to_col.shape
            sim_time = range(1, sim_steps+1)
            for ob_num in range(num_obst):
                if len(min_dist[i]) == 0:
                    min_dist[i] = np.array(ego_obst_dist_to_col[:, ob_num])
                    max_dist[i] = np.array(ego_obst_dist_to_col[:, ob_num])
                    all_dist[i] = [list(ego_obst_dist_to_col[:, ob_num])]
                else:
                    min_dist[i] = np.minimum(min_dist[i], np.array(ego_obst_dist_to_col[:, ob_num]))
                    max_dist[i] = np.maximum(max_dist[i], np.array(ego_obst_dist_to_col[:, ob_num]))
                    all_dist[i].append(list(ego_obst_dist_to_col[:, ob_num]))
                if np.min(np.array(ego_obst_dist_to_col[:, ob_num])) < 0:
                    print('Seed {} results in a collision with obstacle #{}'.format(seed, ob_num))
                plt.plot(sim_time, ego_obst_dist_to_col[:, ob_num], color=color, alpha=0.2)
        print("--------------------")
    plt.plot([1, sim_steps], [0, 0], color='k')
    plt.ylabel('Distance to collision', fontsize=XYAXIS)
    # plt.title('Distance to collision', fontsize=MAIN_TITLE)
    plt.xlabel('Timestep', fontsize=XYAXIS)
    plt.yticks(fontsize=XYAXIS)
    plt.xticks(fontsize=XYAXIS)
    plt.tight_layout()
    plt.savefig(os.path.join('exp_data', exp_type, 'dist_to_col_multi_exp' + '.png'))
    plt.show()

    for i, (metric, color, min_d, max_d) in enumerate(zip(metrics, colors, min_dist, max_dist)):
        plt.fill_between(sim_time, np.squeeze(max_d), np.squeeze(min_d), color=color, alpha=0.3)
    plt.plot([1, sim_steps], [0, 0], color='k')

    # plt.title('Distance to collision', fontsize=MAIN_TITLE)
    plt.ylabel('Distance to collision', fontsize=XYAXIS)
    plt.xlabel('Timestep', fontsize=XYAXIS)
    plt.yticks(fontsize=XYAXIS)
    plt.xticks(fontsize=XYAXIS)
    plt.tight_layout()
    plt.savefig(os.path.join('exp_data', exp_type, 'dist_to_col_multi_exp_ranges' + '.png'))
    plt.show()

    fig, ax = plt.subplots()
    ax.plot([0, end-start], [0, 0], color='k')
    bps = []
    group_width = 0.2
    legend = []
    for i, (metric, color, all_d) in enumerate(zip(metrics, colors, all_dist)):
        ad = np.array(all_d)
        ad = ad[:, start:end]
        group_positions = np.arange(ad.shape[1]) + i * group_width
        bp = ax.boxplot(ad, positions=group_positions, patch_artist=True, widths=group_width)
        bps.append(bp)
        legend.append(bp["boxes"][0])
    for bp, color in zip(bps, colors):
        for patch in bp['boxes']:
            patch.set_facecolor(color)

    group_labels = [f'{i + 1}' for i in range(start, end)]
    ax.set_xticks(np.arange(ad.shape[1]) + (len(metrics) - 1) * group_width / 2)
    ax.set_xticklabels(group_labels)
    ax.set_xlabel('Timestep', fontsize=XYAXIS)
    ax.set_ylabel('Distance to collision', fontsize=XYAXIS)
    # ax.set_title('Distance to collision', fontsize=MAIN_TITLE)
    ax.tick_params(axis='both', labelsize=XYAXIS)

    if add_legend:
        plt.legend(legend, metrics, fontsize=LEGEND, loc='upper center')
    plt.tight_layout()
    plt.savefig(os.path.join('exp_data', exp_type, 'dist_to_col_multi_exp_box_plots' + '.png'))
    plt.show()


def plot_saved_traj(exp_type, filter_slack, seed):
    metrics = ['mean', 'cvar', 'drcvar']
    for i, metric in enumerate(metrics):
        path = os.path.join("exp_data", exp_type, metric, "slack={}".format(filter_slack))
        data_dict = pickle.load(open(os.path.join(path, 'seed={}.pkl'.format(seed)), 'rb'))
        num_obst = data_dict["num_obst"]
        ENV_LIM = data_dict["ENV_LIM"]
        safe_polys_all = data_dict["safe_polys_all"]
        EGO_COLOR = data_dict["EGO_COLOR"]
        ego_pos_all = data_dict["ego_pos_all"]
        ego_polys_all = data_dict["ego_polys_all"]
        OBST_COLORS = data_dict["OBST_COLORS"]
        obst_pos_all = data_dict["obst_pos_all"]
        obst_polys_all = data_dict["obst_polys_all"]
        fig_name = os.path.join('exp_data', exp_type, metric, "slack={}".format(filter_slack),
                                'traj_seed={}'.format(seed) + '.png')
        plot_sim_trajs(num_obst, ENV_LIM, safe_polys_all,
                       EGO_COLOR, ego_pos_all, ego_polys_all,
                       OBST_COLORS, obst_pos_all, obst_polys_all,
                       show=True, save=True, save_file_name=fig_name)


def plot_saved_traj_overlayed(exp_type, filter_slack, seed, show=True, xlim=None, ylim=None, figsize=(8, 8), add_legend=True):
    metrics = ['mean', 'cvar', 'drcvar']
    ego_colors = ['tab:red', 'tab:blue', 'tab:green']
    ob_color = 'k'
    # fig, ax = plt.subplots()  # define the figure
    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)
    legend = []
    for i, (metric, color) in enumerate(zip(metrics, ego_colors)):
        path = os.path.join("exp_data", exp_type, metric, "slack={}".format(filter_slack))
        data_dict = pickle.load(open(os.path.join(path, 'seed={}.pkl'.format(seed)), 'rb'))
        num_obst = data_dict["num_obst"]
        env_limit = data_dict["ENV_LIM"]
        ego_pos_all = data_dict["ego_pos_all"]
        ego_patches_all = data_dict["ego_polys_all"]
        obst_pos_all = data_dict["obst_pos_all"]
        obst_patches_all = data_dict["obst_polys_all"]
        fig_name = os.path.join('exp_data', exp_type, 'traj_seed={}'.format(seed) + '.png')

        if i == 0:
            # plot all obstacle polytopes
            for obst_polys_t in obst_patches_all:
                for po in obst_polys_t:
                    po.set_facecolor(ob_color)
                    po.set_edgecolor(ob_color)
                    ax.add_patch(po)

            # scatter plot of all obstacle trajectories
            o = []
            for ob_num in range(num_obst):
                ob_traj = np.zeros([len(obst_pos_all), 2])
                for t in range(len(obst_pos_all)):
                    ob_traj[t] = obst_pos_all[t][ob_num]
                o.append(plt.scatter(ob_traj[:, 0], ob_traj[:, 1], color=ob_color))
                for t in range(len(obst_pos_all)):  # add timestep text
                    plt.text(ob_traj[t, 0], ob_traj[t, 1], str(t), color=ob_color, fontsize=12)
            legend.extend(o)

        # plot the ego polytopes
        for pe in ego_patches_all:
            pe.set_facecolor(color)
            pe.set_edgecolor(color)
            ax.add_patch(pe)

        # scatter plot of the ego trajectory
        ego_pos_all_array = np.array(ego_pos_all).T
        e = plt.scatter(ego_pos_all_array[0, :], ego_pos_all_array[1, :], color=color)
        for t in range(len(obst_pos_all)):  # add timestep text
            plt.text(ego_pos_all_array[0, t], ego_pos_all_array[1, t], str(t), color=color, fontsize=12)

        legend.append(e)

    # legend
    labels = []
    labels.extend(['ob{}'.format(i) for i in range(num_obst)])
    labels.extend(metrics)
    if add_legend:
        plt.legend(legend, labels, ncol=2, loc='upper center', fontsize=LEGEND)

    # figure size
    ax.axis('equal')
    # plt.xlim([-5, 4.5])
    # plt.ylim([-3.5, 2])
    if xlim is None:
        plt.xlim([-(env_limit + 0.5), (env_limit + 0.5)])
    else:
        plt.xlim(xlim)
    if ylim is None:
        plt.ylim([-(env_limit + 0.5), (env_limit + 0.5)])
    else:
        plt.ylim(ylim)

    ax.tick_params(axis='both', labelsize=XYAXIS)
    # plt.title('Trajectory', fontsize=MAIN_TITLE)
    # plt.ylabel(' ', fontsize=XYAXIS)
    plt.xlabel(' ', fontsize=XYAXIS)
    plt.tight_layout()

    plt.savefig(fig_name)
    if show:
        plt.show()
    plt.close()


def print_collision_statistics(exp_type, filter_slack, seed_range, start, end, alpha):
    from statistics.stat_basics import get_empirical_cvar, get_empirical_value_at_risk
    metrics = ['mean', 'cvar', 'drcvar']
    all_dist = [[] for _ in metrics]
    all_pos = [[] for _ in metrics]
    for i, (metric) in enumerate(metrics):
        path = os.path.join("exp_data", exp_type, metric, "slack={}".format(filter_slack))
        for seed in range(seed_range[0], seed_range[1]):
            data_dict = pickle.load(open(os.path.join(path, 'seed={}.pkl'.format(seed)), 'rb'))
            ego_obst_dist_to_col = data_dict["ego_obst_dist_to_col"]
            ego_pos_all = data_dict["ego_pos_all"]
            ego_obst_dist_to_col = np.array(ego_obst_dist_to_col)  # sim_steps x num_obst
            ego_pos_all = np.array(ego_pos_all)  # sim_steps
            sim_steps, num_obst = ego_obst_dist_to_col.shape
            for ob_num in range(num_obst):
                if len(all_dist[i]) == 0:
                    all_dist[i] = [list(ego_obst_dist_to_col[:, ob_num])]
                else:
                    all_dist[i].append(list(ego_obst_dist_to_col[:, ob_num]))
            if len(all_pos[i]) == 0:
                all_pos[i] = [ego_pos_all]
            else:
                all_pos[i].append(ego_pos_all)

    travel_dist = [[] for _ in metrics]
    for i, (metric, all_d, all_p) in enumerate(zip(metrics, all_dist, all_pos)):
        ap = np.array(all_p)  # rep x time steps x 2
        apd = np.diff(ap, axis=1)
        apds = np.linalg.norm(apd, axis=2)
        travel_dist[i] = np.sum(apds, axis=1)

        ad = np.array(all_d)  # rep x time steps
        ad = ad[:, start:end]
        ad = ad.flatten()

        print('Metric: {}'.format(metric))
        print('    Avrg travel dist = {}'.format(np.mean(travel_dist[i])))
        print('    Variance travel dist = {}'.format(np.var(travel_dist[i])))
        print('    Travel dist min, max = {}, {}'.format(np.min(travel_dist[i]), np.max(travel_dist[i])))

        print('  col_info for t=[{}, {}]'.format(start, end))
        col = np.count_nonzero(ad <= 0)
        ent_tot = len(ad)
        col_flag = (ad <= 0)
        col_amts = ad * col_flag
        mean_col_amt = np.sum(col_amts) / np.sum(col) if np.sum(col) > 0 else 0
        max_col_amt = np.min(col_amts)
        print('    # Col / total = Col % = {} / {} = {}'.format(col, ent_tot, col/ent_tot))
        print('    Mean, Max col amount = {}, {}'.format(-mean_col_amt, -max_col_amt))
        print('    D2Col Mean = {}'.format(np.mean(ad)))
        print('    D2Col Variance = {}'.format(np.var(ad)))
        print('    D2Col min, max = {}, {}'.format(np.min(ad), np.max(ad)))
        print('    D2Col VaR = {}'.format(get_empirical_value_at_risk(ad, alpha)))
        print('    D2Col CVaR = {}'.format(get_empirical_cvar(ad, alpha)[0]))
        print('    D2Col CVaR col only = {}'.format(get_empirical_cvar(np.clip(ad, 0, np.inf), alpha)[0]))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def animated_simulation(exp_type, metric, filter_slack, seed, dt,
                        xlim=None, ylim=None, figsize=(8, 8),
                        add_legend=True, xyticks=False, save=True,
                        fps=None, show_samples=False):
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import PillowWriter
    import polytope

    # load all data
    path = os.path.join("exp_data", exp_type, metric, "slack={}".format(filter_slack))
    data_dict = pickle.load(open(os.path.join(path, 'seed={}.pkl'.format(seed)), 'rb'))
    num_obst = data_dict["num_obst"]
    env_limit = data_dict["ENV_LIM"]
    ego_pos_all = data_dict["ego_pos_all"]
    ego_patches_all = data_dict["ego_polys_all"]
    obst_pos_all = data_dict["obst_pos_all"]
    obst_patches_all = data_dict["obst_polys_all"]
    safe_polys_all = data_dict["safe_polys_all"]
    name = os.path.join(path, 'ani_traj_seed={}'.format(seed))
    fig_name = name + '.gif'
    num_frames = len(ego_pos_all)
    num_obst = len(obst_pos_all[0])

    if show_samples and "samples" in data_dict:
        all_samples = data_dict["samples"]
    else:
        all_samples = None

    # create a figure
    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)
    # set the axis limits
    if xlim is None:
        xlim = [-(env_limit + 0.5), (env_limit + 0.5)]
    if ylim is None:
        ylim = [-(env_limit + 0.5), (env_limit + 0.5)]
    # legend setup
    labels = ['safe polytope', 'ego']
    labels.extend(['ob{}'.format(i + 1) for i in range(num_obst)])
    if all_samples is not None:
        labels.append('samples')

    def setup_fig(t):
        ax.axis('equal')
        plt.xlim(xlim)
        plt.ylim(ylim)
        if xyticks:
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=15)
        else:
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
        if add_legend:
            if len(labels) > 3:
                bbox_to_anchor = (0.0, 1.13)
            else:
                bbox_to_anchor = (0.0, 1.1)
            plt.legend(labels, ncol=3, loc='upper left', fontsize=15, bbox_to_anchor=bbox_to_anchor)
        ax.text(0.02, 0.95, 'Time = {:.2f}s'.format(t), transform=ax.transAxes, fontsize=18)

    setup_fig(0)

    # Function to update the plot for each frame
    def update(frame):
        ax.clear()

        if frame > 0:
            p = safe_polys_all[frame-1]
        else:
            A = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
            b = [-99, 100, -99, 100]
            A, b = np.array(A), np.array(b)
            p = polytope.Polytope(A, b)
        p.plot(ax=ax, color='g', alpha=0.15, linestyle='solid', linewidth=1)

        ego_patch = ego_patches_all[frame]
        ego_patch.set_alpha(0.9)
        ax.add_patch(ego_patch)

        for ob in range(num_obst):
            obst_patch = obst_patches_all[frame][ob]
            obst_patch.set_alpha(1)
            ax.add_patch(obst_patch)

        if all_samples is not None:
            if frame > 0:
                t0_samples = all_samples[frame-1]
                xi = np.hstack(t0_samples)
            else:
                xi = np.array([[100, 100], [100, 100]])
            ax.scatter(xi[0, :], xi[1, :], 100, color='k', marker='.', alpha=0.5)

        if add_legend:
            plt.legend(labels, ncol=3, loc='upper left')

        setup_fig(frame * dt)
        return

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000*dt, blit=False, repeat=False, cache_frame_data=False)

    # Save/Display the animation
    if save:
        # Save plot
        if fps is None:
            fps = int(1/dt)
        writer = PillowWriter(fps=fps)
        ani.save(fig_name, writer=writer)
        plt.close()
    else:
        plt.show()
    plt.show()


def animated_simulation_cvxpy_graphs(exp_type, metric, filter_slack, seed, dt,
                               figsize=(8, 8), save=True, fps=None):
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import PillowWriter
    import polytope

    # load all data
    path = os.path.join("exp_data", exp_type, metric, "slack={}".format(filter_slack))
    data_dict = pickle.load(open(os.path.join(path, 'seed={}.pkl'.format(seed)), 'rb'))
    num_obst = data_dict["num_obst"]
    ego_pos_all = data_dict["ego_pos_all"]
    ego_obst_dist_to_col = data_dict["ego_obst_dist_to_col"]
    mpc_filter_time_info = data_dict["mpc_filter_time_info"]
    mpc_filter_status = data_dict["mpc_filter_status"]
    safe_hs_time_info = data_dict["safe_hs_time_info"]
    OBST_COLORS = data_dict["OBST_COLORS"]
    name = os.path.join(path, 'ani_traj_data_seed={}'.format(seed))
    fig_name = name + '.gif'
    num_frames = len(ego_pos_all)
    sim_steps = len(mpc_filter_time_info)
    sim_time = range(sim_steps)
    horizon, num_obst = safe_hs_time_info[0].shape

    mpc_filter_time_info = np.squeeze(mpc_filter_time_info)
    mpc_filter_status = np.squeeze(mpc_filter_status)
    cumulative_times = [np.zeros([sim_steps, 1]) for _ in range(num_obst)]
    for sim_step in range(sim_steps):
        for ob_num in range(num_obst):
            t = np.sum(safe_hs_time_info[sim_step][:, ob_num])
            if ob_num == 0:
                cumulative_times[ob_num][sim_step] = t
            else:
                cumulative_times[ob_num][sim_step] = cumulative_times[ob_num - 1][sim_step] + t

    fig, axs = plt.subplots(3, figsize=figsize, sharex=True)  # define the figure

    def setup_figure():
        axs[0].set_title('MPC Call Time', fontsize=15)
        axs[0].set_ylabel('Time (ms)', fontsize=15)
        axs[0].tick_params(axis='y', labelsize=15)
        axs[0].set_ylim([0, max(mpc_filter_time_info*1.05)])
        axs[1].set_title('MPC Status', fontsize=15)
        axs[1].tick_params(axis='y', labelsize=15)
        axs[1].set_yticks([0, 1])
        axs[1].set_yticklabels(['failed', 'solved'], rotation=40)
        axs[2].set_title('Safe Halfspace Call Time (Cumulative)', fontsize=15)
        axs[2].set_ylabel('Time (ms)', fontsize=15)
        axs[2].tick_params(axis='y', labelsize=15)
        axs[2].set_xlabel('Experiment Time (s)', fontsize=15)
        axs[2].tick_params(axis='x', labelsize=15)
        axs[2].set_ylim([0, max(cumulative_times[-1])*1.05])
        plt.xlim([0, dt*num_frames])
        plt.tight_layout()

    setup_figure()

    def update(frame):
        t = np.linspace(dt, frame * dt, frame)
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        setup_figure()
        if frame == 0:
            return

        axs[0].plot(t, mpc_filter_time_info[0:frame])
        axs[0].fill_between(t, mpc_filter_time_info[0:frame], alpha=0.2)

        axs[1].plot(t, mpc_filter_status[0:frame])
        axs[1].fill_between(t, mpc_filter_status[0:frame], alpha=0.2)

        for ob_num in range(num_obst):
            color = OBST_COLORS[ob_num % len(OBST_COLORS)]
            axs[2].plot(t, cumulative_times[ob_num][0:frame], color=color)
            if ob_num == 0:
                axs[2].fill_between(t, np.squeeze(cumulative_times[ob_num][0:frame]), color=color, alpha=0.2)
            else:
                axs[2].fill_between(t, np.squeeze(cumulative_times[ob_num][0:frame]),
                                    np.squeeze(cumulative_times[ob_num-1][0:frame]),
                                    color=color, alpha=0.2)

    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000 * dt, blit=False, repeat=False,
                        cache_frame_data=False)

    # Save/Display the animation
    if save:
        # Save plot
        if fps is None:
            fps = int(1 / dt)
        writer = PillowWriter(fps=fps)
        ani.save(fig_name, writer=writer)
        plt.close()
    else:
        plt.show()
    plt.show()


def animated_simulation_distance_to_col_graphs(exp_type, metric, filter_slack, seed, dt,
                                               figsize=(8, 8), save=True, fps=None):
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import PillowWriter

    # load all data
    path = os.path.join("exp_data", exp_type, metric, "slack={}".format(filter_slack))
    data_dict = pickle.load(open(os.path.join(path, 'seed={}.pkl'.format(seed)), 'rb'))
    ego_pos_all = data_dict["ego_pos_all"]
    ego_obst_dist_to_col = data_dict["ego_obst_dist_to_col"]
    safe_hs_time_info = data_dict["safe_hs_time_info"]
    OBST_COLORS = data_dict["OBST_COLORS"]
    name = os.path.join(path, 'ani_col_dist_seed={}'.format(seed))
    fig_name = name + '.gif'
    num_frames = len(ego_pos_all)
    horizon, num_obst = safe_hs_time_info[0].shape
    ego_obst_dist_to_col = np.array(ego_obst_dist_to_col)  # sim_steps x num_obst
    sim_steps, num_obst = ego_obst_dist_to_col.shape
    fig, axs = plt.subplots(1, figsize=figsize, sharex=True)  # define the figure

    def setup_figure():
        plt.ylabel('Distance to collision', fontsize=15)
        plt.xlabel('Timestep', fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.ylim([min(0, np.min(ego_obst_dist_to_col)) - 0.1, np.max(ego_obst_dist_to_col) + 0.1])
        plt.tight_layout()
        plt.grid(visible=True)

    setup_figure()

    def update(frame):
        t = np.linspace(0, (sim_steps - 1) * dt, sim_steps)
        axs.clear()

        plt.plot([0, sim_steps*dt], [0, 0], color='k')
        for ob_num in range(num_obst):
            color = OBST_COLORS[ob_num % len(OBST_COLORS)]
            dists = list(ego_obst_dist_to_col[:, ob_num][0:frame + 1])
            dists_with_nan_pad = dists + [np.nan] * (sim_steps - (frame+1))
            plt.plot(t,  dists_with_nan_pad, color=color)
        setup_figure()

    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000 * dt, blit=False, repeat=False,
                        cache_frame_data=False)

    # Save/Display the animation
    if save:
        # Save plot
        if fps is None:
            fps = int(1 / dt)
        writer = PillowWriter(fps=fps)
        ani.save(fig_name, writer=writer)
        plt.close()
    else:
        plt.show()
    plt.show()


def generate_all_animations(exp_type, metric, filter_slack, seed, dt=0.2,
                            figsize=(8, 8), traj_xlim=(-5, 5), traj_ylim=(-5, 5),
                            show_samples=False, fps=None, save=False,  add_legend=True):
    print('Generating animations for {} with {} metric'.format(exp_type, metric))
    animated_simulation(
        exp_type=exp_type, metric=metric, filter_slack=filter_slack,
        seed=seed, dt=dt,
        xlim=traj_xlim, ylim=traj_ylim, figsize=figsize, add_legend=add_legend,
        save=save, fps=fps, show_samples=show_samples)
    print('     Generated trajectory simulation')
    animated_simulation_cvxpy_graphs(
        exp_type=exp_type, metric=metric, filter_slack=filter_slack,
        seed=seed, dt=dt,
        figsize=figsize, save=save, fps=fps)
    print('     Generated cvxpy graphs simulation')
    animated_simulation_distance_to_col_graphs(
        exp_type=exp_type, metric=metric, filter_slack=filter_slack,
        seed=seed, dt=dt,
        figsize=figsize, save=save, fps=fps)
    print('     Generated distance to collision simulation')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    # print_collision_statistics(exp_type='ego_intersect', filter_slack=False, seed_range=[1, 301], start=5, end=9, alpha=0.2)

    metrics = ['mean', 'cvar', 'drcvar']
    exp_types = ['ego_headon', 'ego_intersect']
    filter_slack, seed, fps = False, 19, 2
    for metric in metrics:
        for exp_type in exp_types:
            generate_all_animations(exp_type, metric, filter_slack, seed, dt=0.2,
                                    show_samples=True, fps=fps, save=True, add_legend=True)

    exp_type, metric, filter_slack, seed, fps = 'ego_and_3_vehicles', 'drcvar', False, 2023, 2
    animated_simulation_cvxpy_graphs(
        exp_type=exp_type, metric=metric, filter_slack=filter_slack,
        seed=seed, dt=0.2, save=True, fps=fps)


if __name__ == "__main__":
    main()
