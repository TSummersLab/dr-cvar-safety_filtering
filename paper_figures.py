"""
Script that generates all figures for the paper

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
July 17, 2023
"""
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

from backend.safe_halfspaces import CVaRHalfspace, DRCVaRHalfspace, MeanHalfspace
from statistics.random_samples_fxns import generate_noise_samples

# font sizes
MAIN_TITLE = 22
SUBPLOT_TITLE = 22
LEGEND = 18
XYAXIS = 22


def safe_halfspaces_comparison(xi, ego, ob, r1=0., r2=0., emph_halfspace_idx=[], show=True):
    """
    Compares between some safe halfspaces
    :param xi: samples
    :param h: halfspace normal (will get normalized)
    :param r1: ego radius
    :param r2: obstacle radius
    :return:
    """
    h = ob - ego
    h = h / np.linalg.norm(h)
    num_samp = xi.shape[1]
    delta = 0.1
    alpha = 0.2
    eps_list = [0.05, 0.1, 0.2]
    r = [r1+r2]

    # compute the safe halfspaces
    mean_hs = MeanHalfspace()
    mean_hs.set_opt_pb_params(h, xi, r)
    mean_hs.solve_opt_pb()

    cvar_hs = CVaRHalfspace(alpha, delta, num_samp, loss_type='continuous')
    cvar_hs.set_opt_pb_params(h, xi, r)
    cvar_hs.solve_opt_pb()

    drcvar_hs0 = DRCVaRHalfspace(alpha, eps_list[0], delta, num_samp)
    drcvar_hs0.set_opt_pb_params(h, xi, r)
    drcvar_hs0.solve_opt_pb()

    drcvar_hs1 = DRCVaRHalfspace(alpha, eps_list[1], delta, num_samp)
    drcvar_hs1.set_opt_pb_params(h, xi, r)
    drcvar_hs1.solve_opt_pb()

    drcvar_hs2 = DRCVaRHalfspace(alpha, eps_list[2], delta, num_samp)
    drcvar_hs2.set_opt_pb_params(h, xi, r)
    drcvar_hs2.solve_opt_pb()

    # plotting
    c_alpha = 0.2
    fig, ax = plt.subplots(figsize=(8, 7))
    circle = Circle(ego, radius=r1, fill=False, edgecolor='tab:blue', linewidth=5)
    ax.add_patch(circle)
    circle = Circle(ob, radius=r2, fill=False, edgecolor='tab:red', linewidth=5)
    ax.add_patch(circle)
    line = Line2D([ego[0], ob[0]], [ego[1], ob[1]], linestyle='--', color='k', linewidth=2)
    ax.add_patch(line)
    arrow = FancyArrowPatch(ego, ego + h, arrowstyle='-|>', mutation_scale=20, color='k', linewidth=2)
    ax.add_patch(arrow)
    ax.scatter(xi[0, :], xi[1, :], 100, color='k', marker='.')
    if 0 in emph_halfspace_idx:
        mean_hs.plot_poly(ax=ax, color='tab:red', alpha=c_alpha, show=False, linewidth=5)
    else:
        mean_hs.plot_poly(ax=ax, color='tab:red', alpha=c_alpha, show=False)

    if 1 in emph_halfspace_idx:
        cvar_hs.plot_poly(ax=ax, color='tab:blue', alpha=c_alpha, show=False, linewidth=5)
    else:
        cvar_hs.plot_poly(ax=ax, color='tab:blue', alpha=c_alpha, show=False)

    if 2 in emph_halfspace_idx:
        drcvar_hs0.plot_poly(ax=ax, color='tab:green', alpha=c_alpha, show=False, linewidth=5)
    else:
        drcvar_hs0.plot_poly(ax=ax, color='tab:green', alpha=c_alpha, show=False)

    if 3 in emph_halfspace_idx:
        drcvar_hs1.plot_poly(ax=ax, color='tab:olive', alpha=c_alpha, show=False, linewidth=5)
    else:
        drcvar_hs1.plot_poly(ax=ax, color='tab:olive', alpha=c_alpha, show=False)

    if 4 in emph_halfspace_idx:
        drcvar_hs2.plot_poly(ax=ax, color='tab:brown', alpha=c_alpha, show=False, linewidth=5)
    else:
        drcvar_hs2.plot_poly(ax=ax, color='tab:brown', alpha=c_alpha, show=False)

    plt.xlim([-2, 2])
    plt.ylim([-1.5, 2])
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=XYAXIS)
    plt.title('Safe Halfspaces Comparison', fontsize=MAIN_TITLE)
    plt.legend(['ego ($y^r$)', 'obstacle ($p$)', '$p-y^r$', 'h', 'samples', 'Mean', 'CVaR',
                'DR-CVaR, $\epsilon$={}'.format(eps_list[0]),
                'DR-CVaR, $\epsilon$={}'.format(eps_list[1]),
                'DR-CVaR, $\epsilon$={}'.format(eps_list[2])],
               # loc='upper center',
               # bbox_to_anchor=(0.5, 0.95),
               # fancybox=True,
               ncol=2,
               fontsize=LEGEND)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1)
    path = os.path.join("exp_data", "halfspace_comparison")
    path_exist = os.path.exists(path)
    if not path_exist:
        os.makedirs(path)
    if len(emph_halfspace_idx) > 0:
        plt.savefig(os.path.join(
            path, 'safe_hs_compare_alpha={},_delta={},_eps={}_emph{}'.format(alpha, delta, eps_list, emph_halfspace_idx)) + '.png')
    else:
        plt.savefig(os.path.join(
            path, 'safe_hs_compare_alpha={},_delta={},_eps={}'.format(alpha, delta, eps_list)) + '.png')
    if show:
        plt.show()
    else:
        plt.close()


def compute_times(num_samp_list, ego, ob, scale, seed=1, repititions=500, remove_min_max=True, r1=0., r2=0.):
    def get_xi(num_samp):
        xi = np.zeros((2, num_samp))
        xi[0, :] = generate_noise_samples(num_samp, ob[0], np.sqrt(scale[0]), dist='norm')
        xi[1, :] = generate_noise_samples(num_samp, ob[1], np.sqrt(scale[1]), dist='norm')
        return xi

    np.random.seed(seed)
    h = ob - ego
    h = h / np.linalg.norm(h)
    delta = 0.1
    alpha = 0.2
    eps = 0.05
    r = [r1 + r2]
    extra = 0
    if remove_min_max:
        extra = 2

    cvar_setup_time, cvar_solve_time, cvar_call_time = [], [], []
    dr_cvar_setup_time, dr_cvar_solve_time, dr_cvar_call_time = [], [], []
    for n, num_samp in enumerate(num_samp_list):
        cvar_setup_time.append([])
        cvar_solve_time.append([])
        cvar_call_time.append([])
        dr_cvar_setup_time.append([])
        dr_cvar_solve_time.append([])
        dr_cvar_call_time.append([])

        cvar_hs = CVaRHalfspace(alpha, delta, num_samp, loss_type='continuous', solver='ECOS')
        drcvar_hs = DRCVaRHalfspace(alpha, eps, delta, num_samp, solver='ECOS')
        for rep in range(repititions + extra):
            xi = get_xi(num_samp)
            cvar_hs.set_opt_pb_params(h, xi, r)
            _, cvar_info = cvar_hs.solve_opt_pb()
            drcvar_hs.set_opt_pb_params(h, xi, r)
            _, dr_cvar_info = drcvar_hs.solve_opt_pb()

            cvar_setup_time[n].append(cvar_info['setup_time'] * 1000)
            cvar_solve_time[n].append(cvar_info['solve_time'] * 1000)
            cvar_call_time[n].append(cvar_info['solve_call_time'] * 1000)
            dr_cvar_setup_time[n].append(dr_cvar_info['setup_time'] * 1000)
            dr_cvar_solve_time[n].append(dr_cvar_info['solve_time'] * 1000)
            dr_cvar_call_time[n].append(dr_cvar_info['solve_call_time'] * 1000)

        if remove_min_max:
            cvar_setup_time[n].remove(min(cvar_setup_time[n]))
            cvar_setup_time[n].remove(max(cvar_setup_time[n]))
            cvar_solve_time[n].remove(min(cvar_solve_time[n]))
            cvar_solve_time[n].remove(max(cvar_solve_time[n]))
            cvar_call_time[n].remove(min(cvar_call_time[n]))
            cvar_call_time[n].remove(max(cvar_call_time[n]))
            dr_cvar_setup_time[n].remove(min(dr_cvar_setup_time[n]))
            dr_cvar_setup_time[n].remove(max(dr_cvar_setup_time[n]))
            dr_cvar_solve_time[n].remove(min(dr_cvar_solve_time[n]))
            dr_cvar_solve_time[n].remove(max(dr_cvar_solve_time[n]))
            dr_cvar_call_time[n].remove(min(dr_cvar_call_time[n]))
            dr_cvar_call_time[n].remove(max(dr_cvar_call_time[n]))

    data = {"cvar_setup_time": cvar_setup_time,
            "cvar_solve_time": cvar_solve_time,
            "cvar_call_time": cvar_call_time,
            "num_samp_list": num_samp_list,
            "repititions": repititions,
            "dr_cvar_setup_time": dr_cvar_setup_time,
            "dr_cvar_solve_time": dr_cvar_solve_time,
            "dr_cvar_call_time": dr_cvar_call_time, }
    path = os.path.join("exp_data", "compute_times")
    path_exist = os.path.exists(path)
    if not path_exist:
        os.makedirs(path)
    pickle.dump(data, open(os.path.join(
        path, 'seed={}_rep={}_num_samp_list={}.pkl'.format(seed, repititions, num_samp_list)), 'wb'))


def plot_compute_times(seed, rep, num_samp_list):
    path = os.path.join("exp_data", "compute_times")
    data = pickle.load(open(os.path.join(
        path, 'seed={}_rep={}_num_samp_list={}.pkl'.format(seed, rep, num_samp_list)), 'rb'))
    cvar_setup_time = data["cvar_setup_time"]
    cvar_solve_time = data["cvar_solve_time"]
    cvar_call_time = data["cvar_call_time"]
    num_samp_list = data["num_samp_list"]
    repititions = data["repititions"]
    dr_cvar_setup_time = data["dr_cvar_setup_time"]
    dr_cvar_solve_time = data["dr_cvar_solve_time"]
    dr_cvar_call_time = data["dr_cvar_call_time"]

    def print_results():
        def p(type_t, mean_t, var_t, max_t):
            print("   {}: mean, var [max] = {}, {} [{}]".format(type_t, mean_t, var_t, max_t))
        for n, n_samp in enumerate(num_samp_list):
            print("CVaR, N={}".format(n_samp))
            mean_t, var_t, max_t = np.mean(cvar_setup_time[n]), np.std(cvar_setup_time[n]), np.max(cvar_setup_time[n])
            p('setup', mean_t, var_t, max_t)
            mean_t, var_t, max_t = np.mean(cvar_solve_time[n]), np.std(cvar_solve_time[n]), np.max(cvar_solve_time[n])
            p('solve', mean_t, var_t, max_t)
            mean_t, var_t, max_t = np.mean(cvar_call_time[n]), np.std(cvar_call_time[n]), np.max(cvar_call_time[n])
            p('call', mean_t, var_t, max_t)

            print("DR-CVaR, N={}".format(n_samp))
            mean_t, var_t, max_t = np.mean(dr_cvar_setup_time[n]), np.std(dr_cvar_setup_time[n]), np.max(dr_cvar_setup_time[n])
            p('setup', mean_t, var_t, max_t)
            mean_t, var_t, max_t = np.mean(dr_cvar_solve_time[n]), np.std(dr_cvar_solve_time[n]), np.max(dr_cvar_solve_time[n])
            p('solve', mean_t, var_t, max_t)
            mean_t, var_t, max_t = np.mean(dr_cvar_call_time[n]), np.std(dr_cvar_call_time[n]), np.max(dr_cvar_call_time[n])
            p('call', mean_t, var_t, max_t)

    print_results()

    def figure_setup(title):
        fig, axs = plt.subplots(3, figsize=(11, 9), sharex=True)
        # fig.suptitle(title, fontsize=MAIN_TITLE)
        axs[0].set_title('Setup Time', fontsize=SUBPLOT_TITLE)
        axs[0].set_ylabel('Time (ms)', fontsize=XYAXIS)
        axs[0].tick_params(axis='y', labelsize=XYAXIS)
        axs[1].set_title('Solve Time', fontsize=SUBPLOT_TITLE)
        axs[1].set_ylabel('Time (ms)', fontsize=XYAXIS)
        axs[1].tick_params(axis='y', labelsize=XYAXIS)
        axs[2].set_title('Call Time', fontsize=SUBPLOT_TITLE)
        axs[2].set_ylabel('Time (ms)', fontsize=XYAXIS)
        axs[2].tick_params(axis='y', labelsize=XYAXIS)
        axs[2].set_xlabel('Number Samples', fontsize=XYAXIS)
        return axs

    axs = figure_setup('CVaR Compute Time')
    axs[0].boxplot(cvar_setup_time)
    axs[1].boxplot(cvar_solve_time)
    axs[2].boxplot(cvar_call_time)
    plt.xticks(range(1, len(num_samp_list) + 1), num_samp_list, fontsize=XYAXIS)
    plt.tight_layout()
    plt.savefig(os.path.join(path,
                             'cvar_compute_times_num_samp={}_reps={}'.format(num_samp_list, repititions)) + '.png')
    plt.show()

    axs = figure_setup('DR-CVaR Compute Time')
    axs[0].boxplot(dr_cvar_setup_time)
    axs[1].boxplot(dr_cvar_solve_time)
    axs[2].boxplot(dr_cvar_call_time)
    plt.xticks(range(1, len(num_samp_list) + 1), num_samp_list, fontsize=XYAXIS)
    plt.tight_layout()
    plt.savefig(os.path.join(path,
                             'dr_cvar_compute_times_num_samp={}_reps={}'.format(num_samp_list, repititions)) + '.png')
    plt.show()

    def single_plot():
        bps = []
        group_width = 0.2
        legend = []

        fig, axs = plt.subplots(4, figsize=(14, 14), sharex=True)
        fig.suptitle('CVXPY Problem Solve Times', fontsize=MAIN_TITLE)
        group_labels = [f'{i}' for i in num_samp_list]
        axs[0].set_title('Setup Time', fontsize=SUBPLOT_TITLE)
        axs[0].set_ylabel('Time (ms)', fontsize=XYAXIS)
        axs[0].tick_params(axis='y', labelsize=XYAXIS)
        axs[1].set_title('Solve Time', fontsize=SUBPLOT_TITLE)
        axs[1].set_ylabel('Time (ms)', fontsize=XYAXIS)
        axs[1].tick_params(axis='y', labelsize=XYAXIS)
        axs[2].set_title('Call Time (CVaR)', fontsize=SUBPLOT_TITLE)
        axs[2].set_ylabel('Time (ms)', fontsize=XYAXIS)
        axs[2].tick_params(axis='y', labelsize=XYAXIS)
        axs[3].set_title('Call Time (DR-CVaR)', fontsize=SUBPLOT_TITLE)
        axs[3].set_ylabel('Time (ms)', fontsize=XYAXIS)
        axs[3].tick_params(axis='y', labelsize=XYAXIS)
        axs[3].set_xlabel('Number Samples', fontsize=XYAXIS)
        axs[3].set_xticks(np.arange(len(num_samp_list)) + group_width / 2)
        axs[3].set_xticklabels(group_labels)

        group_positions = np.arange(len(num_samp_list)) + 0 * group_width
        bp = axs[0].boxplot(cvar_setup_time, positions=group_positions, patch_artist=True, widths=group_width)
        bps.append(bp)
        legend.append(bp["boxes"][0])
        group_positions = np.arange(len(num_samp_list)) + 1 * group_width
        bp = axs[0].boxplot(dr_cvar_setup_time, positions=group_positions, patch_artist=True, widths=group_width)
        bps.append(bp)
        legend.append(bp["boxes"][0])

        group_positions = np.arange(len(num_samp_list)) + 0 * group_width
        bp = axs[1].boxplot(cvar_solve_time, positions=group_positions, patch_artist=True, widths=group_width)
        bps.append(bp)
        legend.append(bp["boxes"][0])
        group_positions = np.arange(len(num_samp_list)) + 1 * group_width
        bp = axs[1].boxplot(dr_cvar_solve_time, positions=group_positions, patch_artist=True, widths=group_width)
        bps.append(bp)
        legend.append(bp["boxes"][0])

        group_positions = np.arange(len(num_samp_list)) + 0.5 * group_width
        bp = axs[2].boxplot(cvar_call_time, positions=group_positions, patch_artist=True, widths=group_width)
        bps.append(bp)
        legend.append(bp["boxes"][0])
        group_positions = np.arange(len(num_samp_list)) + 0.5 * group_width
        bp = axs[3].boxplot(dr_cvar_call_time, positions=group_positions, patch_artist=True, widths=group_width)
        bps.append(bp)
        legend.append(bp["boxes"][0])

        for bp, color in zip(bps, ['tab:blue', 'tab:red'] * 3):
            for patch in bp['boxes']:
                patch.set_facecolor(color)

        # plt.savefig(os.path.join(path,
        #                          'cvar_compute_times_num_samp={}_reps={}'.format(num_samp_list, repititions)) + '.png')
        plt.show()


# #################################################################################################################### #
def paper_halfspace_results():
    np.random.seed(1)
    num_samp = 100
    ob = np.array([0.5, 0])
    ego = np.array([-0.9, -0.8])
    noise_std_dev = np.array([0.1, 0.1])
    r = 0.3
    xi = np.zeros((2, num_samp))
    xi[0, :] = generate_noise_samples(num_samp, ob[0], np.sqrt(noise_std_dev[0]), dist='norm')
    xi[1, :] = generate_noise_samples(num_samp, ob[1], np.sqrt(noise_std_dev[1]), dist='norm')

    safe_halfspaces_comparison(xi, ego=ego, ob=ob, r1=r, r2=r, show=False)


def paper_compute_times_results():
    num_samp_list = [10, 50, 100, 500, 1000, 1500]
    ob = np.array([0.5, 0])
    ego = np.array([-0.9, -0.8])
    noise_std_dev = np.array([0.1, 0.1])
    r = 0.3
    seed = 1
    rep = 500
    # compute_times(num_samp_list, ego, ob, noise_std_dev, seed, rep, r1=r, r2=r) # uncomment to print some statistics about the results
    plot_compute_times(seed, rep, num_samp_list)


if __name__ == "__main__":
    paper_halfspace_results()
    # paper_compute_times_results() # this take a long time. uncomment to generate compute time box plots
