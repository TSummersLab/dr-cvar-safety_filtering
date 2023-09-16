"""
Given some samples, we compute the expected value, VaR, DR-VaR, CVaR, and DR-CVaR and compare true and empirical results
Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
April 8, 2023
"""
import numpy as np
import cvxpy as cp
from scipy.stats import norm
from scipy.stats import expon
import matplotlib.pyplot as plt
import os
from statistics.random_samples_fxns import generate_noise_samples as generate_raw_samples


def get_true_mean(loc, scale, dist):
    if dist == "norm":
        return norm.mean(loc=loc, scale=scale)
    elif dist == 'expo':
        return expon.mean(loc=loc, scale=scale)
    else:
        raise NotImplementedError('Chosen distribution not implemented')


def get_empirical_mean(data):
    return np.mean(data)


def get_true_value_at_risk(loc, scale, dist, alpha):
    if dist == "norm":
        return norm.ppf(1 - alpha, loc=loc, scale=scale)  # Value at Risk
    elif dist == 'expo':
        return expon.ppf(1 - alpha, loc=loc, scale=scale)  # Value at Risk
    else:
        raise NotImplementedError('Chosen distribution not implemented')


def get_moment_based_dr_value_at_risk(mean, var, alpha):
    return mean + np.sqrt((1-alpha)/alpha) * np.sqrt(float(var))


def get_empirical_value_at_risk(data, alpha):
    return np.percentile(data, (1-alpha) * 100)


def get_empirical_cvar(data, alpha, verbose=False):
    n = len(data)
    tau = cp.Variable()  # cvar decision variable
    obj = 1 / n * cp.sum(cp.maximum(tau, 1 / alpha * (data - tau) + tau))  # cvar objective
    constr = []
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()
    cvar = prob.value
    tau_val = tau.value
    if verbose:
        print("CVaR Problem Status: ", prob.status)
    return cvar, tau_val


def get_empirical_drcvar(data, alpha, eps, Xi_bounds=None):
    n = len(data)
    K = 2
    if Xi_bounds is not None:
        d, D = Xi_bounds['d'], Xi_bounds['D']
    tau = cp.Variable()  # cvar decision variable
    lam = cp.Variable()
    eta = cp.Variable((n, ))
    gamma = cp.Variable((n, K))
    obj = lam * eps + 1/n * cp.sum(eta)
    constr = []
    if Xi_bounds is None:
        constr.append((1 - 1 / alpha) * tau + 1 / alpha * data <= eta)
        constr.append(tau <= eta)
        constr.append(1 / alpha <= lam)
    else:
        constr.append((1 - 1 / alpha) * tau + 1 / alpha * data + gamma[:, 0] @ (d - D @ data) <= eta)
        constr.append(tau + gamma[:, 1] @ (d - D @ data) <= eta)
        constr.append(cp.norm(D @ gamma[:, 0] - 1 / alpha) <= lam)
        constr.append(cp.norm(D @ gamma[:, 1]) <= lam)
        constr = [gamma >= 0]
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()
    drcvar = prob.value
    if prob.status != 'optimal':
        print("DRCVaR Problem Status: ", prob.status)
        other_data = {}
    else:
        other_data = {'lam': lam.value}
    return drcvar, other_data


def hist_plot(data, values_dict_true, values_dict_emp, save_path=None, title='plot', save=False, show=False):
    colors = ['k', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:olive', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:cyan', 'tab:blue']
    plt.clf()
    plt.hist(data, 100)
    keys = []
    for i, key in enumerate(values_dict_true):
        val = values_dict_true[key]
        keys.append(key)
        plt.axvline(val, color=colors[i % len(colors)], linestyle='solid', linewidth=5)
    for i, key in enumerate(values_dict_emp):
        val = values_dict_emp[key]
        keys.append(key)
        plt.axvline(val, color=colors[i % len(colors)], linestyle='dashed', linewidth=5)
    plt.legend(keys, fontsize=16)  #, loc='upper left')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('count', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=16)

    if save_path is not None:
        path = os.path.join(save_path, title + '.png')
    else:
        path = title + '.png'

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    return


def many_samples_graphs(n, dist, savepath, save, show):
    mu = 0  # mean
    sigma = 1  # std div
    alpha = 0.1  # tail probability

    # generate basic stats
    xi = generate_raw_samples(n, mu, sigma, dist)
    E_true = get_true_mean(mu, sigma, dist)
    VaR_true = get_true_value_at_risk(mu, sigma, dist, alpha)
    values_dict_true = {'true_mean': E_true, 'true_var': VaR_true}
    values_dict_emp = {}
    title = 'dist=' + dist + ', n=' + str(n) + ', true mean, var'
    hist_plot(xi, values_dict_true, values_dict_emp, savepath, title, save, show)


def inc_samp(n, dist, savepath, save, show):
    mu = 0  # mean
    sigma = 1  # std div
    alpha = 0.1  # tail probability

    xi = generate_raw_samples(n, mu, sigma, dist)
    E_true = get_true_mean(mu, sigma, dist)
    E_emp = get_empirical_mean(xi)
    VaR_true = get_true_value_at_risk(mu, sigma, dist, alpha)
    VaR_emp = get_empirical_value_at_risk(xi, alpha)

    values_dict_true = {'true_mean': E_true, 'true_var': VaR_true}
    values_dict_emp = {'emp_mean': E_emp, 'emp_var': VaR_emp}
    title = 'dist=' + dist + ', n=' + str(n) + ', true mean, var + sample based'
    hist_plot(xi, values_dict_true, values_dict_emp, savepath, title, save, show)


def with_cvar(n, dist, savepath, save, show):
    mu = 0  # mean
    sigma = 1  # std div
    alpha = 0.1  # tail probability
    eps = 0.1

    xi = generate_raw_samples(n, mu, sigma, dist)
    E_true = get_true_mean(mu, sigma, dist)
    E_emp = get_empirical_mean(xi)
    VaR_true = get_true_value_at_risk(mu, sigma, dist, alpha)
    VaR_emp = get_empirical_value_at_risk(xi, alpha)
    CVaR_opt, tau = get_empirical_cvar(xi, alpha)

    tail_count = sum([1 if samp > VaR_true else 0 for samp in xi])
    alpha_emp = tail_count/n

    print('True mean: ', E_true)
    print('Emp mean: ', E_emp)
    print('True VaR: ', VaR_true)
    print('Emp VaR: ', VaR_emp)
    print('True alpha: ', alpha)
    print('Emp alpha: ', alpha_emp)
    print('Emp CVaR (cvxpy): ', CVaR_opt)
    print('Tau (cvxpy): ', tau)

    values_dict_true = {'true_mean': E_true,
                        'true_var': VaR_true}
    values_dict_emp = {'emp_mean': E_emp,
                       'emp_var': VaR_emp,
                       'emp_cvar': CVaR_opt}
    title = 'dist=' + dist + ', n=' + str(n) + ', with cvar'
    hist_plot(xi, values_dict_true, values_dict_emp, savepath, title, save, show)


def with_dr_variants(n, dist, savepath, save, show):
    mu = 0  # mean
    sigma = 1  # std div
    alpha = 0.1  # tail probability
    eps = 0.1

    xi = generate_raw_samples(n, mu, sigma, dist)
    E_true = get_true_mean(mu, sigma, dist)
    E_emp = get_empirical_mean(xi)
    VaR_true = get_true_value_at_risk(mu, sigma, dist, alpha)
    VaR_emp = get_empirical_value_at_risk(xi, alpha)
    DRVaR = get_moment_based_dr_value_at_risk(mu, sigma, alpha)
    CVaR_opt, tau = get_empirical_cvar(xi, alpha)
    DRCVaR_opt, _ = get_empirical_drcvar(xi, alpha, eps, Xi_bounds=None)

    tail_count = sum([1 if samp > VaR_true else 0 for samp in xi])
    alpha_emp = tail_count/n

    print('True mean: ', E_true)
    print('Emp mean: ', E_emp)
    print('True VaR: ', VaR_true)
    print('Emp VaR: ', VaR_emp)
    print('True alpha: ', alpha)
    print('Emp alpha: ', alpha_emp)
    print('DR-VaR: ', DRVaR)
    print('Emp CVaR (cvxpy): ', CVaR_opt)
    print('Tau (cvxpy): ', tau)
    print('DR-CVaR: ', DRCVaR_opt)

    values_dict_true = {'true_mean': E_true,
                        'true_var': VaR_true}
    values_dict_emp = {'emp_mean': E_emp,
                       'emp_var': VaR_emp,
                       'emp_cvar': CVaR_opt,
                       'emp_drcvar': DRCVaR_opt,
                       'drvar': DRVaR}
    title = 'dist=' + dist + ', n=' + str(n) + ', with dr variants'
    hist_plot(xi, values_dict_true, values_dict_emp, savepath, title, save, show)


def many_dist_example(n, savepath, save, show):
    mu = 0  # mean
    sigma = 1  # std div
    alpha = 0.1  # tail probability
    eps = 0.1
    dist = 'multi'

    xi_norm = generate_raw_samples(n, mu, sigma, 'norm')
    xi_expo = generate_raw_samples(n, mu, sigma, 'expo')
    xi_lap = generate_raw_samples(n, mu, sigma, 'lap')
    xi_bern1 = generate_raw_samples(n, mu, sigma * 1, 'bern')
    xi_bern3 = generate_raw_samples(n, mu, sigma * 3, 'bern')
    xi_bern5 = generate_raw_samples(n, mu, sigma * 5, 'bern')
    xi_bern7 = generate_raw_samples(n, mu, sigma * 7, 'bern')
    xi_bern9 = generate_raw_samples(n, mu, sigma * 9, 'bern')

    xi = xi_norm[:]
    for i in range(n):
        r = np.random.rand()
        if r < 0.25:
            xi[i] = xi_norm[i]
        elif r < 0.5:
            xi[i] = xi_expo[i]
        elif r < 0.75:
            xi[i] = xi_lap[i]
        elif r < 0.8:
            xi[i] = xi_bern1[i]
        elif r < 0.85:
            xi[i] = xi_bern3[i]
        elif r < 0.9:
            xi[i] = xi_bern5[i]
        elif r < 0.95:
            xi[i] = xi_bern7[i]
        elif r <= 1:
            xi[i] = xi_bern9[i]

    E_emp = get_empirical_mean(xi)
    sigma = np.var(xi)
    VaR_emp = get_empirical_value_at_risk(xi, alpha)
    DRVaR = get_moment_based_dr_value_at_risk(E_emp, sigma, alpha)
    CVaR_opt, tau = get_empirical_cvar(xi, alpha)
    DRCVaR_opt, _ = get_empirical_drcvar(xi, alpha, eps, Xi_bounds=None)

    print('Emp mean: ', E_emp)
    print('Emp variance', sigma)
    print('Emp VaR: ', VaR_emp)
    print('True alpha: ', alpha)
    print('DR-VaR: ', DRVaR)
    print('Emp CVaR (cvxpy): ', CVaR_opt)
    print('Tau (cvxpy): ', tau)
    print('DR-CVaR: ', DRCVaR_opt)

    values_dict_true = {}
    values_dict_emp = {'mean': E_emp,
                       'var': VaR_emp,
                       'cvar': CVaR_opt,
                       'drcvar': DRCVaR_opt,
                       'drvar': DRVaR}
    title = 'dist=' + dist + ', n=' + str(n)
    hist_plot(xi, values_dict_true, values_dict_emp, savepath, title, save, show)


if __name__ == "__main__":
    savepath = 'pictures'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    save, show = True, False
    # many_samples_graphs(n=1000000, dist='norm', savepath=savepath, save=save, show=show)
    # many_samples_graphs(n=1000000, dist='expo', savepath=savepath, save=save, show=show)
    #
    # inc_samp(n=1000000, dist='norm', savepath=savepath, save=save, show=show)
    # inc_samp(n=10000, dist='norm', savepath=savepath, save=save, show=show)
    # inc_samp(n=100, dist='norm', savepath=savepath, save=save, show=show)
    #
    # inc_samp(n=1000000, dist='expo', savepath=savepath, save=save, show=show)
    # inc_samp(n=10000, dist='expo', savepath=savepath, save=save, show=show)
    # inc_samp(n=100, dist='expo', savepath=savepath, save=save, show=show)

    # with_cvar(n=1000, dist='norm', savepath=savepath, save=save, show=show)
    # with_cvar(n=1000, dist='expo', savepath=savepath, save=save, show=show)

    with_dr_variants(n=1000, dist='norm', savepath=savepath, save=save, show=show)
    with_dr_variants(n=1000, dist='expo', savepath=savepath, save=save, show=show)

    # many_dist_example(n=1000, savepath=savepath, save=save, show=show)
    # many_dist_example(n=10000, savepath=savepath, save=save, show=show)
