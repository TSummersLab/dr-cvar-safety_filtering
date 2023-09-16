"""
Problem classes that compute safe halfspaces.

main also generates the loss function comparison and CVaR/DR-CVaR comparison plots

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
July 13, 2023
"""

import os
import time
from copy import copy

import numpy as np
import cvxpy as cp
import polytope
from matplotlib import pyplot as plt
from statistics.random_samples_fxns import generate_noise_samples
from statistics.stat_basics import get_empirical_cvar, get_empirical_drcvar


class Halfspace:
    """
    Class representing a 2D halfspace given by {x | h x + g <= 0}
    """
    def __init__(self, A=None, b=None):
        self.h = None
        self.g = None
        self._halfspace_bounds(A, b)
        self.poly = None

    def _halfspace_bounds(self, A, b):
        """
        A * X <= b linear constraint to place bounds on X (2 x 1 vector)
        :param A: c x 2 matrix
        :param b: c x 1 vector
        :return:
        """
        if A is None:
            self.A = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
        else:
            self.A = A
        if b is None:
            self.b = [100, 100, 100, 100]
        else:
            self.b = b

    @property
    def get_poly(self):
        """
        Create a Polytope from the polytope library given by the halfspace bound to the specified environment bounds
        :return: polytope.Polytope
        """
        A, b = copy(self.A), copy(self.b)
        A.append(self.h)
        b.append(-self.g)
        A, b = np.array(A), np.array(b)
        poly = polytope.Polytope(A, b)  # polytope defined as Ax <= b
        self.poly = poly
        return poly

    @property
    def get_Ax_leq_b_A(self):
        """ returns the A component of the halfspace formatted as A * x <= b. Excludes the environment bounds """
        return self.h

    @property
    def get_Ax_leq_b_b(self):
        """ returns the b component of the halfspace formatted as A * x <= b. Excludes the environment bounds """
        return -self.g

    def plot_poly(self, poly=None, ax=None, color='g', alpha=0.1, show=False, linewidth=1):
        """
        Plots the polytope on the provides axis or a new one.
        :param poly: poyltope.Polytope object
        :param ax: matplotlib axis
        :param color: polytope color
        :param alpha: opacity value [0, 1] (1 opaque, 0 transparent)
        :param show: True --> show the plot, False --> don't show the plot
        :return: if plot not shown, returns the axis
        """
        if self.h is None or self.g is None:
            raise NotImplementedError('Plotting error: Halfspace not defined yet!')
        if poly is None:
            poly = self.get_poly
        if ax is None:
            fig, ax = plt.subplots()
            ax.axis('equal')
        poly.plot(ax=ax, color=color, alpha=alpha, linestyle='solid', linewidth=linewidth)

        if show:
            plt.show()
        else:
            return ax

    def results(self):
        """
        Any desired results.
        :return:
        """
        raise NotImplementedError('results method not implemented')


class DRCVaRHalfspace(Halfspace):
    def __init__(self, alpha, eps, bound, num_samp, A=None, b=None, solver=None):
        """
        DR-CVaR halfspace
        :param alpha: alpha-worst cases considered for the CVaR computation
        :param eps: wasserstein ball radius for DR part
        :param bound: DR-CVaR bound
        :param num_samp: number of samples for empirically estimating the expected value term in the CVaR definition
        :param A: A matrix of environment bounds on the halfspace Ax <= b (c x 2 matrix)
        :param b: b vector of environment bounds on the halfspace Ax <= b (c x 1 vector)
        :param solver: cvxpy solver to use
        """
        super().__init__(A, b)

        self.alpha = alpha
        self.eps = eps
        self.delta = bound
        self.n = num_samp
        self.xi = None
        self.solver = solver

        self._def_opt_pb_vars()
        self._def_opt_pb()
        self.info = None

    def _def_opt_pb_vars(self):
        """ defines the optimization variables and parameters"""
        self._g = cp.Variable(1, name='g')
        self._tau = cp.Variable(1, name='tau')
        self._lam = cp.Variable(1, name='lam')
        self._eta = cp.Variable(self.n, name='eta')

        # instead of defining params for h and xi then multiplying them, this is the product h @ xi
        self._h_xi_prod = cp.Parameter((self.n, ), name='h_xi_prod')
        self._r = cp.Parameter((1, ), name='r')
        self.param_names = ['h_xi_prod', 'r']

    def _def_opt_pb(self):
        """ defines the optimization problem """
        a_k = [-1 / self.alpha, 0]
        b_k = [-1 / self.alpha, 0]
        c_k = [1 - 1 / self.alpha, 1]

        constraint = [self._lam * self.eps + 1 / self.n * cp.sum(self._eta) <= self.delta]  # cvar objective
        for i in range(self.n):
            for k in range(len(a_k)):
                constraint += [a_k[k] * self._h_xi_prod[i] + b_k[k] * (self._g - self._r) + c_k[k] * self._tau
                               <= self._eta[i]]
            constraint += [1 / self.alpha <= self._lam]
        self.problem = cp.Problem(cp.Minimize(self._g), constraint)

    def set_opt_pb_params(self, h, xi, r):
        """
        Set the optimization problem parameters
        :param h: halfspace normal
        :param xi: samples
        :param r: "radius" or padding to be added to the halfspace
        :return:
        """
        self.h = h
        self.xi = xi
        self._h_xi_prod.value = h @ xi
        self._r.value = r

    def solve_opt_pb(self, solver=None):
        """
        Solve the optimization problem and extract the result
        :param solver: CVXPY-supported solver e.g. ['ECOS', 'ECOS_BB', 'OSQP', 'SCIPY', 'SCS']
        :return: status (solved or not) and problem info (setup time, solve time, ...)
        """
        t0 = time.time()
        if solver is None:
            solver = self.solver
        if solver is None:
            self.problem.solve(verbose=False)
        else:
            self.problem.solve(verbose=False, solver=solver)
        t1 = time.time()
        info = {'solve_call_time': t1 - t0}

        if self.problem.status not in ["infeasible", "unbounded"]:
            self.g = self._g.value[0]
            info['setup_time'] = self.problem.solver_stats.setup_time
            info['solve_time'] = self.problem.solver_stats.solve_time
            self.info = info
            return True, info
        self.info = info
        return False, info

    def results(self):
        """
        Prints the found halfspace,
         computes the empirical value of the constraint using the samples (for verification),
         and prints the problem compute time info
        :return:
        """
        print('DR-CVaR separator: {}^T x + {} <= 0'.format(self.h, self.g))
        loss = - (self.h @ self.xi + (self.g - self._r.value))
        emp_drcvar, _ = get_empirical_drcvar(loss, self.alpha, self.eps)
        print('DR-CVaR: desired = {}, empirical = {}'.format(self.delta, emp_drcvar))
        results = {'loss': loss, 'emp_drcvar': emp_drcvar, 'g': self.g, 'h': self.h}
        print('Setup time = {}, solve time = {}, call time = {}'.format(
            self.info["setup_time"], self.info["solve_time"], self.info["solve_call_time"]))
        return results


class CVaRHalfspace(Halfspace):
    def __init__(self, alpha, bound, num_samp, loss_type, A=None, b=None, solver=None):
        """
        CVaR halfspace
        :param alpha: alpha-worst cases considered for the CVaR computation
        :param bound: DR-CVaR bound
        :param num_samp: number of samples for empirically estimating the expected value term in the CVaR definition
        :param loss_type: type of loss function. One of {continuous, clipped}
        :param A: A matrix of environment bounds on the halfspace Ax <= b (c x 2 matrix)
        :param b: b vector of environment bounds on the halfspace Ax <= b (c x 1 vector)
        :param solver: cvxpy solver to use
        """
        super().__init__(A, b)

        self.alpha = alpha
        self.delta = bound
        self.n = num_samp
        self.solver = solver
        self.xi = None
        if loss_type == 'continuous':
            self.loss_cvx = lambda h_xi_prod, g, r: - (h_xi_prod + g - r)
            self.loss = lambda h_xi_prod, g, r: - (h_xi_prod + g - r)
        elif loss_type == 'clipped':
            self.loss_cvx = lambda h_xi_prod, g, r: cp.maximum(-(h_xi_prod + g - r), 0)
            self.loss = lambda h_xi_prod, g, r: - np.maximum(-(h_xi_prod + g - r), 0)
        else:
            raise NotImplementedError('Loss type error: Type not defined.')

        self._def_opt_pb_vars()
        self._def_opt_pb()
        self.info = None

    def _def_opt_pb_vars(self):
        self._g = cp.Variable(1, name='g')
        self._tau = cp.Variable(1, name='tau')

        self._h_xi_prod = cp.Parameter((self.n, ), name='h_xi_prod')
        self._r = cp.Parameter((1, ), name='r')
        self.param_names = ['h_xi_prod', 'r']

    def _def_opt_pb(self):
        loss = self.loss_cvx(self._h_xi_prod, self._g, self._r)
        # cvar objective
        constraint = [1 / self.n * cp.sum(self._tau + 1/self.alpha * cp.maximum(loss - self._tau, 0)) <= self.delta]
        self.problem = cp.Problem(cp.Minimize(self._g), constraint)

    def set_opt_pb_params(self, h, xi, r):
        self.h = h
        self.xi = xi
        self._h_xi_prod.value = h @ xi
        self._r.value = r

    def solve_opt_pb(self, solver=None):
        """
        :param solver: ['ECOS', 'ECOS_BB', 'OSQP', 'SCIPY', 'SCS']
        :return: time to solve the problem
        """
        if solver is None:
            solver = self.solver
        t0 = time.time()
        if solver is None:
            self.problem.solve(verbose=False)
        else:
            self.problem.solve(verbose=False, solver=solver)
        t1 = time.time()
        info = {'solve_call_time': t1 - t0}

        if self.problem.status not in ["infeasible", "unbounded"]:
            self.g = self._g.value[0]
            info['setup_time'] = self.problem.solver_stats.setup_time
            info['solve_time'] = self.problem.solver_stats.solve_time
            self.info = info
            return True, info
        self.info = info
        return False, info

    def results(self):
        print('CVaR separator: {}^T x + {} <= 0'.format(self.h, self.g))
        loss = self.loss(self._h_xi_prod.value, self.g, self._r.value)
        emp_cvar, _ = get_empirical_cvar(loss, self.alpha)
        print('CVaR: desired = {}, empirical = {}'.format(self.delta, emp_cvar))
        results = {'loss': loss, 'emp_cvar': emp_cvar, 'g': self.g, 'h': self.h}
        print('Setup time = {}, solve time = {}, call time = {}'.format(
            self.info["setup_time"], self.info["solve_time"], self.info["solve_call_time"]))
        return results


class DeterministicHalfspace(Halfspace):
    def __init__(self, A=None, b=None):
        super().__init__(A, b)
        self.xi = None
        self._h_xi_prod = None
        self.r = None
        self.info = None

    def set_opt_pb_params(self, h, xi, r):
        self.h = h
        self.xi = xi
        self._h_xi_prod = h @ xi
        self.r = r

    def solve_opt_pb(self):
        t0 = time.time()
        g = self.r - self._h_xi_prod
        t1 = time.time()
        dt = t1 - t0
        info = {'solve_call_time': dt}
        self.g = g
        info['setup_time'] = 0
        info['solve_time'] = dt
        self.info = info
        return True, info

    def results(self):
        print('Deterministic separator: {}^T x + {} <= 0'.format(self.h, self.g))
        print('Setup time = {}, solve time = {}, call time = {}'.format(
            self.info["setup_time"], self.info["solve_time"], self.info["solve_call_time"]))
        results = {'g': self.g, 'h': self.h}
        return results


class MeanHalfspace(Halfspace):
    def __init__(self, A=None, b=None):
        super().__init__(A, b)
        self.xi = None
        self._h_xi_prod = None
        self.r = None
        self.info = None

    def set_opt_pb_params(self, h, xi, r):
        self.h = h
        self.xi = xi
        self._h_xi_prod = h @ xi
        self.r = r

    def solve_opt_pb(self):
        t0 = time.time()
        g = self.r - np.mean(self._h_xi_prod)
        t1 = time.time()
        dt = t1 - t0
        info = {'solve_call_time': dt}
        self.g = g
        info['setup_time'] = 0
        info['solve_time'] = dt
        self.info = info
        return True, info

    def results(self):
        print('Deterministic separator: {}^T x + {} <= 0'.format(self.h, self.g))
        print('Setup time = {}, solve time = {}, call time = {}'.format(
            self.info["setup_time"], self.info["solve_time"], self.info["solve_call_time"]))
        results = {'g': self.g, 'h': self.h}
        return results


# #################################################################################################################### #
# ################################################## Test Functions ################################################## #
# #################################################################################################################### #
def test_deterministic_halfspace():
    h = np.array([1., 0])
    h = h / np.linalg.norm(h)
    r = [1]
    ob = [2, -1]
    xi = np.array(ob)

    halfspace = DeterministicHalfspace()
    halfspace.set_opt_pb_params(h, xi, r)
    halfspace.solve_opt_pb()
    halfspace.results()

    halfspace.plot_poly(color='g', alpha=0.1, show=False)
    plt.scatter(xi[0], xi[1], color='k')
    plt.show()


def test_mean_halfspace():
    num_samp = 1000
    h = np.array([1., 1])
    h = h / np.linalg.norm(h)
    r = [1]
    ob = [2, -1]
    scale = [0.1, 0.1]
    xi = np.zeros((2, num_samp))
    xi[0, :] = generate_noise_samples(num_samp, ob[0], np.sqrt(scale[0]), dist='norm')
    xi[1, :] = generate_noise_samples(num_samp, ob[1], np.sqrt(scale[1]), dist='norm')

    halfspace = MeanHalfspace()
    halfspace.set_opt_pb_params(h, xi, r)
    halfspace.solve_opt_pb()
    halfspace.results()

    halfspace.plot_poly(color='g', alpha=0.1, show=False)
    plt.scatter(xi[0, :], xi[1, :], color='k')
    plt.show()


def test_dr_cvar_halfspace():
    alpha = 0.1
    eps = 0.01
    delta = -1
    num_samp = 1000
    h = np.array([1., 1])
    h = h / np.linalg.norm(h)
    r = [1]
    ob = [2, -1]
    scale = [0.1, 0.1]
    xi = np.zeros((2, num_samp))
    xi[0, :] = generate_noise_samples(num_samp, ob[0], np.sqrt(scale[0]), dist='norm')
    xi[1, :] = generate_noise_samples(num_samp, ob[1], np.sqrt(scale[1]), dist='norm')

    halfspace = DRCVaRHalfspace(alpha, eps, delta, num_samp)
    halfspace.set_opt_pb_params(h, xi, r)
    halfspace.solve_opt_pb()
    halfspace.results()

    halfspace.plot_poly(color='g', alpha=0.1, show=False)
    plt.scatter(xi[0, :], xi[1, :], color='k')
    plt.show()


def test_loss_fxn_comparison(xi):
    """
    Compares between the continuous and clipped loss functions
    """
    num_samp = xi.shape[1]
    h = np.array([1, 1.3])
    h = h / np.linalg.norm(h)
    delta = 0.3
    alpha = 0.1
    r = [0]

    halfspace_cont = CVaRHalfspace(alpha, delta, num_samp, loss_type='continuous')
    halfspace_clip = CVaRHalfspace(alpha, delta, num_samp, loss_type='clipped')

    # solve optimization problem
    halfspace_cont.set_opt_pb_params(h, xi, r)
    halfspace_cont.solve_opt_pb()

    fig, ax = plt.subplots()
    ax.axis('equal')
    plt.scatter(xi[0, :], xi[1, :], color='k', marker='.')
    halfspace_cont.plot_poly(ax=ax, color='r', alpha=0.1, show=False)
    legend = ['samples', 'cvar - continuous']
    if delta >= 0:
        halfspace_clip.set_opt_pb_params(h, xi, r)
        halfspace_clip.solve_opt_pb()
        halfspace_clip.plot_poly(ax=ax, color='b', alpha=0.1, show=False)
        legend.append('cvar - clipped')
    title = 'CVaR safe halfspace with delta={}'.format(delta)
    plt.title(title)
    plt.legend(legend)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    save_path = os.path.join('exp_data', 'loss_comparison_test')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'cvar_loss_compare_alpha={},_delta={}'.format(alpha, delta)) + '.png')
    plt.show()


def test_cvar_drcvar_comparison(xi):
    h = np.array([1, 1.3])
    h = h / np.linalg.norm(h)
    num_samp = xi.shape[1]
    delta = -1
    alpha = 0.1
    eps = 0.01
    r = [0]

    cvar_hs = CVaRHalfspace(alpha, delta, num_samp, loss_type='continuous')
    cvar_hs.set_opt_pb_params(h, xi, r)
    cvar_hs.solve_opt_pb()
    drcvar_hs = DRCVaRHalfspace(alpha, eps, delta, num_samp)
    drcvar_hs.set_opt_pb_params(h, xi, r)
    drcvar_hs.solve_opt_pb()

    title = 'CVaR vs DR-CVaR safe halfspace with delta={}, eps={}'.format(delta, eps)
    fig, ax = plt.subplots()
    ax.axis('equal')
    plt.scatter(xi[0, :], xi[1, :], color='k', marker='.')
    cvar_hs.plot_poly(ax=ax, color='r', alpha=0.1, show=False)
    drcvar_hs.plot_poly(ax=ax, color='b', alpha=0.1, show=False)
    plt.title(title)
    plt.legend(['samples', 'CVaR', 'DR-CVaR'])
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    save_path = os.path.join('exp_data', 'safe_halfspace_test')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path,
                             'cvar_vs_drcvar_compare_alpha={},_delta={},_eps={}'.format(alpha, delta, eps)) + '.png')
    plt.show()


def main():
    num_samp = 1000
    ob = np.array([0, 0])
    scale = (5, 5)
    xi = np.zeros((2, num_samp))
    xi[0, :] = generate_noise_samples(num_samp, ob[0], np.sqrt(scale[0]), dist='norm')
    xi[1, :] = generate_noise_samples(num_samp, ob[1], np.sqrt(scale[1]), dist='norm')

    test_loss_fxn_comparison(xi)
    test_cvar_drcvar_comparison(xi)


if __name__ == "__main__":
    test_mean_halfspace()
    # main()
