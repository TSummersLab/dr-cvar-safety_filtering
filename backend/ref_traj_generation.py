"""
Classes and functions that compute the reference trajectory for the ego vehicle.

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
July 14, 2023
"""
import time
import types

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

from dynamics import DoubleIntegrator


class MPCReferenceTrajectory:
    """
    Reference trajectory class
    """
    def __init__(self, horizon, A, B, Q=1., QT=3., R=1., state_Ax_leq_b=None, control_Ax_leq_b=None):
        """
        :param horizon: Trajectory horizon
        :param A: Dynamics matrix (n x n)
        :param B: Input matrix (n x m)
        :param Q: scalar cost for x - goal
        :param QT: scalar terminal cost xT - goal
        :param R: scalar control cost
        """
        self.horizon = horizon
        self.A, self.B = A, B
        self.n, self.m = B.shape
        self.Q, self.QT, self.R = Q, QT, R
        self.x_ref, self.u_ref = None, None
        self.state_Ax_leq_b = state_Ax_leq_b
        self.control_Ax_leq_b = control_Ax_leq_b
        self.constraints, self.objective, self.problem = None, None, None

        self._def_opt_pb_vars()
        self._def_opt_pb()

        self.redefine_problem = types.MethodType(self.define_problem, self)

    def _def_opt_pb_vars(self):
        self.x = cp.Variable((self.n, self.horizon + 1), name='x')
        self.u = cp.Variable((self.m, self.horizon), name='u')
        self.x0 = cp.Parameter((self.n,), name='x0')  # initial state
        self.goal = cp.Parameter((self.n,), name='xT')  # final desired state

    def _def_opt_pb(self):
        # initial state
        self.constraints = [self.x[:, 0] == self.x0]

        for t in range(self.horizon):
            # dynamics
            self.constraints += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t]]

            # state bounds
            if self.state_Ax_leq_b is not None:
                self.constraints += [np.array(self.state_Ax_leq_b['A']) @ self.x[:, t + 1]
                                     <= np.array(self.state_Ax_leq_b['b'])]

            # input bounds
            if self.control_Ax_leq_b is not None:
                self.constraints += [self.control_Ax_leq_b['A'] @ self.u[:, t]
                                     <= self.control_Ax_leq_b['b']]

        # objective
        x_final = self.x[:, -1]
        self.objective = cp.sum_squares((x_final - self.goal) * self.QT)
        for t in range(1, self.horizon):
            self.objective += cp.sum_squares((self.x[:, t] - self.goal) * self.Q)
            self.objective += cp.sum_squares(self.u[:, t] * self.R)

        self.define_problem()

    def define_problem(self):
        self.problem = cp.Problem(cp.Minimize(self.objective), self.constraints)

    def set_opt_pb_params(self, x0, goal):
        self.x0.value = x0
        self.goal.value = goal

    def solve_opt_pb(self, solver=None):
        """
        :param solver: ['ECOS', 'ECOS_BB', 'OSQP', 'SCIPY', 'SCS']
        :return: time to solve the problem
        """
        t0 = time.time()
        if solver is None:
            self.problem.solve(verbose=False)
        else:
            self.problem.solve(verbose=False, solver=solver)
        t1 = time.time()
        info = {'solve_call_time': t1 - t0}

        if self.problem.status not in ["infeasible", "unbounded"]:
            self.x_ref, self.u_ref = self.x.value, self.u.value
            info['setup_time'] = self.problem.solver_stats.setup_time
            info['solve_time'] = self.problem.solver_stats.solve_time
            self.info = info
            return True, info
        self.info = info
        return False, info


# #################################################################################################################### #
# ################################################## Test Functions ################################################## #
# #################################################################################################################### #
def test_mpc_ref_traj():
    model = DoubleIntegrator(x0=np.array([0, 0, 0, 0]), dt=0.2, t0=0)
    mpc = MPCReferenceTrajectory(horizon=10, A=model.A, B=model.B, Q=1., QT=3., R=1.)
    goal = np.array([4, 1, 0.2, 0.01])

    traj_x, traj_y = [], []
    for t in range(50):
        mpc.set_opt_pb_params(model.get_state, goal)
        mpc.solve_opt_pb()
        u = mpc.u_ref[:, 0]
        model.step(u)
        new_pos = model.get_pos
        traj_x.append(new_pos[0])
        traj_y.append(new_pos[1])

    fig, axs = plt.subplots()
    plt.scatter(traj_x, traj_y, 10)
    plt.legend(['actual'])
    axs.axis('equal')
    plt.show()
    print('Final state: ', model.get_state)
    print('State at the end of the horizon: ', mpc.x_ref[:, -1])
    print('Desired state at the end of the horizon: ', goal)
    print('Error: ', np.linalg.norm(mpc.x_ref[:, -1] - goal))


def main():
    test_mpc_ref_traj()


if __name__ == "__main__":
    main()

