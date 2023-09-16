"""
Filters for motion planning corrections using precomputed halfspaces.

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
July 19, 2023
"""
import types
import time
import cvxpy as cp
from dynamics import *


class MPCFilter:
    """
    Class for MPC-based safety filtering using safe halfspaces
    """
    def __init__(self, num_obstacles: int, dynamic_model, horizon: int, Q, QT, R,
                 state_Ax_leq_b=None, measurement_Ax_leq_b=None, control_Ax_leq_b=None,
                 id=None):
        """
        :param num_obstacles: number of obstacles
        :param dynamic_model: dynamics model. Must be a dynamics.DTVehicleDynamics-based model
        :param horizon: MPC horizon
        :param Q: state cost matrix (n x n) for (n x 1) state
        :param QT: terminal state cost matrix (n x n)
        :param R: control cost matrix (m x m) for (m x 1) inputs
        :param state_Ax_leq_b: dictionary with keys A, b such that A * x <= b where x is the state
        :param measurement_Ax_leq_b: dictionary with keys A, b such that A * y <= b where y is the measurement
        :param control_Ax_leq_b: dictionary with keys A, b such that A * u <= b where u is the control
        :param id: MPC filter id (optional string)
        """
        self.num_obstacles = num_obstacles
        self.dyn_model = dynamic_model
        self.horizon = horizon
        self.Q, self.QT, self.R = Q, QT, R
        self.state_Ax_leq_b = state_Ax_leq_b
        self.measurement_Ax_leq_b = measurement_Ax_leq_b
        self.control_Ax_leq_b = control_Ax_leq_b
        self.id = id if id is not None else ''

        # optimization problem and its terms
        self.constraints, self.objective, self.problem = None, None, None

        # filter outputs
        self.x_mpc, self.u_mpc, self.info = None, None, None
        self.solved = False

        # set up the filter
        self._def_opt_pb_vars()
        self._def_opt_pb()

        # create an alias redefine_problem = define_problem
        self.redefine_problem = types.MethodType(self.define_problem, self)

    def _def_opt_pb_vars(self):
        """
        Define optimization problem variables and parameters
        :return:
        """
        self.x = cp.Variable((self.dyn_model.n, self.horizon+1), name='x' + self.id)
        self.u = cp.Variable((self.dyn_model.m, self.horizon), name='u' + self.id)

        self.halfspaces_A = [cp.Parameter((self.num_obstacles, self.dyn_model.d),
                                          name='A' + self.id + '_horizon_step=' + str(t)) for t in range(self.horizon)]
        self.halfspaces_b = cp.Parameter((self.num_obstacles, self.horizon), name='b' + self.id)
        self.x_ref = cp.Parameter((self.dyn_model.n, self.horizon+1), name='xref' + self.id)

    def _def_opt_pb(self):
        """
        Define the optimization problem objective and constraints
        :return:
        """
        # # # quadratic objective
        self.objective = 0
        # input cost
        for t in range(self.horizon):
            self.objective += cp.sum_squares(self.u[:, t] @ self.R)
        # state cost
        for t in range(1, self.horizon-1):
            self.objective += cp.sum_squares((self.x[:, t] - self.x_ref[:, t]) @ self.Q)
        # terminal state cost
        self.objective += cp.sum_squares((self.x[:, self.horizon] - self.x_ref[:, self.horizon]) @ self.QT)

        # # # constraints
        self.constraints = []
        self.constraints += [self.x[:, 0] == self.x_ref[:, 0]]  # initial state
        for t in range(self.horizon):
            # linear dynamics
            self.constraints += [self.x[:, t + 1] == self.dyn_model.A @ self.x[:, t] + self.dyn_model.B @ self.u[:, t]]
            # safe-halfspace
            self.constraints += [self.halfspaces_A[t] @ self.dyn_model.C @ self.x[:, t + 1] <= self.halfspaces_b[:, t]]
            # state bounds
            if self.state_Ax_leq_b is not None:
                self.constraints += [np.array(self.state_Ax_leq_b['A']) @ self.x[:, t + 1]
                                     <= np.array(self.state_Ax_leq_b['b'])]
            # output bounds
            if self.measurement_Ax_leq_b is not None:
                self.constraints += [np.array(self.measurement_Ax_leq_b['A']) @ self.dyn_model.C @ self.x[:, t + 1]
                                     <= np.array(self.measurement_Ax_leq_b['b'])]
            # input bounds
            if self.control_Ax_leq_b is not None:
                self.constraints += [self.control_Ax_leq_b['A'] @ self.u[:, t]
                                     <= self.control_Ax_leq_b['b']]
        self.define_problem()

    def define_problem(self):
        """
        Define the filter optimization problem
        :return:
        """
        self.problem = cp.Problem(cp.Minimize(self.objective), self.constraints)

    def set_opt_pb_params(self, halfspaces_A, halfspaces_b, x_ref):
        """
        Set the parameter values
        :param halfspaces_A: all A vectors for the safe halfspaces A * p <= b
        :param halfspaces_b: all B vectors for the safe halfspaces A * p <= b
        :param x_ref: reference trajectory
        :return:
        """
        for t, halfspace_A in enumerate(halfspaces_A):
            self.halfspaces_A[t].value = np.reshape(halfspace_A, (self.num_obstacles, self.dyn_model.m))
        self.halfspaces_b.value = np.reshape(halfspaces_b, (self.num_obstacles, self.horizon))
        self.x_ref.value = x_ref

    def solve_opt_pb(self, solver=None):
        """
        Solve the optimization problem and obtain the solution
        :param solver: ['ECOS', 'ECOS_BB', 'OSQP', 'SCIPY', 'SCS']. If None: CVXPY automatically chooses a solver
        :return: time to solve the problem and the solver info
        """
        t0 = time.time()
        if solver is None:
            self.problem.solve(verbose=False)
        else:
            self.problem.solve(verbose=False, solver=solver)
        t1 = time.time()
        info = {'solve_call_time': t1 - t0}

        if self.problem.status not in ["infeasible", "unbounded"]:
            # problem feasible
            self.x_mpc, self.u_mpc = self.x.value, self.u.value
            info['setup_time'] = self.problem.solver_stats.setup_time
            info['solve_time'] = self.problem.solver_stats.solve_time
            self.info = info
            self.solved = True
        else:
            if self.x_mpc is None or self.u_mpc is None:
                raise NotImplementedError('MPC Filter error: Infeasible with no x_mpc or u_mpc values.')
            # problem infeasible
            no_control = np.zeros([self.dyn_model.m, ])
            no_ctrl_state = self.dyn_model.sim(self.x_mpc[:, -1], no_control)
            self.x_mpc = self._shift_mpc_output(self.x_mpc, no_ctrl_state)
            self.u_mpc = self._shift_mpc_output(self.u_mpc, no_control)
            info['setup_time'] = None
            info['solve_time'] = None
            self.info = info
            self.solved = False
        return self.solved, self.info

    @staticmethod
    def _shift_mpc_output(output, final_val=None):
        """
        Shift the output value to the left and append a final value.
        This is useful when the MPC problem is infeasible to get the latest trajectory.
        :param output: output to be shifted (e.g. x_mpc)
        :param final_val: new final value based on some infeasibility handling methodology (e.g. breaking controller)
        :return:
        """
        result = np.empty_like(output, dtype=float)
        result[:, :-1] = output[:, 1:]
        if final_val is None:
            result[:, -1] = np.inf
        else:
            result[:, -1] = final_val
        return result


class MPCFilterWithSlack(MPCFilter):
    def __init__(self, num_obstacles: int, dynamic_model, horizon: int, Q, QT, R,
                 state_Ax_leq_b=None, measurement_Ax_leq_b=None, control_Ax_leq_b=None, id=None):
        # set up the filter
        self.slack_cost = 1000
        super().__init__(num_obstacles, dynamic_model, horizon, Q, QT, R,
                         state_Ax_leq_b, measurement_Ax_leq_b, control_Ax_leq_b, id)
        self._def_opt_pb_vars()
        self._def_opt_pb()

    def _def_opt_pb_vars(self):
        """
        Define optimization problem variables and parameters
        :return:
        """
        super()._def_opt_pb_vars()
        self.slack = cp.Variable((self.num_obstacles, self.horizon), name='slack' + self.id)

    def _def_opt_pb(self):
        """
        Define the optimization problem objective and constraints
        :return:
        """
        super()._def_opt_pb()

        # slack variable cost
        for t in range(self.horizon):
            self.objective += cp.norm(self.slack_cost * self.slack[:, t], 1)

        # # # constraints
        self.constraints = []
        self.constraints += [self.x[:, 0] == self.x_ref[:, 0]]  # initial state
        for t in range(self.horizon):
            # linear dynamics
            self.constraints += [self.x[:, t + 1] == self.dyn_model.A @ self.x[:, t] + self.dyn_model.B @ self.u[:, t]]
            # safe-halfspace
            self.constraints += [self.halfspaces_A[t] @ self.dyn_model.C @ self.x[:, t + 1]
                                 <= self.halfspaces_b[:, t] + self.slack[:, t]]
            # state bounds
            if self.state_Ax_leq_b is not None:
                self.constraints += [np.array(self.state_Ax_leq_b['A']) @ self.x[:, t + 1]
                                     <= np.array(self.state_Ax_leq_b['b'])]
            # output bounds
            if self.measurement_Ax_leq_b is not None:
                self.constraints += [np.array(self.measurement_Ax_leq_b['A']) @ self.dyn_model.C @ self.x[:, t + 1]
                                     <= np.array(self.measurement_Ax_leq_b['b'])]
            # input bounds
            if self.control_Ax_leq_b is not None:
                self.constraints += [self.control_Ax_leq_b['A'] @ self.u[:, t]
                                     <= self.control_Ax_leq_b['b']]
        self.constraints += [self.slack >= 0]
        for t in range(self.horizon-1):
            self.constraints += [self.slack[:, t+1] >= self.slack[:, t]]
        self.define_problem()

    @property
    def get_slack_values(self):
        return self.slack.value
