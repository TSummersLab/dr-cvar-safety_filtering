"""
Classes that define vehicle dynamics.

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
July 15, 2023
"""
import matplotlib.pyplot as plt
import numpy as np


class DTVehicleDynamics:
    """
    Class that defines the discrete-time dynamics of a vehicle.
    """
    def __init__(self, A, B, C, x0, dt, t0):
        """
        :param A: dynamics matrix
        :param B: input matrix
        :param C: measurement matrix
        :param x0: initial state
        :param dt: discrete time step
        :param t0: initial time
        """
        self.A, self.B, self.C = A, B, C  # dynamics matrices
        self.n, self.m = B.shape  # num states and inputs
        self.d = C.shape[0]  # num measurements
        self.dt, self.t = dt, t0

        self.x = np.array(x0)
        self.y = self._measurement_update(self.x)
        self.u = None

    @property
    def get_state(self):
        return self.x

    @property
    def get_control(self):
        return self.u

    @property
    def get_pos(self):
        raise NotImplementedError('DTVehicleDynamics Error: get_pos property not defined')

    @property
    def get_vel(self):
        raise NotImplementedError('DTVehicleDynamics Error: get_vel property not defined')

    @property
    def get_measurement(self):
        return self.y

    def _dynamics(self, x, u):
        """
        Deterministic dynamics equations, both state and measurement
        :param x: state
        :param u: input
        :return: returns the next state and the corresponding measurement
        """
        x = self._state_update(x, u)
        y = self._measurement_update(x)
        return x, y

    def _state_update(self, x, u):
        """ state updated equation """
        return self.A @ x + self.B @ u

    def _measurement_update(self, x):
        """ measurement update equation """
        return self.C @ x

    def sim(self, x, u, return_measurement=False):
        """
        Simulates the dynamical system one step forward.
        Does not update the class instance variables
        :param x: state
        :param u: input
        :param return_measurement: True --> returns x, y. False, returns x only
        :return: x, y  or just x
        """
        x, y = self._dynamics(x, u)
        if return_measurement:
            return x, y
        else:
            return x

    def step(self, u, return_measurement=False):
        """
        Steps the dynamical system one step forward.
        Updates the class instance variables self.x, self.u, self.y, and self.t
        :param u: input
        :param return_measurement: True --> returns x, y. False, returns x only
        :return: x, y  or just x
        """
        self.u = np.array(u)
        self.x, self.y = self._dynamics(self.x, self.u)
        self.t += self.dt
        if return_measurement:
            return self.x, self.y
        else:
            return self.x

    def overwrite_state(self, state):
        """ Overwrite self.x by provided state. Can be used to model state uncertainty. """
        self.x = state

    def add_to_state(self, w):
        """ Updates self.x by adding w. Can be used to model additive state uncertainty. """
        self.x += w

    def overwrite_measurement(self, meas):
        """ Overwrite self.y by provided measurement. Can be used to model measurement uncertainty. """
        self.y = meas

    def add_to_measurement(self, v):
        """ Updates self.y by adding v. Can be used to model additive measurement uncertainty. """
        self.y += v


class SingleIntegrator(DTVehicleDynamics):
    """
    2D Single integrator dynamics class
    """
    def __init__(self, x0, dt, t0):
        """
        :param x0: initial state
        :param dt: discrete timestep
        :param t0: initial time
        """
        A, B, C = np.eye(2), np.eye(2) * dt, np.eye(2)
        super().__init__(A, B, C, x0, dt, t0)

    @property
    def get_pos(self):
        return self.x

    @property
    def get_vel(self):
        return self.u


class DoubleIntegrator(DTVehicleDynamics):
    """
    2D Double integrator dynamics class
    """
    def __init__(self, x0, dt, t0):
        """
        :param x0: initial state
        :param dt: discrete timestep
        :param t0: initial time
        """
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        B = np.array([[0.5 * dt ** 2, 0],
                      [0, 0.5 * dt ** 2],
                      [dt, 0],
                      [0, dt]])
        C = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        Cv = np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1]])
        self.Cv = Cv
        super().__init__(A, B, C, x0, dt, t0)

    @property
    def get_pos(self):
        return self.C @ self.x

    @property
    def get_vel(self):
        return self.Cv @ self.x


# #################################################################################################################### #
# ################################################## Test Functions ################################################## #
# #################################################################################################################### #
def test_model(model, u):
    fig, axs = plt.subplots()
    traj_x, traj_y = [], []
    traj_x_sim, traj_y_sim = [], []
    for t in range(100):
        sim = model.sim(model.x, u)
        traj_x_sim.append(sim[0])
        traj_y_sim.append(sim[1])
        model.step(u)
        actual = model.get_pos
        traj_x.append(actual[0])
        traj_y.append(actual[1])

    plt.scatter(traj_x, traj_y, 50)
    plt.scatter(traj_x_sim, traj_y_sim, 10)
    plt.legend(['actual', 'simulated'])
    axs.axis('equal')
    plt.show()


def test_single_integrator():
    model = SingleIntegrator(x0=np.array([0, 0]), dt=0.1, t0=0)
    u = np.array([0.3, 0.1])
    test_model(model, u)


def test_double_integrator():
    model = DoubleIntegrator(x0=np.array([0, 0, 1, 1]), dt=0.1, t0=0)
    u = np.array([0.5, 0.1])
    test_model(model, u)


def main():
    test_single_integrator()
    test_double_integrator()


if __name__ == "__main__":
    main()
