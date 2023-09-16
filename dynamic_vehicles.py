"""
Classes that define vehicles including their geometry.

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
July 19, 2023
"""
import numpy as np
from matplotlib import pyplot as plt

from dynamics import SingleIntegrator, DoubleIntegrator
from geometry_vehicles import CircularVehicle


class CircularSingleIntegrator(SingleIntegrator, CircularVehicle):
    def __init__(self, x0, dt, t0, radius):
        super().__init__(x0, dt, t0)
        CircularVehicle.__init__(self, self.get_pos, radius)

    def get_patch(self, color='tab:blue', alpha=0.3):
        self.center = self.get_pos
        return super().get_patch(color, alpha)


class CircularDoubleIntegrator(DoubleIntegrator, CircularVehicle):
    def __init__(self, x0, dt, t0, radius):
        super().__init__(x0, dt, t0)
        CircularVehicle.__init__(self, self.get_pos, radius)

    def get_patch(self, color='tab:blue', alpha=0.3):
        self.center = self.get_pos
        return super().get_patch(color, alpha)


# #################################################################################################################### #
# ################################################## Test Functions ################################################## #
# #################################################################################################################### #
def test_model(model, u):
    fig, axs = plt.subplots()
    traj_x, traj_y = [], []
    traj_x_sim, traj_y_sim = [], []
    patches = []
    for t in range(20):
        sim = model.sim(model.x, u)
        traj_x_sim.append(sim[0])
        traj_y_sim.append(sim[1])
        model.step(u)
        actual = model.get_pos
        traj_x.append(actual[0])
        traj_y.append(actual[1])
        patches.append(model.get_patch(color='tab:olive', alpha=0.5))

    for p in patches:
        axs.add_patch(p)
    s1 = plt.scatter(traj_x, traj_y, 50)
    s2 = plt.scatter(traj_x_sim, traj_y_sim, 10)
    plt.legend([patches[0], s1, s2], ['vehicle', 'actual', 'simulated'])
    axs.axis('equal')
    plt.show()


def test_circular_single_integrator():
    geom_model = CircularSingleIntegrator(x0=np.array([0, 0]), dt=0.2, t0=0, radius=0.5)
    u = np.array([1, 1])
    test_model(geom_model, u)


def test_circular_double_integrator():
    geom_model = CircularDoubleIntegrator(x0=np.array([0, 0, 1, 1]), dt=0.2, t0=0, radius=0.5)
    u = np.array([1, 0.1])
    test_model(geom_model, u)


def main():
    test_circular_single_integrator()
    test_circular_double_integrator()


if __name__ == "__main__":
    main()
