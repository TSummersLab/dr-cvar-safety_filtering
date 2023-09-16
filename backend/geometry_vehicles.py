"""
Classes that define vehicles including their geometry.

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
July 16, 2023
"""
import matplotlib.patches as patch
import matplotlib.pyplot as plt


class VehicleGeometry:
    """
    Top level class for defining a vehicle's geometry and related functions
    """
    def __init__(self):
        pass

    def get_patch(self):
        raise NotImplementedError('Vehicle Geometry Error: get_patch not defined')


class CircularVehicle(VehicleGeometry):
    """
    2D Circular vehicle geometry
    """
    def __init__(self, center, radius):
        """
        :param center: position of the vehicle's center
        :param radius: vehicle radius
        """
        super().__init__()
        self.center = center
        self.radius = radius

    def get_patch(self, color='tab:blue', alpha=0.3):
        return patch.Circle(self.center, self.radius, facecolor=color, alpha=alpha)


# #################################################################################################################### #
# ################################################## Test Functions ################################################## #
# #################################################################################################################### #
def test_circular_vehicle():
    veh0 = CircularVehicle([0, 0], 0.5)
    veh1 = CircularVehicle([1, 0], 0.5)
    veh2 = CircularVehicle([0, 2], 1)

    fig, axs = plt.subplots()
    axs.add_patch(veh0.get_patch('tab:blue', alpha=0.5))
    axs.add_patch(veh1.get_patch('tab:green', alpha=0.3))
    axs.add_patch(veh2.get_patch('tab:red', alpha=1.0))
    plt.scatter([veh0.center[0], veh1.center[0], veh2.center[0]],
                [veh0.center[1], veh1.center[1], veh2.center[1]], color='k', marker='x')
    axs.axis('equal')
    plt.show()


if __name__ == "__main__":
    test_circular_vehicle()
