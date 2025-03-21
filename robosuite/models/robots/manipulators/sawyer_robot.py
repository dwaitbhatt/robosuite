import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Sawyer(ManipulatorModel):
    """
    Sawyer is a witty single-arm robot designed by Rethink Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "RethinkGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_sawyer"}

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "bins_nomount": (0.54968339, 0.16029991, -0.01774591),
            "empty": (-0.6, 0, 0),
            # "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "table": lambda table_length: (-0.6497, -0.1603, 0.03075),
            # "table_nomount": lambda table_length: (table_length/2 + 0.243, 0.14994, 0),
            "table_nomount": lambda table_length: (0.5496, 0.16030, 0.00725409),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
