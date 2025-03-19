import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class xArm6(ManipulatorModel):
    """
    xArm6 is a single arm robot created by UFactory.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/xarm6/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", 
            values=np.array((0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

        self.set_joint_attribute(attrib="frictionloss", 
            values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1)))

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        # return None
        # return "XArmGripper"
        return {"right": "XArmGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_panda"}

    @property
    def init_qpos(self):
        # NOTE: When xarm6 is placed on table directly and all joint angles are 0, then with random initiliazation, the gripper
        # sometimes hits the table at initial position. The init joints listed below are to avoid this.
        # return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # For EEF z 0.1 above table when xarm6 is placed on table directly
        # return np.array([3.74565364e-03, -9.37793861e-01, -2.82236445e-03, 6.77735371e-04, 9.08002899e-01, -8.60226130e-03])

        # For EEF z 0.05 above table when xarm6 is placed on table directly
        # np.array([8.84830716e-03, -6.66687447e-01, 4.89064751e-02, 1.94762656e-04, 5.84571386e-01, -3.86422130e-03])

        # From maniskill experiments
        return np.array([6.672368e-4, 2.2206e-1, -1.2311444, 1.6927806e-4, 1.0088931, 7.304605e-4])

    @property
    def base_xpos_offset(self):
        return {
            # "bins": (-0.5, -0.1, 0),
            "bins": (-0.407, 0.0, 0.171),    # PickPlace
            "bins_nomount": (0.307, 0.0, -0.158),
            "empty": (-0.6, 0, 0),
            # "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "table": lambda table_length: (-0.407, 0, 0.171),      # Reach/Lift
            # "table": lambda table_length: (-0.437, 0, 0.171)
            "table_nomount": lambda table_length: (table_length/2 - 0.05, 0, 0),
            # "table_nomount": lambda table_length: (0.307, 0, -0.1333),
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
