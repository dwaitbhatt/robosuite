from .gripper_model import GripperModel
from .gripper_factory import gripper_factory
from .gripper_tester import GripperTester

from .panda_gripper import PandaGripper, PandaTouchGripper, PandaTactileGripper
from .rethink_gripper import RethinkGripper, RethinkTouchGripper
from .robotiq_85_gripper import Robotiq85Gripper, Robotiq85TouchGripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper, RobotiqThreeFingerDexterousGripper
from .panda_gripper import PandaGripper
from .jaco_three_finger_gripper import JacoThreeFingerGripper, JacoThreeFingerDexterousGripper
from .robotiq_140_gripper import Robotiq140Gripper
from .wiping_gripper import WipingGripper
from .xarm_gripper import XArmGripper
from .bd_gripper import BDGripper
from .null_gripper import NullGripper
from .inspire_hands import InspireLeftHand, InspireRightHand
from .fourier_hands import FourierLeftHand, FourierRightHand

GRIPPER_MAPPING = {
    "RethinkGripper": RethinkGripper,
    "RethinkTouchGripper": RethinkTouchGripper,
    "PandaGripper": PandaGripper,
    "PandaTouchGripper": PandaTouchGripper,    
    "PandaTactileGripper": PandaTactileGripper,
    "JacoThreeFingerGripper": JacoThreeFingerGripper,
    "JacoThreeFingerDexterousGripper": JacoThreeFingerDexterousGripper,
    "WipingGripper": WipingGripper,
    "Robotiq85Gripper": Robotiq85Gripper,
    "Robotiq85TouchGripper": Robotiq85TouchGripper,
    "Robotiq140Gripper": Robotiq140Gripper,
    "RobotiqThreeFingerGripper": RobotiqThreeFingerGripper,
    "RobotiqThreeFingerDexterousGripper": RobotiqThreeFingerDexterousGripper,
    "XArmGripper": XArmGripper,
    "BDGripper": BDGripper,
    "InspireLeftHand": InspireLeftHand,
    "InspireRightHand": InspireRightHand,
    "FourierLeftHand": FourierLeftHand,
    "FourierRightHand": FourierRightHand,
    None: NullGripper,
}

ALL_GRIPPERS = GRIPPER_MAPPING.keys()


def register_gripper(target_class):
    GRIPPER_MAPPING[target_class.__name__] = target_class
    return target_class
