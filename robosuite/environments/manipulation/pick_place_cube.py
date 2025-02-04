from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.environments.base import register_env

# TODO: Copied from track_cube.py, need to modify to pick and place cube
@register_env
class PickPlaceCube(ManipulationEnv):
    """
    This class corresponds to a pick and place task matching the real-world environment.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer_cube = placement_initializer

        self.cube_half_size = 0.02
        self.target_pad_half_size = 0.03

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="NullMount",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        
        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Set table color after creating the arena
        mujoco_arena.table_visual.set("rgba", "0.1 0.1 0.1 1")

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Adjust base pose to place the robot on top of the table
        xpos = self.table_offset.copy()
        if "table_nomount" in self.robots[0].robot_model.base_xpos_offset:
            xpos -= self.robots[0].robot_model.base_xpos_offset["table_nomount"](self.table_full_size[0])
        else:
            raise ValueError(f"Offset for table arena without mount is not defined in robot_model for {self.robots[0].robot_model.name}.\
                             Please specify this offset to ensure initial eef position (x,y) is same [-0.1, 0] across different robots.")
        self.robots[0].robot_model.set_base_xpos(xpos)

        # initialize objects of interest
        tex_attrib_wood = {
            "type": "cube",
        }
        mat_attrib_wood = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib_wood,
            mat_attrib=mat_attrib_wood,
        )

        tex_attrib_paper = {
            "type": "2d",
        }
        mat_attrib_paper = {
            "texrepeat": "1 1",
            "specular": "0",
            "shininess": "0",
        }
        paper = CustomMaterial(
            texture="PlasterWhite",
            tex_name="paper",
            mat_name="paper_mat",
            tex_attrib=tex_attrib_paper,
            mat_attrib=mat_attrib_paper,
        )
        
        self.cube = BoxObject(
            name="cube",
            size=[self.cube_half_size] * 3,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        self.target_pad = BoxObject(
            name="target_pad",
            size=[self.target_pad_half_size, self.target_pad_half_size, 0.001],
            rgba=[0.9, 0.9, 0.9, 1],
            material=paper,
        )
        
        # Similar to Maniskill, cube is always behind the target in x
        self.placement_initializer_cube = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cube,
            x_range=(-0.05, 0.1),
            y_range=(-0.25, 0.25),
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.placement_initializer_target_pad = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.target_pad,
            x_range=(0.15, 0.3),
            y_range=(-0.25, 0.25),
            rotation=0,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube, self.target_pad],
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        if self.use_object_obs:
            # Get prefix from robot model to avoid naming clashes for multiple robots
            pf = self.robots[0].robot_model.naming_prefix

            @sensor(modality="object")
            def cube_pos(obs_cache):
                return self._cube_xpos

            @sensor(modality="object")
            def cube_to_eef_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
                    if "cube_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality="object")
            def target_pos(obs_cache):
                return self._target_pos
            
            @sensor(modality="object")
            def target_to_eef_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["target_pos"]
                    if "target_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cube_pos, cube_to_eef_pos, target_pos, target_to_eef_pos]
            names = [f"cube_pos", f"cube_to_{pf}eef_pos", f"target_pos", f"target_to_{pf}eef_pos"]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset cube position using placement initializer
        if not self.deterministic_reset:
            cube_placement = self.placement_initializer_cube.sample()
            cube_pos, cube_quat, _ = cube_placement[self.cube.name]
            self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([cube_pos, cube_quat]))
            
            target_pad_placement = self.placement_initializer_target_pad.sample()
            target_pad_pos, target_pad_quat, _ = target_pad_placement[self.target_pad.name]
            self.sim.data.set_joint_qpos(self.target_pad.joints[0], np.concatenate([target_pad_pos, target_pad_quat]))
            
    def _setup_references(self):
        """
        Sets up references to important components.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id("cube_main")
        self.target_pad_body_id = self.sim.model.body_name2id("target_pad_main")
        
    def staged_rewards(self, action):
        """
        Reward function for the task. Similar to PickPlace in bins arena.
        """
        # Max reward for each stage
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # reaching reward
        dist = self._gripper_to_target(
            gripper=self.robots[0].gripper, target=self.cube.root_body, target_type="body", return_distance=True
        )
        r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        # grasping reward
        r_grasp = int(self._check_grasp(self.robots[0].gripper, self.cube.root_body)) * grasp_mult

        # lifting reward
        r_lift = 0.0
        if r_grasp > 0:
            z_target = self._target_pos[2] + 0.1
            z_dist = np.maximum(z_target - self._cube_xpos[2], 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * z_dist)) * (lift_mult - grasp_mult)

        # hover reward
        r_hover = 0.0
        x_check = np.abs(self._cube_xpos[0] - self._target_pos[0]) < self.target_pad_half_size
        y_check = np.abs(self._cube_xpos[1] - self._target_pos[1]) < self.target_pad_half_size
        xy_dist = np.linalg.norm(self._cube_xpos[:2] - self._target_pos[:2])
        if x_check and y_check:
            r_hover = lift_mult + (1 - np.tanh(10.0 * xy_dist)) * (hover_mult - lift_mult)

        return r_reach, r_grasp, r_lift, r_hover

    def reward(self, action):
        """
        Reward function for the task.
        """
        if self._check_success():
            return 1.0
        
        reward = 0.0
        if self.reward_shaping:
            rewards = self.staged_rewards(action)
            reward += max(rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _check_success(self):
        """
        Check if the task has been completed successfully.
        """
        cube_to_target_dist = np.linalg.norm(self._cube_xpos - self._target_pos)
        threshold = 0.005
        is_cube_grasped = self._check_grasp(self.robots[0].gripper, self.cube.root_body)
        return (not is_cube_grasped) and (cube_to_target_dist <= threshold)
    

    def _get_eef_xpos(self, arm):
        """
        Grabs End Effector position as specifed by the arm argument

        Args:
            arm (str): Arm name

        Returns:
            np.array: End effector(x,y,z)
        """
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])
    
    @property
    def _cube_xpos(self):
        """
        Grabs the position of the cube.

        Returns:
            np.array: Cube (x,y,z)
        """
        return self.sim.data.body_xpos[self.cube_body_id]
    
    @property
    def _target_pad_xpos(self):
        return self.sim.data.body_xpos[self.target_pad_body_id]
    
    @property
    def _target_pos(self):
        return self._target_pad_xpos + np.array([0, 0, self.cube_half_size])
    