from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, BallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.environments.base import register_env


class TrackCube(SingleArmEnv):
    """
    This class corresponds to tracking the position of a moving cube object by a robot arm.
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
        static_cube=True,
        cube_init_pos_range=[-0.35, 0.35],
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
        self.placement_initializer = placement_initializer

        # cube movement
        self.static_cube = static_cube
        self.cube_init_pos_range = cube_init_pos_range
        self.cube_velocity = np.random.uniform(-0.03, 0.03, size=3)
        self.cube_velocity[2] = 0  # Set z-velocity to 0 to keep the cube on the table

        # Set initial qpos to avoid eef collision with table (since we're not using a mount)
        # Determined by bringing default init eef position for xArm6 up in z axis until collision is avoided
        initial_qpos_dict = {
            "xArm6": np.array([1.55499306e-03, -1.02603911e+00, 5.01407118e-02, 
                                2.35911318e-04, 9.74043449e-01, 1.70899325e-03]),
            "Panda": np.array([ 0.02393744, -0.12377735, -0.02544775, 
                                -2.55581167, -0.00426772,  2.43323904, 0.78727797]),
            "Sawyer": np.array([0.01857909, -1.46858229, -0.06723376,  
                                2.10537169,  0.00965024,  0.93301487, -1.45014191])
        }
        initial_qpos = initial_qpos_dict[robots]

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=None,
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
            initial_qpos=initial_qpos,
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
        mujoco_arena.table_top.set("rgba", "0.1 0.1 0.1 1")

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Adjust base pose to place the robot on top of the table
        # robot_base_position = self.robots[0].robot_model.base_xpos_offset["bins"]
        robot_base_position = self.table_offset.copy()
        robot_base_position[0] -= self.table_full_size[0]/2 - 0.05
        self.robots[0].robot_model.set_base_xpos(robot_base_position)


        # # Add a red ball to indicate origin
        # origin_ball = BallObject(
        #     name="origin_ball",
        #     size=[0.2],
        #     rgba=[1, 0, 0, 1],
        #     obj_type='visual',
        #     joints=None
        # )
        # origin_ball.get_obj().set("pos", "0 0 0")

        # # Add a vertical line to indicate z axis
        # z_axis_line = BoxObject(
        #     name="z_axis_line",
        #     size=[0.005, 0.005, 1],  # Thin, long box to represent a line
        #     rgba=[0, 0, 1, 1],  # Blue color
        #     obj_type='visual',
        #     joints=None
        # )
        # z_axis_line.get_obj().set("pos", "0 0 0.5")  # Position it at the origin, extending upwards

        # # Add an orange ball to indicate center of table
        # table_center_ball = BallObject(
        #     name="table_center_ball",
        #     size=[0.1],  # Smaller size than the origin ball
        #     rgba=[1, 0.5, 0, 1],  # Orange color
        #     obj_type='visual',
        #     joints=None
        # )
        # table_center_pos = self.table_offset.copy()
        # table_center_ball.get_obj().set("pos", " ".join(map(str, table_center_pos)))


        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        # initialize objects of interest
        self.cube = BoxObject(
            name="cube",
            size=[0.025, 0.025, 0.025],
            rgba=[0, 0.6, 0.13, 1],
            material=greenwood,
        )
        
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cube,
            x_range=self.cube_init_pos_range,
            y_range=self.cube_init_pos_range,
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            # mujoco_objects=[self.cube, origin_ball, z_axis_line, table_center_ball],
            mujoco_objects=[self.cube],
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

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

        # For compatibility with scripts that expect target_pos
        @sensor(modality="object")
        def target_pos(obs_cache):
            return self._cube_xpos

        sensors = [cube_pos, cube_to_eef_pos, target_pos]
        names = [f"cube_pos", f"cube_to_{pf}eef_pos", f"target_pos"]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def reset(self):
        reset_joint_pos = False
        if hasattr(self.robots[0], "_joint_positions"):
            joint_pos = self.robots[0]._joint_positions
            reset_joint_pos = True
        
        super().reset()
        
        if reset_joint_pos:
            self.robots[0].set_robot_joint_positions(joint_pos)
        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )
        return observations

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset cube position using placement initializer
        if not self.deterministic_reset:
            # Sample from the placement initializer for the cube
            object_placements = self.placement_initializer.sample()

            # Get the sampled position and orientation for the cube
            cube_pos, cube_quat, _ = object_placements[self.cube.name]

            # Set the cube's position and orientation
            self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([cube_pos, cube_quat]))

    def _setup_references(self):
        """
        Sets up references to important components.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id("cube_main")

    def reward(self, action):
        """
        Reward function for the task.
        """
        reward = 0
        # sparse completion reward
        if self._check_success():
            reward = 1

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            target_pos = self._cube_xpos
            dist = np.linalg.norm(gripper_site_pos - target_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            # reaching_reward = -((2 * dist) ** 2)
            reward += reaching_reward

        reward *= self.reward_scale

        return reward

    def _pre_action(self, action, policy_step=False):
        """
        Do any preprocessing before taking an action.
        """
        super()._pre_action(action, policy_step)

        if policy_step and not self.static_cube:
            # Update cube velocity to a new random velocity every 100 steps
            if self.timestep % 100 == 0:
                self.cube_velocity = np.random.uniform(-0.03, 0.03, size=3)
                self.cube_velocity[2] = 0  # Set z-velocity to 0 to keep the cube on the table

            # Move the cube
            cube_pos = self.sim.data.get_joint_qpos(self.cube.joints[0])[:3]
            new_pos = cube_pos + self.cube_velocity / self.control_freq
            
            # Ensure the cube stays on the table
            new_pos[0] = np.clip(new_pos[0], 
                                self.table_offset[0] - self.table_full_size[0]/2 + self.cube.size[0]/2, 
                                self.table_offset[0] + self.table_full_size[0]/2 - self.cube.size[0]/2)
            new_pos[1] = np.clip(new_pos[1], 
                                self.table_offset[1] - self.table_full_size[1]/2 + self.cube.size[1]/2, 
                                self.table_offset[1] + self.table_full_size[1]/2 - self.cube.size[1]/2)
            new_pos[2] = self.table_offset[2] + self.cube.size[2]  # Keep z-coordinate constant

            # Update cube position
            self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([new_pos, [1, 0, 0, 0]]))

    def _check_success(self):
        """
        Check if the task has been completed successfully.
        """
        cube_to_eef_pos = self._cube_xpos - self._eef_xpos
        return np.linalg.norm(cube_to_eef_pos) < 0.05
    
    @property
    def _cube_xpos(self):
        """
        Grabs the position of the cube.

        Returns:
            np.array: Cube (x,y,z)
        """
        return self.sim.data.body_xpos[self.cube_body_id]
    