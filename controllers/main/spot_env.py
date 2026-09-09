import numpy as np
from gymnasium import spaces

from environment_template import EnvironmentTemplate
from motors import SpotMotors
from sensors import SpotSensors
from reward import calculate_reward


class SpotEnv(EnvironmentTemplate):
    """Spot environment implemented through EnvironmentTemplate hooks."""

    def configure_devices_and_spaces(self):
        """Configure Spot devices, task constants and Gymnasium spaces."""
        # EnvironmentTemplate creates the Supervisor before calling this hook.
        self.robot = self.webots_supervisor

        # Keep the original Spot setup: Webots physics at the world's basic step,
        # while the RL agent acts every 32 ms.
        self.physics_step = int(self.robot.getBasicTimeStep())
        self.control_step = 32

        if self.control_step % self.physics_step != 0:
            raise ValueError(
                f"control_step ({self.control_step} ms) must be a multiple of "
                f"Webots basicTimeStep ({self.physics_step} ms)."
            )

        # The template advances Webots using self.time_step.  For this robot,
        # one Gymnasium step corresponds to one 32 ms control step.
        self.time_step = self.control_step

        self.min_height_ratio = 0.35
        self.upright_threshold = 0.9
        self.fall_threshold = 0.78
        self.fall_steps = 0
        self.fall_steps_limit = 3
        self.max_steps = 1000

        self.spot_motors = SpotMotors(self.robot, self.control_step)
        self.spot_sensors = SpotSensors(self.robot, self.control_step)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(48,),
            dtype=np.float32,
        )

        self.last_action = np.zeros(12, dtype=np.float32)
        self._previous_action_for_reward = np.zeros(12, dtype=np.float32)
        self.previous_vertical_velocity = 0.0

        self.spawn_height = None
        self.target_height = None
        self._capture_spawn_height = False

        # Latest raw state. observe() refreshes these values and
        # evaluate_transition() consumes them.
        self._positions = np.zeros(12, dtype=np.float32)
        self._joint_velocities = np.zeros(12, dtype=np.float32)
        self._orientation_features = np.zeros(6, dtype=np.float32)
        self._angular_velocity = np.zeros(3, dtype=np.float32)
        self._gps_position = np.zeros(3, dtype=np.float32)
        self._linear_velocity = np.zeros(3, dtype=np.float32)
        self._local_velocity = np.zeros(3, dtype=np.float32)
        self._upright = 1.0
        self._body_height = 0.0

    def reset_robot(self, options):
        """Reset the Spot pose, physics and all temporal histories."""
        self.robot.simulationReset()

        # Preserve the first synchronization step from the original reset().
        # EnvironmentTemplate performs the second step after this hook returns.
        if self.robot.step(self.time_step) == -1:
            raise RuntimeError("Webots encerrou durante reset do Spot.")

        self.fall_steps = 0

        spot_node = self.robot.getSelf()
        rotation_field = spot_node.getField("rotation")

        random_yaw = self.np_random.uniform(-np.pi, np.pi)
        rotation_field.setSFRotation([
            0.0,
            0.0,
            1.0,
            float(random_yaw),
        ])

        spot_node.resetPhysics()

        # Clear finite-difference histories.  The first observe() after the
        # template's synchronization step will therefore report zero velocity,
        # exactly as intended at episode start.
        self.spot_motors.previous_positions = None
        self.spot_sensors.previous_position = None

        self.previous_vertical_velocity = 0.0
        self.last_action = np.zeros(12, dtype=np.float32)
        self._previous_action_for_reward = np.zeros(12, dtype=np.float32)

        # Spawn/target height is captured on the first observation, after the
        # template has advanced Webots once with the new pose.
        self.spawn_height = None
        self.target_height = None
        self._capture_spawn_height = True

    def observe(self):
        """Read Spot state and return the normalized 48-value observation."""
        positions = self.spot_motors.get_motor_positions()
        joint_velocities = self.spot_motors.get_joint_velocities(positions)

        orientation_features = self.spot_sensors.get_orientation_features()
        angular_velocity = self.spot_sensors.get_angular_velocity()
        upright = self.spot_sensors.get_upright(orientation_features)

        gps_position = self.spot_sensors.get_position()
        body_height = float(gps_position[2])

        linear_velocity = self.spot_sensors.get_linear_velocity(gps_position)
        local_velocity = self.spot_sensors.get_local_linear_velocity(
            linear_velocity,
            orientation_features,
        )

        if self._capture_spawn_height:
            self.spawn_height = body_height
            self.target_height = self.spawn_height * 0.85
            self._capture_spawn_height = False

        # Cache raw values for reward/termination calculation.
        self._positions = positions
        self._joint_velocities = joint_velocities
        self._orientation_features = orientation_features
        self._angular_velocity = angular_velocity
        self._gps_position = gps_position
        self._linear_velocity = linear_velocity
        self._local_velocity = local_velocity
        self._upright = float(upright)
        self._body_height = body_height

        observation = self._build_observation(
            positions,
            joint_velocities,
            orientation_features,
            angular_velocity,
            local_velocity,
        )

        if not np.all(np.isfinite(observation)):
            raise RuntimeError("Spot produziu observação com NaN ou infinito.")

        return observation

    def apply_action(self, action):
        """Convert normalized actions to joint positions and command the motors."""
        action = np.asarray(action, dtype=np.float32)

        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Ação com shape {action.shape}; esperado {self.action_space.shape}."
            )

        # The observation after this action should contain the current action,
        # while the reward still needs access to the previous one.
        self._previous_action_for_reward = self.last_action.copy()
        self.last_action = action.copy()

        target_positions = self._action_to_positions(action)
        self.spot_motors.set_motor_positions(target_positions)

    def evaluate_transition(self, observation, action):
        """Calculate locomotion reward, fall termination and debug metrics."""
        if self.target_height is None:
            raise RuntimeError("target_height ainda não foi inicializado.")

        min_body_height = self.target_height * self.min_height_ratio
        bad_orientation = self._upright < self.fall_threshold
        too_low = self._body_height < min_body_height

        if bad_orientation or too_low:
            self.fall_steps += 1
        else:
            self.fall_steps = 0

        fallen = self.fall_steps >= self.fall_steps_limit

        current_action = np.asarray(action, dtype=np.float32)

        reward, reward_info = calculate_reward(
            local_velocity=self._local_velocity,
            angular_velocity=self._angular_velocity,
            upright=self._upright,
            body_height=self._body_height,
            target_height=self.target_height,
            fall_threshold=self.fall_threshold,
            upright_threshold=self.upright_threshold,
            fallen=fallen,
            current_action=current_action,
            previous_action=self._previous_action_for_reward,
            previous_vertical_velocity=self.previous_vertical_velocity,
        )

        self.previous_vertical_velocity = float(self._local_velocity[2])

        yaw = np.arctan2(
            self._orientation_features[4],
            self._orientation_features[5],
        )

        info = {
            "yaw": float(yaw),
            "vx": float(self._linear_velocity[0]),
            "vy": float(self._linear_velocity[1]),
            "vz": float(self._linear_velocity[2]),
            "forward": float(self._local_velocity[0]),
            "lateral": float(self._local_velocity[1]),
            "upright": float(self._upright),
            "fallen": bool(fallen),
            "body_height": float(self._body_height),
            "target_height": float(self.target_height),
            **reward_info,
        }

        terminated = fallen
        return reward, terminated, info

    def close(self):
        """Release local histories without shutting down the Webots process."""
        self.spot_motors.previous_positions = None
        self.spot_sensors.previous_position = None

    def _action_to_positions(self, action):
        """Map an action in [-1, 1] to the physical joint limits."""
        action = np.asarray(action, dtype=np.float32)

        positions = (
            self.spot_motors.joint_min
            + (action + 1.0) / 2.0
            * (self.spot_motors.joint_max - self.spot_motors.joint_min)
        )

        return positions

    def _normalize_joint_positions(self, positions):
        normalized = (
            2.0
            * (positions - self.spot_motors.joint_min)
            / (self.spot_motors.joint_max - self.spot_motors.joint_min)
            - 1.0
        )

        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _normalize_joint_velocities(self, velocities):
        normalized = velocities / self.spot_motors.joint_max_velocity
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _normalize_angular_velocity(self, angular_velocity):
        scale = 10.0
        normalized = angular_velocity / scale
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _normalize_local_velocity(self, local_velocity):
        scale = np.array([4.0, 4.0, 2.0], dtype=np.float32)
        normalized = local_velocity / scale
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _build_observation(
        self,
        positions,
        velocities,
        orientation_features,
        angular_velocity,
        local_velocity,
    ):
        normalized_positions = self._normalize_joint_positions(positions)
        normalized_velocities = self._normalize_joint_velocities(velocities)
        normalized_angular_velocity = self._normalize_angular_velocity(
            angular_velocity
        )
        normalized_local_velocity = self._normalize_local_velocity(local_velocity)

        observation = np.concatenate([
            normalized_positions,
            normalized_velocities,
            orientation_features,
            normalized_angular_velocity,
            normalized_local_velocity,
            self.last_action,
        ])

        return observation.astype(np.float32)
