import numpy as np
import gymnasium as gym

from gymnasium import spaces
from controller import Supervisor
from motors import SpotMotors
from sensors import SpotSensors

class SpotEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.robot = Supervisor()   #instancia o robo supervisor

        self.time_step = int(self.robot.getBasicTimeStep()) #timestep

        self.spot_motors = SpotMotors(self.robot, self.time_step)
        #instancia a classe de motors (inicia-los, move-los, etc...)

        self.spot_sensors = SpotSensors(self.robot, self.time_step)
        #instancia a classe de sensores (iniciar, ler, yaw, roll, pitch, etc...)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        position_low = self.spot_motors.joint_min
        position_high = self.spot_motors.joint_max

        velocity_low = np.full(12, -np.inf, dtype=np.float32)
        velocity_high = np.full(12, np.inf, dtype=np.float32)

        orientation_low = np.full(6, -1.0, dtype=np.float32)
        orientation_high = np.full(6, 1.0, dtype=np.float32)

        gyro_low = np.full(3, -np.inf, dtype=np.float32)
        gyro_high = np.full(3, np.inf, dtype=np.float32)

        linear_velocity_low = np.full(3, -np.inf, dtype=np.float32)
        linear_velocity_high = np.full(3, np.inf, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.concatenate([
                position_low,
                velocity_low,
                orientation_low,
                gyro_low,
                linear_velocity_low
            ]),
            high=np.concatenate([
                position_high,
                velocity_high,
                orientation_high,
                gyro_high,
                linear_velocity_high
            ]),
            dtype=np.float32
        )

        self.steps = 0
        self.max_steps = 1000

        self.upright_threshold = 0.9
        self.fall_threshold = 0.78

        self.fall_steps = 0
        self.fall_steps_limit = 3

    def _action_to_positions(self, action):
        """
        Converte uma ação dada em números em um array para os radianos permitidos pela junta
        """

        action = np.asarray(
            action,
            dtype=np.float32
        )

        positions = (
            self.spot_motors.joint_min
            + (action + 1.0) / 2.0
            * (
                self.spot_motors.joint_max
                - self.spot_motors.joint_min
            )
        )

        return positions

    def reset(self, seed=None, options=None):
        """
        Inicia um novo episódio e restaura dados para a simulação
        """

        super().reset(seed=seed)

        self.robot.simulationReset()

        self.robot.step(self.time_step)

        self.steps = 0
        self.fall_steps = 0

        # Pega o próprio Spot no mundo
        spot_node = self.robot.getSelf()

        # Pega o campo "rotation"
        rotation_field = spot_node.getField("rotation")

        # Escolhe um ângulo aleatório entre -180° e +180°
        random_yaw = self.np_random.uniform(
            -np.pi,
            np.pi
        )

        # Rotação ao redor do eixo Z
        rotation_field.setSFRotation([
            0.0,
            0.0,
            1.0,
            float(random_yaw)
        ])

        # Remove qualquer velocidade/inércia anterior
        spot_node.resetPhysics()

        self.robot.step(self.time_step)

        self.spot_motors.previous_positions = None
        self.spot_sensors.previous_position = None

        gps_position = self.spot_sensors.get_position()
        self.target_height = float(gps_position[2])

        positions = self.spot_motors.get_motor_positions()

        velocities = self.spot_motors.get_joint_velocities(positions)

        orientation_features = (self.spot_sensors.get_orientation_features())

        angular_velocity = (self.spot_sensors.get_angular_velocity())

        linear_velocity = (self.spot_sensors.get_linear_velocity(gps_position))

        observation = np.concatenate([
            positions,
            velocities,
            orientation_features,
            angular_velocity,
            linear_velocity
        ])

        return observation, {}

    def step(self, action):
        """
        Roda um passo da simulação, com ação, físicas
        """

        self.steps += 1

        target_positions = self._action_to_positions(action)

        self.spot_motors.set_motor_positions(target_positions)

        status = self.robot.step(self.time_step)

        orientation_features = (self.spot_sensors.get_orientation_features())

        upright = self.spot_sensors.get_upright(orientation_features)

        positions = (self.spot_motors.get_motor_positions())

        velocities = (self.spot_motors.get_joint_velocities(positions))

        angular_velocity = (self.spot_sensors.get_angular_velocity())

        gps_position = self.spot_sensors.get_position()
        body_height = gps_position[2]

        linear_velocity = (self.spot_sensors.get_linear_velocity(gps_position))

        local_velocity = (self.spot_sensors.get_local_linear_velocity(linear_velocity, orientation_features))

        observation = np.concatenate([
            positions,
            velocities,
            orientation_features,
            angular_velocity,
            linear_velocity
        ])

        if upright < self.fall_threshold:
            self.fall_steps += 1
        else:
            self.fall_steps = 0

        fallen = self.fall_steps >= self.fall_steps_limit

        reward = self._calculate_reward(
            upright,
            local_velocity,
            angular_velocity,
            body_height,
            fallen
        )

        terminated = fallen or status == -1
        truncated = self.steps >= self.max_steps

        yaw = np.arctan2(
            orientation_features[4],
            orientation_features[5]
        )

        info = {
            "yaw": yaw,
            "vx": linear_velocity[0],
            "vy": linear_velocity[1],
            "vz": linear_velocity[2],
            "forward": local_velocity[0],
            "lateral": local_velocity[1],
            "upright": upright,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    def _calculate_reward(self, upright, local_velocity, angular_velocity, body_height, fallen):

        stability_factor = ((upright - self.fall_threshold) / (self.upright_threshold - self.fall_threshold))

        stability_factor = np.clip(stability_factor, 0.0, 1.0)

        height_ratio = (body_height / self.target_height)

        height_factor = np.clip(height_ratio, 0.0, 1.0)

        forward_velocity = local_velocity[0]
        vertical_velocity = local_velocity[2]

        if forward_velocity >= 0.0:
            forward_reward = (forward_velocity * stability_factor * height_factor)
        else:
            forward_reward = forward_velocity

        vertical_penalty = (
            0.15
            * abs(vertical_velocity)
        )

        rotation_penalty = (
            0.03
            * np.linalg.norm(
                angular_velocity
            )
        )

        reward = (forward_reward - vertical_penalty - rotation_penalty)

        if fallen:
            reward -= 10.0

        return float(reward)