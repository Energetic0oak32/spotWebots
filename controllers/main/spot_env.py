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

        self.observation_space = spaces.Box(
            low=np.concatenate([
                position_low,
                velocity_low,
                orientation_low,
                gyro_low
            ]),
            high=np.concatenate([
                position_high,
                velocity_high,
                orientation_high,
                gyro_high
            ]),
            dtype=np.float32
        )

        self.steps = 0
        self.max_steps = 1000

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
        Inicia um novo episódio e retorna
        a observação inicial e info.
        """
        super().reset(seed=seed)

        self.steps = 0
        self.robot.step(self.time_step)

        self.spot_motors.previous_positions = None

        positions = self.spot_motors.get_motor_positions()

        velocities = self.spot_motors.get_joint_velocities(
            positions
        )

        orientation_features = (
            self.spot_sensors.get_orientation_features()
        )

        angular_velocity = (
            self.spot_sensors.get_angular_velocity()
        )

        observation = np.concatenate([
            positions,
            velocities,
            orientation_features,
            angular_velocity
        ])

        return observation, {}

    def step(self, action):

        self.steps += 1

        target_positions = self._action_to_positions(
            action
        )

        self.spot_motors.set_motor_positions(
            target_positions
        )

        status = self.robot.step(
            self.time_step
        )

        orientation_features = (
            self.spot_sensors.get_orientation_features()
        )

        positions = (
            self.spot_motors.get_motor_positions()
        )

        velocities = (
            self.spot_motors.get_joint_velocities(
                positions
            )
        )

        angular_velocity = (
            self.spot_sensors.get_angular_velocity()
        )

        observation = np.concatenate([
            positions,
            velocities,
            orientation_features,
            angular_velocity
        ])

        reward = 0.0

        terminated = status == -1
        truncated = self.steps >= self.max_steps

        return (
            observation,
            reward,
            terminated,
            truncated,
            {}
        )