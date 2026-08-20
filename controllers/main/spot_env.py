import numpy as np
import gymnasium as gym

from gymnasium import spaces
from controller import Supervisor
from motors import SpotMotors

class SpotEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.robot = Supervisor()   #instancia o robo supervisor

        self.time_step = int(self.robot.getBasicTimeStep()) #timestep

        self.spot_motors = SpotMotors(self.robot, self.time_step)
        #instancia a classe de motors (inicia-los, move-los, etc...)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=self.spot_motors.joint_min,
            high=self.spot_motors.joint_max,
            shape=(24,)
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
        reseta a simulação (pode ser usado para encerrar uma simulação inteira, ou encerrar um episódio)
        retorna obs e info.
        """

        super().reset(seed=seed)

        self.steps = 0

        positions = self.spot_motors.get_motor_positions()
        self.spot_motors.previous_positions = None

        return positions, {}

    def step(self, action):
        """
        Avança um passo da simulação retorna
        positions(posição dos motors), reward, terminated, truncated, {} (info)
        """

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

        positions = (
            self.spot_motors.get_motor_positions()
        )

        reward = 0.0

        terminated = status == -1
        truncated = self.steps >= self.max_steps

        return (
            positions,
            reward,
            terminated,
            truncated,
            {}
        )