import numpy as np
import gymnasium as gym

from gymnasium import spaces
from controller import Supervisor
from motors import SpotMotors


class SpotEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.robot = Supervisor()
        self.spot_motors = SpotMotors(self.robot)

        self.time_step = int(self.robot.getBasicTimeStep())

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

        self.state = np.zeros(
            12,
            dtype=np.float32
        )

        self.steps = 0
        self.max_steps = 1000


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = np.zeros(
            12,
            dtype=np.float32
        )

        self.steps = 0

        return self.state.copy(), {}


    def step(self, action):

        self.steps += 1

        # Teste: move somente o primeiro motor
        self.spot_motors.set_motor_position(
            0,
            0.3
        )

        # Agora deixa o Webots aplicar fisicamente esse comando
        status = self.robot.step(
            self.time_step
        )

        reward = 0.0

        if status == -1:
            terminated = True
        else:
            terminated = False

        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        return (
            self.state.copy(),
            reward,
            terminated,
            truncated,
            {}
        )