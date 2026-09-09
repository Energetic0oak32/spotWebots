"""Abstract starting point. Copy to the robot project and implement every hook.

This is not a runnable environment. It intentionally cannot be instantiated
until robot-specific methods are implemented. Importing it does not open Webots.
"""
from abc import ABC, abstractmethod
import gymnasium as gym


class EnvironmentTemplate(gym.Env, ABC):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        from controller import Supervisor
        self.webots_supervisor = Supervisor()
        self.time_step = int(self.webots_supervisor.getBasicTimeStep())
        self.steps = 0
        self.max_steps = 1000
        self.configure_devices_and_spaces()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.reset_robot(options or {})
        if self.webots_supervisor.step(self.time_step) == -1:
            raise RuntimeError("Webots encerrou durante reset.")
        return self.observe(), {}

    def step(self, action):
        self.apply_action(action)
        if self.webots_supervisor.step(self.time_step) == -1:
            raise RuntimeError("Webots encerrou durante step.")
        self.steps += 1
        observation = self.observe()
        reward, terminated, info = self.evaluate_transition(observation, action)
        truncated = self.steps >= self.max_steps
        return observation, float(reward), bool(terminated), truncated, info

    @abstractmethod
    def configure_devices_and_spaces(self):
        """Enable devices and define observation_space/action_space for ONE robot."""

    @abstractmethod
    def reset_robot(self, options):
        """Restore pose, physics, histories and use self.np_random for randomization."""

    @abstractmethod
    def observe(self):
        """Return a finite observation matching the declared shape and dtype."""

    @abstractmethod
    def apply_action(self, action):
        """Convert the action into valid commands for this robot's actuators."""

    @abstractmethod
    def evaluate_transition(self, observation, action):
        """Return reward, terminated, info for the task."""

    @abstractmethod
    def close(self):
        """Stop actuators/release resources, without quitting the whole simulator."""
