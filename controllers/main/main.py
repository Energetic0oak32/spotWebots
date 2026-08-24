from spot_env import SpotEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


env = SpotEnv()

env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=10_000
)

model.save("ppo_spot_test")