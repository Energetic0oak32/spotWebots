import os
from pathlib import Path

from stable_baselines3 import PPO
from spot_env import SpotEnv

env = SpotEnv()

CURRENT_DIR = Path(__file__).resolve().parent
os.chdir(CURRENT_DIR)

if os.path.exists("../main/ppo_spot.zip"):
    model = PPO.load("../main/ppo_spot")

    obs, info = env.reset()

    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            obs, info = env.reset()

else:
    print("Model not found.")
