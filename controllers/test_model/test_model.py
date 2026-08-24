from stable_baselines3 import PPO
from spot_env import SpotEnv

env = SpotEnv()

model = PPO.load(
    "ppo_spot_test"
)

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