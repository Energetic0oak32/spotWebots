import os
from pathlib import Path

from spot_env import SpotEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


MODEL_PATH = "ppo_spot"

env = SpotEnv()
env = Monitor(env)

total_timesteps = 1_000_000

if os.path.exists(MODEL_PATH + ".zip"):

    print("Modelo encontrado. Continuando treinamento...")
    model = PPO.load(MODEL_PATH, env=env)

else:

    print("Nenhum modelo encontrado. Criando novo PPO...")
    model = PPO("MlpPolicy", env, verbose=1)


model.learn(total_timesteps, reset_num_timesteps=False)

model.save(MODEL_PATH)
print("Modelo salvo!")