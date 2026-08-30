import os
import numpy as np

from spot_env import SpotEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

TEST = True
CONTINUE = True

MODEL_PATH = "ppo_spot"
TOTAL_TIMESTEPS = 2_000_000

env = SpotEnv()
env = Monitor(env)

if os.path.exists(MODEL_PATH + ".zip"):

    print("Modelo encontrado.")
    model = PPO.load(MODEL_PATH, env=env)

elif TEST:

    print("Nenhum modelo encontrado, teste abortado.")
    CONTINUE = False
    TEST = False

else:

    print("Nenhum modelo encontrado. Criando novo PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        verbose=1
    )


if TEST:

    obs, info = env.reset()

    step_count = 0

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1

        if step_count % 20 == 0:

            print(
                f"\n"
                f"Step:     {step_count}\n"
                f"Yaw:      {np.degrees(info['yaw']):8.2f}°\n"
                f"Global:   "
                f"vx={info['vx']:7.3f}  "
                f"vy={info['vy']:7.3f}  "
                f"vz={info['vz']:7.3f}\n"
                f"Local:    "
                f"forward={info['forward']:7.3f}  "
                f"lateral={info['lateral']:7.3f}\n"
                f"Upright:  {info['upright']:7.3f}\n"
                f"Reward:   {reward:7.3f}\n"
                f"Height:   "
                f"{info['body_height']:7.3f} / "
                f"{info['target_height']:7.3f}  "
                f"factor={info['height_factor']:5.2f}\n"
            )

        if terminated or truncated:
            obs, info = env.reset()


elif CONTINUE:

    model.learn(TOTAL_TIMESTEPS, reset_num_timesteps=False)

    model.save(MODEL_PATH)

    print("Modelo salvo!")