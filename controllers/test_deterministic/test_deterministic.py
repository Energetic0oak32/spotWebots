import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

MAIN_CONTROLLER = (
    PROJECT_ROOT
    / "controllers"
    / "main"
)

sys.path.insert(
    0,
    str(MAIN_CONTROLLER)
)

from spot_env import SpotEnv
from stable_baselines3 import PPO


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_spot.zip"
)

SEED = None


env = SpotEnv()

model = PPO.load(
    str(MODEL_PATH),
    env=env
)

obs, info = env.reset(seed=SEED)

episode = 1
step_count = 0
episode_reward = 0.0

print(
    f"\nTeste determinístico iniciado"
    f"\nModelo: {MODEL_PATH}"
    f"\nSeed: {SEED}\n"
)


while True:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    step_count += 1
    episode_reward += reward

    if step_count % 20 == 0:

        print(
            f"\nEpisode: {episode}"
            f"\nStep: {step_count}"
            f"\nForward: {info['forward']:.3f}"
            f"\nLateral: {info['lateral']:.3f}"
            f"\nUpright: {info['upright']:.3f}"
            f"\nHeight: {info['body_height']:.3f}"
            f"\nReward: {reward:.3f}"
            f"\nEpisode reward: {episode_reward:.3f}"
        )

    if terminated or truncated:

        print(
            f"\n===== EPISÓDIO {episode} TERMINADO ====="
            f"\nSteps: {step_count}"
            f"\nReward total: {episode_reward:.3f}"
            f"\nFallen: {info['fallen']}"
            f"\n========================================\n"
        )

        episode += 1
        step_count = 0
        episode_reward = 0.0

        obs, info = env.reset(seed=SEED)