from spot_env import SpotEnv


env = SpotEnv()

obs, info = env.reset()

print("Estado inicial:")
print(obs)


for i in range(1000):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    print("Step:", i)

    if terminated or truncated:
        break