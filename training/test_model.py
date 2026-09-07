import argparse
import sys
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO

from webots_vec_env import WebotsVecEnv


def normalize_model_path(path):

    path = Path(path)

    if path.suffix.lower() != ".zip":
        path = Path(
            str(path) + ".zip"
        )

    return path


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ports",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    ports = [
        int(port.strip())
        for port in args.ports.split(",")
        if port.strip()
    ]

    if not ports:
        print(
            "ERROR|Nenhuma porta informada."
        )
        return 1

    # Para o teste determinístico,
    # usamos apenas uma instância.
    if len(ports) != 1:
        print(
            "ERROR|O teste determinístico "
            "deve usar exatamente 1 instância."
        )
        return 1

    if args.algorithm.lower() != "ppo":
        print(
            f"ERROR|Algoritmo "
            f"'{args.algorithm}' "
            "ainda não implementado."
        )
        return 1

    model_path = normalize_model_path(
        args.model_path
    )

    if not model_path.exists():
        print(
            f"ERROR|Modelo não encontrado: "
            f"{model_path}"
        )
        return 1

    env = None

    try:

        print(
            "INFO|Conectando ao ambiente..."
        )

        env = WebotsVecEnv(
            webots_ports=ports
        )

        print(
            "INFO|Carregando modelo..."
        )

        model = PPO.load(
            str(model_path),
            env=env,
            device="cpu",
        )

        model.set_random_seed(
            args.seed
        )

        episode_rewards = []
        episode_lengths = []

        print(
            "\n=== TESTE DETERMINÍSTICO ==="
        )

        print(
            f"Modelo: {model_path}"
        )

        print(
            f"Seed base: {args.seed}"
        )

        print(
            f"Episódios: {args.episodes}"
        )

        print(
            "deterministic=True"
        )

        print(
            "============================\n"
        )

        for episode in range(
            args.episodes
        ):

            # Cada episódio recebe uma seed
            # conhecida e reproduzível:
            #
            # 42, 43, 44...
            episode_seed = (
                args.seed
                + episode
            )

            env.seed(
                episode_seed
            )

            obs = env.reset()

            total_reward = 0.0
            step_count = 0

            print(
                f"\nEPISODE_START|"
                f"episode={episode + 1}|"
                f"seed={episode_seed}"
            )

            while True:

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                (
                    obs,
                    rewards,
                    dones,
                    infos,
                ) = env.step(action)

                reward = float(
                    rewards[0]
                )

                done = bool(
                    dones[0]
                )

                info = infos[0]

                total_reward += reward
                step_count += 1

                if step_count % 20 == 0:

                    print(
                        f"STEP|"
                        f"episode={episode + 1}|"
                        f"step={step_count}|"
                        f"reward={reward:.3f}|"
                        f"forward="
                        f"{info.get('forward', 0.0):.3f}|"
                        f"lateral="
                        f"{info.get('lateral', 0.0):.3f}|"
                        f"upright="
                        f"{info.get('upright', 0.0):.3f}|"
                        f"height="
                        f"{info.get('body_height', 0.0):.3f}"
                    )

                if done:

                    fallen = bool(
                        info.get(
                            "fallen",
                            False,
                        )
                    )

                    print(
                        f"EPISODE_END|"
                        f"episode={episode + 1}|"
                        f"seed={episode_seed}|"
                        f"steps={step_count}|"
                        f"reward="
                        f"{total_reward:.3f}|"
                        f"fallen={fallen}"
                    )

                    episode_rewards.append(
                        total_reward
                    )

                    episode_lengths.append(
                        step_count
                    )

                    break

        mean_reward = float(
            np.mean(
                episode_rewards
            )
        )

        mean_length = float(
            np.mean(
                episode_lengths
            )
        )

        print(
            "\n=== RESULTADO ==="
        )

        print(
            f"Reward médio: "
            f"{mean_reward:.3f}"
        )

        print(
            f"Duração média: "
            f"{mean_length:.1f} steps"
        )

        print(
            "Teste concluído."
        )

        return 0

    except Exception as error:

        print(
            f"ERROR|{error}"
        )

        return 1

    finally:

        if env is not None:
            env.close()


if __name__ == "__main__":
    sys.exit(
        main()
    )