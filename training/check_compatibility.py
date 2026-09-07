import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO

from webots_vec_env import WebotsVecEnv


def normalize_model_path(path):
    path = Path(path)

    if path.suffix.lower() == ".zip":
        return path

    return Path(str(path) + ".zip")


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

    return parser.parse_args()


def main():
    args = parse_args()

    ports = [
        int(port.strip())
        for port in args.ports.split(",")
        if port.strip()
    ]

    if not ports:
        print("ERROR|Nenhuma porta informada.")
        return 1

    model_path = normalize_model_path(
        args.model_path
    )

    if not model_path.exists():
        print(
            "NEW|Modelo ainda não existe. "
            "Será criado no primeiro treinamento."
        )
        return 0

    if args.algorithm.lower() != "ppo":
        print(
            f"ERROR|Algoritmo "
            f"'{args.algorithm}' ainda não implementado."
        )
        return 1

    env = None

    try:
        print(
            "INFO|Conectando ao ambiente..."
        )

        env = WebotsVecEnv(
            ports
        )

        print(
            f"ENV_OBS|{env.observation_space}"
        )

        print(
            f"ENV_ACTION|{env.action_space}"
        )

        print(
            "INFO|Carregando modelo..."
        )

        model = PPO.load(
            str(model_path),
            env=env,
            device="cpu",
        )

        print(
            f"MODEL_OBS|"
            f"{model.observation_space}"
        )

        print(
            f"MODEL_ACTION|"
            f"{model.action_space}"
        )

        print(
            "COMPATIBLE|Modelo e ambiente "
            "são compatíveis."
        )

        return 0

    except Exception as error:

        print(
            f"INCOMPATIBLE|{error}"
        )

        return 2

    finally:

        if env is not None:
            env.close()


if __name__ == "__main__":
    sys.exit(main())