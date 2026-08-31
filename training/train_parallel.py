import argparse
import os

from pathlib import Path

from stable_baselines3 import PPO

from webots_vec_env import WebotsVecEnv

from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_spot"
)

TENSORBOARD_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tensorboard"
)

STOP_FILE = (
    PROJECT_ROOT
    / "training"
    / ".stop_training"
)

class StopTrainingCallback(BaseCallback):

    def __init__(self, stop_file):
        super().__init__()

        self.stop_file = Path(stop_file)

    def _on_step(self):

        if self.stop_file.exists():

            print(
                "\nPedido de parada recebido. "
                "Encerrando treinamento..."
            )

            return False

        return True


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ports",
        type=str,
        required=True
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH)
    )

    return parser.parse_args()


def main():

    args = parse_args()

    webots_ports = [
        int(port.strip())
        for port in args.ports.split(",")
        if port.strip()
    ]

    print(
        f"Portas Webots: {webots_ports}"
    )

    print(
        f"Timesteps: {args.timesteps}"
    )

    print(
        f"Learning rate: {args.learning_rate}"
    )

    print(
        f"n_steps: {args.n_steps}"
    )

    env = WebotsVecEnv(
        webots_ports=webots_ports
    )

    print(
        f"\nAmbiente paralelo conectado: "
        f"{env.num_envs} instâncias.\n"
    )

    try:

        if os.path.exists(
            args.model_path + ".zip"
        ):

            print(
                "Modelo encontrado. "
                "Continuando treinamento..."
            )

            model = PPO.load(
                args.model_path,
                env=env,
                custom_objects={
                    "learning_rate":
                        args.learning_rate,

                    "n_steps":
                        args.n_steps
                }
            )

        else:

            print(
                "Nenhum modelo encontrado. "
                "Criando novo PPO..."
            )

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                verbose=1,
                tensorboard_log=str(
                    TENSORBOARD_PATH
                )
            )

        stop_callback = StopTrainingCallback(
            STOP_FILE
        )

        model.learn(
            args.timesteps,
            reset_num_timesteps=False,
            callback=stop_callback
        )

        model.save(
            args.model_path
        )

        print(
            "Modelo salvo!"
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()