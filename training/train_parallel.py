import argparse
import os
import sys

from pathlib import Path

from stable_baselines3 import PPO

from webots_vec_env import WebotsVecEnv, WorkerFailure

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
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH)
    )

    parser.add_argument(
        "--output-model-path",
        type=str,
        default=None
    )

    return parser.parse_args()


def main():

    args = parse_args()

    input_model_path = Path(
        args.model_path
    )

    if input_model_path.suffix.lower() == ".zip":
        input_model_path = (
            input_model_path.with_suffix("")
        )

    if args.output_model_path:
        output_model_path = Path(
            args.output_model_path
        )

        if output_model_path.suffix.lower() == ".zip":
            output_model_path = (
                output_model_path.with_suffix("")
            )
    else:
        output_model_path = input_model_path

    output_model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"N epochs: {args.n_epochs}"
    )

    print(
        f"Gamma: {args.gamma}"
    )

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Modelo de origem: "
        f"{input_model_path}"
    )

    print(
        f"Modelo de saída: "
        f"{output_model_path}"
    )

    print(
        f"Rollout efetivo: "
        f"{len(webots_ports)} × "
        f"{args.n_steps} = "
        f"{len(webots_ports) * args.n_steps}"
    )

    try:

        model_zip = Path(
            str(input_model_path) + ".zip"
        )

        if model_zip.exists():

            print(
                "Modelo encontrado. "
                "Continuando treinamento..."
            )

            model = PPO.load(
                str(input_model_path),
                env=env,
                custom_objects={
                    "learning_rate":
                        args.learning_rate,

                    "n_steps":
                        args.n_steps,

                    "batch_size":
                        args.batch_size,

                    "n_epochs":
                        args.n_epochs,

                    "gamma":
                        args.gamma,
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

                learning_rate=
                    args.learning_rate,

                n_steps=
                    args.n_steps,

                batch_size=
                    args.batch_size,

                n_epochs=
                    args.n_epochs,

                gamma=
                    args.gamma,

                seed=
                    args.seed,

                verbose=1,

                tensorboard_log=str(
                    TENSORBOARD_PATH
                )
            )

        model.set_random_seed(
            args.seed
        )

        # Modelos carregados podem trazer um tensorboard_log antigo
        # salvo dentro do .zip. Força o diretório atual do projeto.
        model.tensorboard_log = str(
            TENSORBOARD_PATH
        )

        print(
            "\n=== CONFIGURAÇÃO EFETIVA DO MODELO ==="
        )

        print(
            f"n_steps efetivo: {model.n_steps}"
        )

        print(
            f"batch_size efetivo: {model.batch_size}"
        )

        print(
            f"n_epochs efetivo: {model.n_epochs}"
        )

        print(
            f"gamma efetivo: {model.gamma}"
        )

        print(
            f"n_envs efetivo: {model.n_envs}"
        )

        print(
            f"rollout buffer: "
            f"{model.rollout_buffer.buffer_size}"
        )

        print(
            f"learning rate efetivo: "
            f"{model.lr_schedule(1.0)}"
        )

        print(
            "======================================\n"
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
            str(output_model_path)
        )

        print(
            f"Modelo salvo em: "
            f"{output_model_path}.zip"
        )

    finally:

        env.close()


if __name__ == "__main__":
    try:
        main()
    except WorkerFailure as error:
        print(f"ERROR|{error}", flush=True)
        sys.exit(2)
