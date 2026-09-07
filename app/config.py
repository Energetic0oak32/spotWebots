import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "spot_manager_config.json"


DEFAULT_CONFIG = {

    "available_algorithms": [
        "ppo"
    ],

    # Projeto
    "robot": "spot",
    "algorithm": "ppo",

    # Webots
    "webots_home": r"C:\Program Files\Webots",

    "world_path": str(
        PROJECT_ROOT
        / "worlds"
        / "static_spot_parallel.wbt"
    ),

    "venv_path": r"G:\UFSC\WeBots\spotWebots\env",

    "controller_path": str(
        PROJECT_ROOT
        / "controllers"
        / "main"
    ),

    # Paralelismo
    "instances": 2,
    "base_port": 1234,

    # Modelo
    "model_path": str(
        PROJECT_ROOT
        / "models"
        / "ppo_spot"
    ),

    "output_model_path": str(
        PROJECT_ROOT
        / "models"
        / "ppo_spot_output"
    ),

    # Treinamento geral
    "total_timesteps": 500_000,
    "seed": 42,
    "random_seed": False,
    "test_episodes": 3,

    # Configurações específicas dos algoritmos
    "algorithm_settings": {

        "ppo": {
            "learning_rate": 5e-5,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
        },

        "dqn": {
            "learning_rate": 1e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 50_000,
            "batch_size": 32,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "exploration_fraction": 0.1,
        },
    },
}


def load_config():

    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()

    try:

        with open(
            CONFIG_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            saved_config = json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):
        return DEFAULT_CONFIG.copy()

    config = DEFAULT_CONFIG.copy()
    config.update(saved_config)

    return config


def save_config(config):

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )
