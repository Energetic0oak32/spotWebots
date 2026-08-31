import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "spot_manager_config.json"


DEFAULT_CONFIG = {
    "webots_home": r"C:\Program Files\Webots",

    "world_path": str(
        PROJECT_ROOT
        / "worlds"
        / "static_spot_ppo.wbt"
    ),

    "venv_path": str(
        PROJECT_ROOT
        / "env"
    ),

    "controller_path": str(
        PROJECT_ROOT
        / "controllers"
        / "main"
    ),

    "instances": 2,
    "base_port": 1234,

    "total_timesteps": 500_000,
    "learning_rate": 5e-5,
    "n_steps": 2048
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