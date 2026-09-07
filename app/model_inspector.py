from pathlib import Path

from stable_baselines3 import (
    PPO,
    DQN,
)


SUPPORTED_ALGORITHMS = {
    "ppo": PPO,
    "dqn": DQN,
}


def normalize_model_path(path):

    path = Path(path)

    if path.suffix.lower() != ".zip":
        path = Path(
            str(path) + ".zip"
        )

    return path


def inspect_model(
    path,
    algorithm,
):

    model_path = normalize_model_path(
        path
    )

    if not model_path.exists():
        return {
            "exists": False,
            "path": str(model_path),
        }

    algorithm = algorithm.lower()

    if algorithm not in SUPPORTED_ALGORITHMS:
        return {
            "exists": True,
            "valid": False,
            "error": (
                f"Algoritmo '{algorithm}' "
                "não suportado."
            ),
        }

    model_class = (
        SUPPORTED_ALGORITHMS[
            algorithm
        ]
    )

    try:

        model = model_class.load(
            str(model_path),
            device="cpu",
        )

    except Exception as error:

        return {
            "exists": True,
            "valid": False,
            "error": str(error),
        }

    result = {
        "exists": True,
        "valid": True,

        "algorithm": algorithm,

        "observation_space":
            str(model.observation_space),

        "action_space":
            str(model.action_space),
    }

    for attribute in [
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
    ]:

        if hasattr(model, attribute):

            result[attribute] = (
                getattr(
                    model,
                    attribute
                )
            )

    return result