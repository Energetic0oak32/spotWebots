"""Single registry used by UI, validation, training and model loading.

Entries expose a deliberately small parameter set. Add algorithms here, not in
GUI conditionals. Imports of Gymnasium/SB3 are deferred until needed.
"""
import json
import math
from zipfile import ZipFile
from pathlib import Path


def param(default, minimum, maximum):
    return {"default": default, "min": minimum, "max": maximum,
            "type": "int" if isinstance(default, int) else "float"}


LR = param(5e-5, 1e-8, 1.0)
GAMMA = param(0.99, 0.0, 1.0)
OFF_POLICY = {
    "learning_rate": LR, "gamma": GAMMA,
    "buffer_size": param(100_000, 1, 10_000_000),
    "learning_starts": param(1000, 0, 10_000_000),
    "batch_size": param(256, 1, 1_000_000),
    "train_freq": param(1, 1, 100_000),
    "gradient_steps": param(1, -1, 100_000),
}
ALGORITHMS = {
    "ppo": {"class": "PPO", "actions": ("Box", "Discrete", "MultiDiscrete", "MultiBinary"),
            "parameters": {"learning_rate": LR, "n_steps": param(2048, 1, 65536),
                           "batch_size": param(64, 2, 1_000_000), "n_epochs": param(10, 1, 1000), "gamma": GAMMA}},
    "a2c": {"class": "A2C", "actions": ("Box", "Discrete", "MultiDiscrete", "MultiBinary"),
            "parameters": {"learning_rate": param(7e-4, 1e-8, 1.0), "n_steps": param(5, 1, 65536), "gamma": GAMMA}},
    "sac": {"class": "SAC", "actions": ("Box",), "parameters": dict(OFF_POLICY)},
    "dqn": {"class": "DQN", "actions": ("Discrete",),
            "parameters": {**OFF_POLICY, "batch_size": param(32, 1, 1_000_000),
                           "train_freq": param(4, 1, 100_000), "exploration_fraction": param(0.1, 0.0, 1.0)}},
}


def specification(name):
    try:
        return ALGORITHMS[name.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"Algoritmo não registrado: {name}") from None


def defaults(name):
    return {k: v["default"] for k, v in specification(name)["parameters"].items()}


def validate_parameters(name, values, n_envs):
    if n_envs < 1:
        raise ValueError("Use pelo menos uma instância.")
    spec = specification(name)["parameters"]
    unknown = set(values) - set(spec)
    if unknown:
        raise ValueError(f"Parâmetros não registrados para {name}: {sorted(unknown)}")
    result = defaults(name)
    result.update(values)
    for key, value in result.items():
        rule = spec[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} deve ser um número finito.")
        if rule["type"] == "int" and not isinstance(value, int):
            raise ValueError(f"{key} deve ser inteiro.")
        if not rule["min"] <= value <= rule["max"]:
            raise ValueError(f"{key} deve estar entre {rule['min']} e {rule['max']}.")
    if name == "ppo" and result["n_steps"] * n_envs < 2:
        raise ValueError("PPO precisa de pelo menos duas amostras por rollout.")
    return result


def validate_spaces(name, observation_space, action_space):
    import numpy as np
    from gymnasium import spaces
    allowed = tuple(getattr(spaces, t) for t in specification(name)["actions"])
    if not isinstance(action_space, allowed):
        raise ValueError(f"{name.upper()} não aceita ações {type(action_space).__name__}. "
                         f"Aceita: {', '.join(specification(name)['actions'])}.")
    if isinstance(action_space, spaces.Box) and (not np.isfinite(action_space.low).all() or not np.isfinite(action_space.high).all()):
        raise ValueError("Limites das ações contínuas devem ser finitos.")
    def check(space, label, allow_dict=False):
        if isinstance(space, spaces.Dict) and allow_dict:
            for key, child in space.spaces.items():
                check(child, f"{label}.{key}")
        elif isinstance(space, spaces.Box):
            if len(space.shape) != 1:
                raise ValueError(f"{label}: esta etapa suporta vetores; imagens/CNN ficam para outra etapa.")
        elif isinstance(space, spaces.Discrete):
            if space.start != 0:
                raise ValueError(f"{label}: Discrete deve começar em zero.")
        elif isinstance(space, spaces.MultiDiscrete):
            if len(space.nvec.shape) != 1 or np.any(space.start != 0):
                raise ValueError(f"{label}: MultiDiscrete deve ser unidimensional e começar em zero.")
        elif isinstance(space, spaces.MultiBinary):
            if len(space.shape) != 1:
                raise ValueError(f"{label}: MultiBinary deve ser unidimensional.")
        else:
            raise ValueError(f"{label}: espaço não suportado nesta etapa: {space}.")
    check(action_space, "Ações")
    check(observation_space, "Observações", allow_dict=True)


def policy_for(observation_space):
    from gymnasium import spaces
    return "MultiInputPolicy" if isinstance(observation_space, spaces.Dict) else "MlpPolicy"


def algorithm_class(name):
    import stable_baselines3 as sb3
    return getattr(sb3, specification(name)["class"])


def model_path(path):
    path = Path(path)
    return path if path.suffix.lower() == ".zip" else Path(str(path) + ".zip")


def saved_algorithm(path):
    # Inspect ordinary JSON keys first; no model construction is needed here.
    with ZipFile(model_path(path)) as archive:
        data = json.loads(archive.read("data"))
    marker = data.get("rl_interface_algorithm")
    if isinstance(marker, str):
        return marker
    # Compatibility with legacy archives from these four SB3 algorithms.
    if "clip_range" in data and "n_epochs" in data:
        return "ppo"
    if "exploration_fraction" in data and "exploration_final_eps" in data:
        return "dqn"
    if "target_entropy" in data and "ent_coef" in data:
        return "sac"
    if "n_steps" in data and "gae_lambda" in data and "normalize_advantage" in data:
        return "a2c"
    raise ValueError("Não foi possível identificar o algoritmo desse modelo.")


def load_model(name, path, env=None, **kwargs):
    actual = saved_algorithm(path)
    if actual != name:
        raise ValueError(f"Modelo salvo é {actual.upper()}, mas foi selecionado {name.upper()}.")
    if env is not None:
        validate_spaces(name, env.observation_space, env.action_space)
    return algorithm_class(name).load(str(model_path(path)), env=env, **kwargs)
