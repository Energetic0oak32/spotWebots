import importlib
import sys
from pathlib import Path


def validate_entry(directory, entry):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Pasta do ambiente não encontrada: {directory}")
    if entry.count(":") != 1:
        raise ValueError("Informe o ambiente no formato modulo:Classe.")
    module, cls = entry.split(":")
    if not module or not all(p.isidentifier() for p in module.split('.')) or not cls.isidentifier():
        raise ValueError("Módulo ou classe do ambiente inválidos.")
    path = directory.joinpath(*module.split('.'))
    if not path.with_suffix('.py').is_file() and not (path/'__init__.py').is_file():
        raise ValueError(f"Módulo {module} não encontrado na pasta selecionada.")
    return directory, module, cls


def load_environment(directory, entry):
    directory, module_name, class_name = validate_entry(directory, entry)
    sys.path.insert(0, str(directory))
    module = importlib.import_module(module_name)
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(directory):
        raise ValueError(f"O módulo {module_name} foi importado de outra pasta: {origin}")
    factory = getattr(module, class_name)
    env = factory()
    import gymnasium as gym
    if not isinstance(env, gym.Env):
        raise TypeError("O ambiente deve herdar de gymnasium.Env.")
    return env


def supervisor_for(env):
    target = env.unwrapped
    supervisor = getattr(target, 'webots_supervisor', None)
    if supervisor is None:
        supervisor = getattr(target, 'robot', None)  # Legacy SpotEnv bridge.
    if supervisor is None or not all(callable(getattr(supervisor, n, None)) for n in ('simulationGetMode', 'simulationSetMode', 'step')):
        raise ValueError("Para controlar tempo real, exponha o Supervisor em env.webots_supervisor.")
    return supervisor
