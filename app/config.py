import json
import os
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from rl_interface.algorithms import ALGORITHMS, defaults

CONFIG_PATH = PROJECT_ROOT / 'spot_manager_config.json'
DEFAULT_CONFIG = {
    'available_algorithms': list(ALGORITHMS),
    'robot': 'spot', 'algorithm': 'ppo',
    'webots_home': r'C:\Program Files\Webots',
    'world_path': str(PROJECT_ROOT / 'worlds/static_spot_parallel.wbt'),
    'venv_path': r'G:\UFSC\WeBots\spotWebots\env',
    'controller_path': str(PROJECT_ROOT / 'controllers/main'),
    'env_class': 'spot_env:SpotEnv',
    'instances': 2, 'base_port': 1234,
    'model_path': str(PROJECT_ROOT / 'models/ppo_spot'),
    'output_model_path': str(PROJECT_ROOT / 'models/ppo_spot_output'),
    'total_timesteps': 500_000, 'seed': 42, 'random_seed': False,
    'test_episodes': 3,
    'algorithm_settings': {name: defaults(name) for name in ALGORITHMS},
}


def load_config():
    config = deepcopy(DEFAULT_CONFIG)
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        if not isinstance(saved, dict):
            return config
    except (OSError, ValueError):
        return config
    config.update({k: v for k, v in saved.items() if k not in ('algorithm_settings', 'available_algorithms')})
    settings = saved.get('algorithm_settings', {})
    if isinstance(settings, dict):
        for name in ALGORITHMS:
            if isinstance(settings.get(name), dict):
                config['algorithm_settings'][name].update({k: v for k, v in settings[name].items() if k in defaults(name)})
    if config['algorithm'] not in ALGORITHMS:
        config['algorithm'] = 'ppo'
    return config


def save_config(config):
    temporary = CONFIG_PATH.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(config, indent=4, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, CONFIG_PATH)
    finally:
        temporary.unlink(missing_ok=True)
