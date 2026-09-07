from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl_interface.algorithms import load_model, model_path, specification


def inspect_model(path, algorithm):
    path = model_path(path)
    if not path.is_file():
        return {'exists': False, 'path': str(path)}
    try:
        model = load_model(algorithm, path, device='cpu')
        result = {'exists': True, 'valid': True, 'algorithm': algorithm,
                  'observation_space': str(model.observation_space), 'action_space': str(model.action_space)}
        for key in specification(algorithm)['parameters']:
            value = getattr(model, key, None)
            if key == 'train_freq' and hasattr(value, 'frequency'):
                value = value.frequency
            if isinstance(value, (int, float)):
                result[key] = value
        return result
    except Exception as error:
        return {'exists': True, 'valid': False, 'error': str(error)}
