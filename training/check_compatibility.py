import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from webots_vec_env import WebotsVecEnv
from rl_interface.algorithms import ALGORITHMS, load_model, model_path, validate_spaces
from rl_interface.cli import add_environment_args, environment_kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ports', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--algorithm', choices=list(ALGORITHMS), default='ppo')
    add_environment_args(parser)
    args = parser.parse_args()
    env = None
    try:
        ports = [int(p.strip()) for p in args.ports.split(',') if p.strip()]
        env = WebotsVecEnv(ports, **environment_kwargs(args))
        validate_spaces(args.algorithm, env.observation_space, env.action_space)
        print(f'ENV_OBS|{env.observation_space}\nENV_ACTION|{env.action_space}')
        if model_path(args.model_path).is_file():
            load_model(args.algorithm, args.model_path, env=env, device='cpu')
            print('COMPATIBLE|Algoritmo, modelo e espaços do ambiente são compatíveis.')
        else:
            print('NEW|Algoritmo aceita os espaços do ambiente; um modelo será criado.')
        return 0
    except Exception as error:
        print(f'INCOMPATIBLE|{error}', flush=True)
        return 2
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    sys.exit(main())
