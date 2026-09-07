import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from stable_baselines3.common.callbacks import BaseCallback
from webots_vec_env import WebotsVecEnv, WorkerFailure
from rl_interface.algorithms import (ALGORITHMS, algorithm_class, load_model, model_path,
                                      policy_for, validate_parameters, validate_spaces)
from rl_interface.cli import add_environment_args, environment_kwargs

STOP_FILE = PROJECT_ROOT / 'training/.stop_training'


class StopTrainingCallback(BaseCallback):
    def _on_step(self):
        if STOP_FILE.exists():
            print('Pedido de parada recebido. Salvando ao encerrar o treinamento...', flush=True)
            return False
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ports', required=True)
    parser.add_argument('--algorithm', choices=list(ALGORITHMS), default='ppo')
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-path', default=str(PROJECT_ROOT / 'models/ppo_spot'))
    parser.add_argument('--output-model-path')
    parser.add_argument('--parameters', default='{}', help='JSON dos parâmetros do algoritmo')
    # Backward compatibility for existing PPO commands.
    for name, kind in [('learning-rate', float), ('n-steps', int), ('batch-size', int), ('n-epochs', int), ('gamma', float)]:
        parser.add_argument('--' + name, type=kind, default=None)
    add_environment_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    ports = [int(p.strip()) for p in args.ports.split(',') if p.strip()]
    if args.timesteps < 1:
        raise ValueError('Timesteps deve ser positivo.')
    values = json.loads(args.parameters)
    if not isinstance(values, dict):
        raise ValueError('--parameters precisa ser um objeto JSON.')
    for key in ('learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma'):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    parameters = validate_parameters(args.algorithm, values, len(ports))
    source = model_path(args.model_path)
    output = model_path(args.output_model_path or args.model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        env = WebotsVecEnv(ports, **environment_kwargs(args))
        validate_spaces(args.algorithm, env.observation_space, env.action_space)
        tensorboard = str(PROJECT_ROOT / 'logs/tensorboard')
        if source.is_file():
            model = load_model(args.algorithm, source, env=env, device='cpu',
                               custom_objects={**parameters, 'seed': args.seed})
            if args.algorithm in ('dqn', 'sac'):
                print('INFO|Retomando pesos e otimizador; o replay buffer inicia vazio.')
        else:
            model = algorithm_class(args.algorithm)(policy_for(env.observation_space), env,
                    seed=args.seed, verbose=1, device='cpu', tensorboard_log=tensorboard, **parameters)
        model.tensorboard_log = tensorboard
        model.set_random_seed(args.seed)
        model.rl_interface_algorithm = args.algorithm
        model.rl_interface_env_class = args.env_class
        print(f'Algoritmo: {args.algorithm.upper()} | Ambiente: {args.env_class}')
        print(f'Portas: {ports} | Seed: {args.seed} | Parâmetros: {parameters}')
        print(f'Modelo de origem: {source}\nModelo de saída: {output}', flush=True)
        model.learn(args.timesteps, reset_num_timesteps=False, callback=StopTrainingCallback())
        model.save(str(output))
        print(f'Modelo salvo em: {output}', flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'ERROR|{error}', flush=True)
        sys.exit(2)
