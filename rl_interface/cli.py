from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def add_environment_args(parser):
    parser.add_argument('--env-path', default=str(PROJECT_ROOT / 'controllers/main'))
    parser.add_argument('--env-class', default='spot_env:SpotEnv')


def environment_kwargs(args):
    return {'env_path': args.env_path, 'env_class': args.env_class}
