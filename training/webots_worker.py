import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import traceback
from multiprocessing.connection import Client

import numpy as np
from stable_baselines3.common.monitor import Monitor

from rl_interface.environments import load_environment, supervisor_for
from rl_interface.cli import add_environment_args


def get_env_attr(env, name):
    try:
        return env.get_wrapper_attr(name)
    except Exception:
        return getattr(env.unwrapped, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-port", type=int, required=True)
    parser.add_argument("--authkey", type=str, required=True)
    parser.add_argument("--simulation-mode", choices=("realtime", "fast"), default=None)
    add_environment_args(parser)
    args = parser.parse_args()

    connection = Client(
        ("127.0.0.1", args.trainer_port),
        authkey=args.authkey.encode("utf-8"),
    )

    env = None
    previous_mode = None

    def restore_mode():
        nonlocal previous_mode
        if previous_mode is not None:
            robot.simulationSetMode(previous_mode)
            robot.step(0)
            previous_mode = None

    try:
        env = Monitor(load_environment(args.env_path, args.env_class))
        robot = None
        requested_mode = None
        if args.simulation_mode is not None:
            robot = supervisor_for(env)
            previous_mode = robot.simulationGetMode()
            requested_mode = (
                robot.SIMULATION_MODE_REAL_TIME
                if args.simulation_mode == "realtime"
                else robot.SIMULATION_MODE_FAST
            )
            robot.simulationSetMode(requested_mode)
            robot.step(0)


        connection.send(
            (
                "ready",
                env.observation_space,
                env.action_space,
            )
        )

        while True:
            command, data = connection.recv()

            if command == "reset":
                seed, options = data

                if options is None:
                    obs, info = env.reset(seed=seed)
                else:
                    obs, info = env.reset(seed=seed, options=options)

                # Reapply after reset in case the world reset changes its mode.
                if requested_mode is not None:
                    robot.simulationSetMode(requested_mode)
                    robot.step(0)
                connection.send((obs, info))

            elif command == "step":
                from gymnasium import spaces
                if isinstance(env.action_space, spaces.Discrete):
                    action = int(np.asarray(data).item())
                else:
                    action = np.asarray(data, dtype=env.action_space.dtype)
                connection.send(env.step(action))

            elif command == "get_attr":
                connection.send(get_env_attr(env, data))

            elif command == "set_attr":
                name, value = data
                setattr(env.unwrapped, name, value)
                connection.send(None)

            elif command == "env_method":
                method_name, method_args, method_kwargs = data
                method = get_env_attr(env, method_name)
                connection.send(method(*method_args, **method_kwargs))

            elif command == "close":
                restore_mode()
                env.close()
                env = None
                connection.send(None)
                break

            else:
                raise RuntimeError(f"Comando desconhecido: {command}")

    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except Exception:
            pass
        raise

    finally:
        try:
            if env is not None:
                try:
                    restore_mode()
                finally:
                    env.close()
        finally:
            connection.close()


if __name__ == "__main__":
    main()
