import os
import subprocess
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


class WebotsVecEnv(VecEnv):
    """
    VecEnv do Stable-Baselines3 em que cada ambiente roda em:
        1 processo webots-controller
            -> 1 instância separada do Webots
            -> 1 SpotEnv

    As instâncias do Webots precisam estar abertas antes de criar este VecEnv.
    """

    def __init__(
        self,
        webots_ports,
        controller_exe=None,
        worker_script=None,
        authkey="spot-parallel",
    ):
        self.webots_ports = list(webots_ports)
        self.authkey = authkey.encode("utf-8")

        if controller_exe is None:
            webots_home = Path(
                os.environ.get(
                    "WEBOTS_HOME",
                    r"C:\Program Files\Webots",
                )
            )

            controller_exe = (
                webots_home
                / "msys64"
                / "mingw64"
                / "bin"
                / "webots-controller.exe"
            )

        self.controller_exe = Path(controller_exe)

        if not self.controller_exe.exists():
            raise FileNotFoundError(
                "webots-controller.exe não encontrado em:\n"
                f"{self.controller_exe}\n\n"
                "Defina WEBOTS_HOME ou passe controller_exe explicitamente."
            )

        if worker_script is None:
            worker_script = (
                PROJECT_ROOT
                / "controllers"
                / "main"
                / "parallel_worker.py"
            )

        self.worker_script = Path(worker_script)

        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"parallel_worker.py não encontrado: {self.worker_script}"
            )

        self.listeners = []
        self.connections = []
        self.processes = []
        self.waiting = False
        self.closed = False

        # Abre primeiro os sockets do trainer para evitar corrida de conexão.
        trainer_ports = []

        for _ in self.webots_ports:
            listener = Listener(
                ("127.0.0.1", 0),
                authkey=self.authkey,
            )

            self.listeners.append(listener)
            trainer_ports.append(listener.address[1])

        # Agora lança um controller independente para cada Webots.
        for webots_port, trainer_port in zip(
            self.webots_ports,
            trainer_ports,
        ):
            command = [
                str(self.controller_exe),
                f"--port={webots_port}",
                str(self.worker_script),
                f"--trainer-port={trainer_port}",
                f"--authkey={authkey}",
            ]

            process = subprocess.Popen(
                command,
                cwd=str(self.worker_script.parent),
            )

            self.processes.append(process)

        # Cada controller se conecta ao seu Listener.
        for listener in self.listeners:
            connection = listener.accept()
            self.connections.append(connection)
            listener.close()

        observation_space = None
        action_space = None

        # Confirma que todos criaram SpotEnv corretamente.
        for index, connection in enumerate(self.connections):
            message = connection.recv()

            if (
                isinstance(message, tuple)
                and len(message) == 2
                and message[0] == "error"
            ):
                self.close()
                raise RuntimeError(
                    f"Worker {index} falhou:\n{message[1]}"
                )

            tag, worker_observation_space, worker_action_space = message

            if tag != "ready":
                self.close()
                raise RuntimeError(
                    f"Resposta inesperada do worker {index}: {message}"
                )

            if observation_space is None:
                observation_space = worker_observation_space
                action_space = worker_action_space
            else:
                if worker_observation_space != observation_space:
                    self.close()
                    raise RuntimeError(
                        "Os workers possuem observation spaces diferentes."
                    )

                if worker_action_space != action_space:
                    self.close()
                    raise RuntimeError(
                        "Os workers possuem action spaces diferentes."
                    )

        super().__init__(
            len(self.connections),
            observation_space,
            action_space,
        )

    def reset(self):
        for index, connection in enumerate(self.connections):
            seed = self._seeds[index]
            options = self._options[index]

            connection.send(
                (
                    "reset",
                    (seed, options),
                )
            )

        observations = []

        for index, connection in enumerate(self.connections):
            obs, info = connection.recv()
            observations.append(obs)
            self.reset_infos[index] = info

        self._reset_seeds()
        self._reset_options()

        return np.stack(observations)

    def step_async(self, actions):
        for connection, action in zip(
            self.connections,
            actions,
        ):
            connection.send(
                (
                    "step",
                    np.asarray(action, dtype=np.float32),
                )
            )

        self.waiting = True

    def step_wait(self):
        observations = []
        rewards = []
        dones = []
        infos = []

        for index, connection in enumerate(self.connections):
            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = connection.recv()

            done = bool(terminated or truncated)

            info = dict(info)
            info["TimeLimit.truncated"] = bool(
                truncated and not terminated
            )

            if done:
                info["terminal_observation"] = obs

                connection.send(
                    (
                        "reset",
                        (self._seeds[index], self._options[index]),
                    )
                )

                obs, reset_info = connection.recv()
                self.reset_infos[index] = reset_info

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        self.waiting = False

        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def close(self):
        if self.closed:
            return

        if self.waiting:
            try:
                self.step_wait()
            except Exception:
                pass

        for connection in self.connections:
            try:
                connection.send(("close", None))
            except Exception:
                pass

        for connection in self.connections:
            try:
                connection.recv()
            except Exception:
                pass

            try:
                connection.close()
            except Exception:
                pass

        for process in self.processes:
            try:
                process.wait(timeout=3)
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass

        for listener in self.listeners:
            try:
                listener.close()
            except Exception:
                pass

        self.closed = True

    def get_attr(self, attr_name, indices=None):
        target_indices = self._get_indices(indices)

        for index in target_indices:
            self.connections[index].send(
                ("get_attr", attr_name)
            )

        return [
            self.connections[index].recv()
            for index in target_indices
        ]

    def set_attr(self, attr_name, value, indices=None):
        target_indices = self._get_indices(indices)

        for index in target_indices:
            self.connections[index].send(
                (
                    "set_attr",
                    (attr_name, value),
                )
            )

        for index in target_indices:
            self.connections[index].recv()

    def env_method(
        self,
        method_name,
        *method_args,
        indices=None,
        **method_kwargs,
    ):
        target_indices = self._get_indices(indices)

        for index in target_indices:
            self.connections[index].send(
                (
                    "env_method",
                    (
                        method_name,
                        method_args,
                        method_kwargs,
                    ),
                )
            )

        return [
            self.connections[index].recv()
            for index in target_indices
        ]

    def env_is_wrapped(self, wrapper_class, indices=None):
        # Não é necessário para o PPO aqui; o worker já usa Monitor.
        target_indices = self._get_indices(indices)
        return [False for _ in target_indices]

    def get_images(self):
        return [None for _ in self.connections]
