import os
import queue
import subprocess
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def stack_observations(observations):
    if isinstance(observations[0], dict):
        return {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}
    return np.stack(observations)


class WorkerFailure(RuntimeError):
    """An actionable transport error, including the affected Webots port."""


class _Reply:
    def __init__(self):
        self.done = threading.Event()
        self.value = None
        self.error = None
        self.started = time.monotonic()

    def finish(self, value=None, error=None):
        self.value, self.error = value, error
        self.done.set()


class _Channel:
    """One daemon owns blocking socket I/O; the trainer always has deadlines.

    The same thread handles authentication, send and receive. This also covers
    partial frames, for which Connection.poll() alone cannot bound recv().
    """

    def __init__(self, authkey):
        self.listener = Listener(("127.0.0.1", 0), authkey=authkey)
        self.port = self.listener.address[1]
        self.connection = None
        self.lock = threading.Lock()
        self.closed = False
        self.jobs = queue.Queue()
        self.ready = _Reply()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            connection = self.listener.accept()
            with self.lock:
                if self.closed:
                    connection.close()
                    return
                self.connection = connection
            self.listener.close()
            self.ready.finish(connection.recv())
        except Exception as error:
            self.ready.finish(error=error)
            return
        while True:
            job = self.jobs.get()
            if job is None:
                return
            command, data, reply = job
            try:
                connection.send((command, data))
                reply.finish(connection.recv())
            except Exception as error:
                reply.finish(error=error)
                return
            if command == "close":
                return

    def submit(self, command, data):
        reply = _Reply()
        self.jobs.put((command, data, reply))
        return reply

    def close(self):
        with self.lock:
            self.closed = True
            connection = self.connection
        self.jobs.put(None)
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        try:
            self.listener.close()
        except OSError:
            pass


class WebotsVecEnv(VecEnv):
    def __init__(self, webots_ports, controller_exe=None, worker_script=None,
                 authkey="spot-parallel", simulation_mode=None,
                 startup_timeout=120.0, response_timeout=30.0, close_timeout=3.0,
                 env_path=None, env_class="spot_env:SpotEnv"):
        env_path = Path(env_path or PROJECT_ROOT / "controllers/main").resolve()
        self.webots_ports = list(webots_ports)
        if not self.webots_ports or len(set(self.webots_ports)) != len(self.webots_ports):
            raise ValueError("Informe portas distintas para pelo menos uma instância.")
        if min(startup_timeout, response_timeout, close_timeout) <= 0:
            raise ValueError("Os timeouts devem ser positivos.")
        if simulation_mode not in (None, "realtime", "fast"):
            raise ValueError("Modo de simulação inválido.")
        self.startup_timeout = float(startup_timeout)
        self.response_timeout = float(response_timeout)
        self.close_timeout = float(close_timeout)
        self.channels = []
        self.processes = []
        self.pending = []
        self.waiting = False
        self.closed = False
        self.failed = False
        home = Path(os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots"))
        controller_exe = Path(controller_exe or home / "msys64/mingw64/bin/webots-controller.exe")
        worker_script = Path(worker_script or PROJECT_ROOT / "training/webots_worker.py")
        for path in (controller_exe, worker_script):
            if not path.is_file():
                raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        try:
            for port in self.webots_ports:
                channel = _Channel(authkey.encode("utf-8"))
                self.channels.append(channel)
                command = [str(controller_exe), f"--port={port}", str(worker_script),
                           f"--trainer-port={channel.port}", f"--authkey={authkey}",
                           f"--env-path={env_path}", f"--env-class={env_class}"]
                if simulation_mode is not None:
                    command.append(f"--simulation-mode={simulation_mode}")
                self.processes.append(subprocess.Popen(command, cwd=str(worker_script.parent)))
            spaces = []
            for index, channel in enumerate(self.channels):
                message = self._receive(index, channel.ready, self.startup_timeout)
                if not isinstance(message, tuple) or len(message) != 3 or message[0] != "ready":
                    self._fail(index, "WORKER_PROTOCOL", "Resposta de inicialização inválida.")
                spaces.append(message[1:])
            if any(pair != spaces[0] for pair in spaces[1:]):
                raise ValueError("Os ambientes têm espaços de observação/ação diferentes.")
            super().__init__(len(spaces), *spaces[0])
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _fail(self, index, tag, detail):
        self.failed = True
        raise WorkerFailure(f"{tag}|porta={self.webots_ports[index]}|{detail}")

    def _receive(self, index, reply, timeout=None):
        timeout = self.response_timeout if timeout is None else timeout
        deadline = reply.started + timeout
        while not reply.done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail(index, "WORKER_TIMEOUT", f"Sem resposta por {timeout:g}s; verifique se o Webots está pausado ou travado.")
            reply.done.wait(min(0.05, remaining))
            if reply.done.is_set():
                break
            code = self.processes[index].poll()
            if code is not None:
                self._fail(index, "WORKER_DISCONNECTED", f"Controller encerrou (código {code}); o Webots pode ter sido fechado.")
        if reply.error is not None:
            self._fail(index, "WORKER_DISCONNECTED", f"Conexão encerrada: {reply.error}")
        message = reply.value
        if isinstance(message, tuple) and len(message) == 2 and isinstance(message[0], str) and message[0] == "error":
            self._fail(index, "WORKER_ERROR", str(message[1]))
        return message

    def _requests(self, command, values, indices):
        pending = [(i, self.channels[i].submit(command, value)) for i, value in zip(indices, values)]
        return [self._receive(i, reply) for i, reply in pending]

    def reset(self):
        indices = list(range(self.num_envs))
        results = self._requests("reset", list(zip(self._seeds, self._options)), indices)
        observations = []
        for index, (obs, info) in enumerate(results):
            observations.append(obs)
            self.reset_infos[index] = info
        self._reset_seeds()
        self._reset_options()
        return stack_observations(observations)

    def step_async(self, actions):
        if self.waiting:
            raise RuntimeError("Já existe um step pendente.")
        if len(actions) != self.num_envs:
            raise ValueError("Quantidade de ações diferente da quantidade de ambientes.")
        self.pending = [(i, channel.submit("step", action))
                        for i, (channel, action) in enumerate(zip(self.channels, actions))]
        self.waiting = True

    def step_wait(self):
        observations, rewards, dones, infos = [], [], [], []
        try:
            for index, reply in self.pending:
                obs, reward, terminated, truncated, info = self._receive(index, reply)
                done = bool(terminated or truncated)
                info = dict(info)
                info["TimeLimit.truncated"] = bool(truncated and not terminated)
                if done:
                    info["terminal_observation"] = obs
                    obs, self.reset_infos[index] = self._requests("reset", [(None, None)], [index])[0]
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            return (stack_observations(observations), np.asarray(rewards, dtype=np.float32),
                    np.asarray(dones, dtype=bool), infos)
        finally:
            self.waiting = False
            self.pending = []

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Queue close behind any outstanding step. Never call step_wait here:
        # closing must not reset worlds or wait indefinitely for a worker.
        replies = [channel.submit("close", None) for channel in self.channels]
        deadline = time.monotonic() + self.close_timeout
        for reply in replies:
            reply.done.wait(max(0, deadline - time.monotonic()))
        for channel in self.channels:
            channel.close()
        # Shared deadlines keep shutdown bounded even with many workers.
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        process.terminate()
                    except OSError:
                        pass
                except OSError:
                    pass
        deadline = time.monotonic() + self.close_timeout
        for process in self.processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
            except OSError:
                pass
        deadline = time.monotonic() + self.close_timeout
        for process in self.processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except (subprocess.TimeoutExpired, OSError):
                pass

    def get_attr(self, attr_name, indices=None):
        indices = list(self._get_indices(indices))
        return self._requests("get_attr", [attr_name] * len(indices), indices)

    def set_attr(self, attr_name, value, indices=None):
        indices = list(self._get_indices(indices))
        self._requests("set_attr", [(attr_name, value)] * len(indices), indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        indices = list(self._get_indices(indices))
        return self._requests("env_method", [(method_name, method_args, method_kwargs)] * len(indices), indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        return [None for _ in self.channels]
