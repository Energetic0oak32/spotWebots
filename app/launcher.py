import subprocess
import time
from pathlib import Path


class WebotsLauncher:

    def __init__(self, config):

        self.config = config

        self.processes = []


    def get_ports(self):

        base_port = self.config["base_port"]

        instances = self.config["instances"]

        return [
            base_port + i
            for i in range(instances)
        ]


    def get_webots_executable(self):

        webots_home = Path(
            self.config["webots_home"]
        )

        return (
            webots_home
            / "msys64"
            / "mingw64"
            / "bin"
            / "webots.exe"
        )


    def start_webots_instances(self):

        webots_exe = (
            self.get_webots_executable()
        )

        world_path = Path(
            self.config["world_path"]
        )

        ports = self.get_ports()

        for port in ports:

            command = [
                str(webots_exe),
                f"--port={port}",
                "--mode=fast",
                str(world_path)
            ]

            print(
                f"Iniciando Webots "
                f"na porta {port}..."
            )

            process = subprocess.Popen(
                command
            )

            self.processes.append(
                process
            )


    def stop_all(self):
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        deadline = time.monotonic() + 2.0
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
        deadline = time.monotonic() + 2.0
        for process in self.processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except (subprocess.TimeoutExpired, OSError):
                pass
        self.processes.clear()


if __name__ == "__main__":

    from config import load_config

    config = load_config()

    launcher = WebotsLauncher(
        config
    )

    print(
        "Portas:",
        launcher.get_ports()
    )

    launcher.start_webots_instances()