import json
import re
import os
import tempfile
from uuid import uuid4
import subprocess

from copy import deepcopy
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_PATH = PROJECT_ROOT / "runs"


def get_git_value(*args):

    try:

        result = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()

    except Exception:
        return None


class RunSession:

    def __init__(
        self,
        operation,
        config,
        seed,
        ports,
    ):

        self.operation = operation

        now = datetime.now().astimezone()

        robot_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(config["robot"])).strip("_") or "robot"
        run_id = (
            f"{now.strftime('%Y-%m-%d_%H-%M-%S-%f')}_"
            f"{robot_slug}_"
            f"{config['algorithm']}_"
            f"{operation}_{uuid4().hex}"
        )

        RUNS_PATH.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.directory = (
            RUNS_PATH
            / run_id
        )

        self.directory.mkdir(
            parents=False,
            exist_ok=False,
        )

        if operation == "training":
            log_name = "training.log"
        else:
            log_name = "test.log"

        self.log_path = (
            self.directory
            / log_name
        )

        self.metadata_path = (
            self.directory
            / "run.json"
        )

        self.config_path = (
            self.directory
            / "config.json"
        )

        effective_config = deepcopy(
            config
        )

        # Garante que o snapshot contém
        # a seed realmente utilizada.
        effective_config["seed"] = seed

        input_model_path = Path(
            config["model_path"]
        )

        if (
            input_model_path.suffix.lower()
            != ".zip"
        ):
            input_model_path = Path(
                str(input_model_path)
                + ".zip"
            )


        output_model_path = Path(
            config.get(
                "output_model_path",
                config["model_path"],
            )
        )

        if (
            output_model_path.suffix.lower()
            != ".zip"
        ):
            output_model_path = Path(
                str(output_model_path)
                + ".zip"
            )

        self.metadata = {

            "run_id": run_id,

            "operation": operation,

            "status": "running",

            "started_at":
                now.isoformat(),

            "finished_at": None,

            "robot":
                config["robot"],

            "algorithm":
                config["algorithm"],

            "seed":
                seed,

            "random_seed":
                bool(
                    config.get(
                        "random_seed",
                        False,
                    )
                ),

            "ports":
                list(ports),

            "instances":
                len(ports),

            "input_model_path":
                str(input_model_path),

            "output_model_path":
                str(output_model_path),

            "input_model_existed_at_start":
                input_model_path.is_file(),

            "output_model_existed_at_start":
                output_model_path.is_file(),

            "env_class": config.get("env_class", "spot_env:SpotEnv"),
            "env_path": config["controller_path"],

            "world_path":
                config["world_path"],

            "git_branch":
                get_git_value(
                    "rev-parse",
                    "--abbrev-ref",
                    "HEAD",
                ),

            "git_commit":
                get_git_value(
                    "rev-parse",
                    "HEAD",
                ),
        }

        if operation == "training":

            self.metadata[
                "requested_timesteps"
            ] = config[
                "total_timesteps"
            ]

        elif operation == "testing":

            self.metadata[
                "test_episodes"
            ] = config[
                "test_episodes"
            ]

            self.metadata[
                "deterministic"
            ] = True

            self.metadata[
                "simulation_mode"
            ] = "realtime"

        self._write_json(
            self.config_path,
            effective_config,
        )

        self._write_json(
            self.metadata_path,
            self.metadata,
        )

        # Cria o log imediatamente.
        self.log_path.touch()

    def _write_json(self, path, data):
        # Replace only after a complete, flushed write in the same directory.
        # If serialization/write/replace fails, the previous JSON stays intact.
        path = Path(path)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as file:
                temporary_path = Path(file.name)
                json.dump(data, file, indent=4, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def append(
        self,
        text,
    ):

        if not text:
            return

        with open(
            self.log_path,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(text)

    def finish(
        self,
        status,
        exit_code=None,
        details=None,
    ):

        self.metadata[
            "status"
        ] = status

        self.metadata[
            "finished_at"
        ] = (
            datetime.now()
            .astimezone()
            .isoformat()
        )

        self.metadata[
            "exit_code"
        ] = exit_code

        if details:

            self.metadata.update(
                details
            )

        self._write_json(
            self.metadata_path,
            self.metadata,
        )