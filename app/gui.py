import sys
import secrets
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
    QScrollArea,
)

from config import load_config, save_config
from launcher import WebotsLauncher
from model_inspector import inspect_model
from run_manager import RunSession


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_PATH = PROJECT_ROOT / "training"
STOP_FILE = TRAINING_PATH / ".stop_training"
TEST_STOP_FILE = TRAINING_PATH / ".stop_test"


class SpotManager(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Webots RL Trainer")
        self.resize(700, 580)
        self.setMinimumSize(600, 450)

        self.config = load_config()

        self.launcher = None
        self.training_process = None
        self.test_process = None
        self.operation = None
        self.current_run = None
        self.stop_requested = False
        self.close_pending = False
        self.webots_closed = False
        self.transport_failure = None
        self.transport_log_tail = ""


        # Verificação de compatibilidade
        self.compatibility_process = None
        self.model_compatible = None

        self._build_ui()
        self.load_values()
        self.sync_ui_state()
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(500)
        self.health_timer.timeout.connect(self.check_webots_health)
        self.health_timer.start()

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def sync_ui_state(self):
        if not hasattr(self, "start_test_button"):
            return
        idle = self.operation is None
        running = self.launcher is not None
        self.config_panel.setEnabled(idle)
        for widget in (self.webots_input, self.world_input, self.venv_input,
                       self.controller_input, self.instances_input, self.port_input):
            widget.setEnabled(idle and not running)
        for button in self.transport_buttons:
            button.setEnabled(idle and not running)
        self.start_webots_button.setEnabled(idle and not running)
        self.stop_webots_button.setEnabled(idle and running)
        self.compatibility_button.setEnabled(idle and running)
        self.start_training_button.setEnabled(idle and running and self.model_compatible is True)
        self.stop_training_button.setEnabled(self.operation == "training" and not self.stop_requested)
        model_path = Path(self.model_input.text().strip())
        if model_path.suffix.lower() != ".zip":
            model_path = Path(str(model_path) + ".zip")
        self.start_test_button.setEnabled(idle and running and model_path.is_file())
        self.stop_test_button.setEnabled(self.operation == "testing" and not self.stop_requested)

    def finish_operation(self):
        operation = self.operation
        if self.webots_closed or self.transport_failure:
            if self.launcher is not None:
                self.launcher.stop_all()
                self.launcher = None
            self.model_compatible = None
            label = {"testing": "Teste", "training": "Treinamento", "checking": "Verificação"}.get(operation, "Operação")
            if self.webots_closed:
                reason = "uma instância do Webots foi fechada"
            elif self.transport_failure == "WORKER_TIMEOUT":
                reason = "uma instância não respondeu no prazo"
            else:
                reason = "a conexão com um worker foi encerrada ou falhou"
            self.status_label.setText(f"{label} interrompido: {reason}. Inicie o Webots novamente.")
            self.log_output.append("Sessão encerrada; instâncias restantes foram liberadas.")
        self.operation = None
        self.stop_requested = False
        self.sync_ui_state()
        if self.close_pending:
            self.close_pending = False
            QTimer.singleShot(0, self.close)

    def begin_run(
            self,
            operation,
            config,
            seed,
            ports,
        ):

            try:

                self.current_run = RunSession(
                    operation=operation,
                    config=config,
                    seed=seed,
                    ports=ports,
                )

            except Exception as error:

                self.current_run = None

                self.status_label.setText(
                    f"Erro ao criar run: {error}"
                )

                return False

            self.log_output.append(
                f"Run: "
                f"{self.current_run.directory}\n"
            )

            return True


    def finish_run(
        self,
        status,
        exit_code=None,
        details=None,
    ):

        if self.current_run is None:
            return

        try:

            self.current_run.finish(
                status=status,
                exit_code=exit_code,
                details=details,
            )

        finally:

            self.current_run = None

    def prepare_operation(self):
        self.webots_closed = False
        self.transport_failure = None
        self.transport_log_tail = ""

    def record_process_output(self, text):

        if self.current_run is not None:

            self.current_run.append(
                text
            )

        # QProcess can split a protocol tag across output chunks.
        combined = self.transport_log_tail + text
        for tag in ("WORKER_DISCONNECTED", "WORKER_TIMEOUT", "WORKER_ERROR", "WORKER_PROTOCOL"):
            if tag + "|" in combined:
                self.transport_failure = tag
        self.transport_log_tail = combined[-256:]

    def check_webots_health(self):
        if self.launcher is None:
            return
        exited = [p for p in self.launcher.processes if p.poll() is not None]
        if not exited:
            return
        self.webots_closed = True
        if self.operation is None:
            self.launcher.stop_all()
            self.launcher = None
            self.model_compatible = None
            self.status_label.setText("Webots fechado. Inicie as instâncias novamente.")
            self.sync_ui_state()
        else:
            self.status_label.setText("Webots fechado. Aguardando a liberação dos controllers...")

    def process_error(self, attribute, error):
        process = getattr(self, attribute)
        detail = process.errorString() if process is not None else str(error)
        self.log_output.append(f"Erro: {detail}")
        self.status_label.setText(f"Falha no processo: {detail}")
        # FailedToStart does not produce a finished signal.
        if error == QProcess.FailedToStart:

            self.finish_run(
                status="failed",
                details={
                    "process_error": detail,
                    "failed_to_start": True,
                },
            )

            setattr(
                self,
                attribute,
                None,
            )

            self.finish_operation()

    def start_test(self):
        self.check_webots_health()
        if self.operation is not None or self.launcher is None:
            return
        config = self.get_config()
        model_path = Path(config["model_path"].strip())
        if model_path.suffix.lower() != ".zip":
            model_path = Path(str(model_path) + ".zip")
        if not model_path.is_file():
            self.status_label.setText("Selecione um modelo salvo para testar.")
            return
        try:
            TEST_STOP_FILE.unlink(missing_ok=True)
        except OSError as error:
            self.status_label.setText(f"Não foi possível preparar o teste: {error}")
            return
        seed = self.resolve_seed(config)
        save_config(config)
        # Test uses the first existing instance; no new Webots is launched.
        port = self.launcher.get_ports()[0]

        self.log_output.clear()

        if not self.begin_run(
            operation="testing",
            config=config,
            seed=seed,
            ports=[port],
        ):
            return

        process = QProcess(self)
        self.test_process = process
        process.setProgram(str(Path(config["venv_path"]) / "Scripts" / "python.exe"))
        process.setArguments([
            "-u", str(TRAINING_PATH / "test_model.py"),
            "--ports", str(port), "--model-path", str(model_path),
            "--algorithm", config["algorithm"], "--seed", str(seed),
            "--episodes", str(config["test_episodes"]),
            "--stop-file", str(TEST_STOP_FILE),
            "--simulation-mode", "realtime",
        ])
        process.setWorkingDirectory(str(PROJECT_ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("WEBOTS_HOME", config["webots_home"])
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_test_output)
        process.finished.connect(self.test_finished)
        process.errorOccurred.connect(lambda error: self.process_error("test_process", error))
        self.log_output.append(f"Testando na porta {port}, seed {seed}, {config['test_episodes']} episódios. Tempo real.\n")
        self.prepare_operation()
        self.operation = "testing"
        self.stop_requested = False
        self.sync_ui_state()
        self.status_label.setText("Teste em execução.")
        process.start()

    def read_test_output(self):
        if self.test_process is not None:
            data = self.test_process.readAllStandardOutput()
            text = bytes(data).decode("utf-8", errors="replace")
            self.record_process_output(text)
            self.log_output.insertPlainText(text)
            scrollbar = self.log_output.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def stop_test(self):
        if self.operation != "testing" or self.stop_requested:
            return
        try:
            TEST_STOP_FILE.touch()
        except OSError as error:
            self.status_label.setText(f"Erro ao solicitar parada: {error}")
            return
        self.stop_requested = True
        self.status_label.setText("Aguardando o teste encerrar e liberar o ambiente...")
        self.sync_ui_state()

    def test_finished(self, exit_code, exit_status):
        self.read_test_output()
        stopped = self.stop_requested
        self.test_process = None
        try:
            TEST_STOP_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            message = "Teste interrompido." if stopped else "Teste concluído. Resultados no log."
        else:
            message = f"Teste falhou (código {exit_code}). Consulte o log."
        self.status_label.setText(message)

        success = (
            exit_code == 0
            and exit_status
            == QProcess.NormalExit
        )

        if success:

            status = (
                "stopped"
                if stopped
                else "completed"
            )

        else:

            status = "failed"

        self.finish_run(
            status=status,
            exit_code=exit_code,
            details={
                "stopped_by_user":
                    stopped,

                "transport_failure":
                    self.transport_failure,

                "webots_closed":
                    self.webots_closed,
            },
        )

        self.finish_operation()

    def ensure_runtime_directories(self):
        """
        Cria diretórios que são ignorados pelo Git e, portanto, podem não
        existir imediatamente após um clone novo do repositório.
        """
        (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "logs" / "tensorboard").mkdir(
            parents=True,
            exist_ok=True,
        )

    def clear_stop_file(self):
        """
        Remove um sinal de parada antigo, caso uma execução anterior tenha
        terminado de forma inesperada.
        """
        if STOP_FILE.exists():
            try:
                STOP_FILE.unlink()
            except OSError:
                pass

    def resolve_seed(self, config):

        if config.get(
            "random_seed",
            False,
        ):

            seed = secrets.randbelow(
                2_147_483_648
            )

            config["seed"] = seed

            self.seed_input.setValue(
                seed
            )

            return seed

        return config["seed"]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def start_training(self):
        self.check_webots_health()
        if self.operation is not None:
            return
        if self.launcher is None:
            self.status_label.setText(
                "Inicie as instâncias do Webots primeiro."
            )
            return

        if self.model_compatible is not True:
            self.status_label.setText(
                "Verifique a compatibilidade antes de iniciar o treinamento."
            )
            return

        if (
            self.training_process is not None
            and self.training_process.state() != QProcess.NotRunning
        ):
            self.status_label.setText(
                "Treinamento já está em execução."
            )
            return

        config = self.get_config()

        if not config["output_model_path"].strip():
            self.status_label.setText(
                "Informe o caminho do modelo de saída."
            )
            return

        seed = self.resolve_seed(
            config
        )

        save_config(config)

        self.ensure_runtime_directories()
        self.clear_stop_file()

        python_exe = (
            Path(config["venv_path"])
            / "Scripts"
            / "python.exe"
        )

        trainer_path = (
            TRAINING_PATH
            / "train_parallel.py"
        )

        ports = self.launcher.get_ports()
        ports_text = ",".join(str(port) for port in ports)

        algorithm = config["algorithm"]

        if algorithm != "ppo":
            self.status_label.setText(
                f"Algoritmo '{algorithm}' "
                "ainda não implementado."
            )
            return

        ppo = config[
            "algorithm_settings"
        ]["ppo"]

        arguments = [
            "-u",
            str(trainer_path),

            "--ports",
            ports_text,

            "--timesteps",
            str(
                config["total_timesteps"]
            ),

            "--learning-rate",
            str(
                ppo["learning_rate"]
            ),

            "--n-steps",
            str(
                ppo["n_steps"]
            ),

            "--batch-size",
            str(
                ppo["batch_size"]
            ),

            "--n-epochs",
            str(
                ppo["n_epochs"]
            ),

            "--gamma",
            str(
                ppo["gamma"]
            ),

            "--seed",
            str(seed),

            "--model-path",
            str(
                config["model_path"]
            ),

            "--output-model-path",
            str(
                config["output_model_path"]
            ),
        ]

        self.log_output.clear()

        if not self.begin_run(
            operation="training",
            config=config,
            seed=seed,
            ports=ports,
        ):
            return

        self.training_process = QProcess(self)
        self.training_process.setProgram(str(python_exe))
        self.training_process.setArguments(arguments)
        self.training_process.setWorkingDirectory(str(PROJECT_ROOT))

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert(
            "WEBOTS_HOME",
            config["webots_home"],
        )

        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        self.training_process.setProcessEnvironment(environment)
        self.training_process.setProcessChannelMode(
            QProcess.MergedChannels
        )

        self.training_process.readyReadStandardOutput.connect(
            self.read_training_output
        )
        self.training_process.finished.connect(
            self.training_finished
        )
        self.training_process.errorOccurred.connect(
            self.training_error
        )

        self.log_output.append("Iniciando treinamento...\n")

        rollout = (
            config["instances"]
            * ppo["n_steps"]
        )

        self.log_output.append(
            f"Robô: {config['robot']}"
        )

        self.log_output.append(
            f"Algoritmo: "
            f"{config['algorithm'].upper()}"
        )

        self.log_output.append(
            f"Modelo de origem: "
            f"{config['model_path']}"
        )

        self.log_output.append(
            f"Modelo de saída: "
            f"{config['output_model_path']}"
        )

        self.log_output.append(
            f"Portas: {ports_text}"
        )

        self.log_output.append(
            f"Instâncias: "
            f"{config['instances']}"
        )

        self.log_output.append(
            f"Learning rate: "
            f"{ppo['learning_rate']}"
        )

        self.log_output.append(
            f"N steps: {ppo['n_steps']}"
        )

        self.log_output.append(
            f"Batch size: "
            f"{ppo['batch_size']}"
        )

        self.log_output.append(
            f"N epochs: "
            f"{ppo['n_epochs']}"
        )

        self.log_output.append(
            f"Gamma: {ppo['gamma']}"
        )

        if config["random_seed"]:
            self.log_output.append(
                f"Seed: {seed} (aleatória)"
            )
        else:
            self.log_output.append(
                f"Seed: {seed}"
            )

        self.log_output.append(
            f"Rollout: "
            f"{config['instances']} × "
            f"{ppo['n_steps']} = "
            f"{rollout}\n"
        )

        self.prepare_operation()
        self.operation = "training"
        self.stop_requested = False
        self.sync_ui_state()
        self.training_process.start()






        self.status_label.setText(
            "Treinamento em execução."
        )


    def stop_training(self):
        """
        Solicita uma parada graciosa.

        O train_parallel.py monitora training/.stop_training por meio de
        StopTrainingCallback. Quando o arquivo aparece, model.learn()
        termina normalmente, o modelo é salvo e os workers são fechados,
        enquanto as instâncias do Webots permanecem abertas.
        """
        if self.training_process is None:
            return

        if self.training_process.state() == QProcess.NotRunning:
            return

        try:
            STOP_FILE.touch()
        except OSError as error:
            self.status_label.setText(
                f"Erro ao solicitar parada: {error}"
            )
            return

        self.status_label.setText(
            "Solicitando parada do treinamento..."
        )

        self.stop_requested = True
        self.sync_ui_state()

    def training_finished(
        self,
        exit_code,
        exit_status,
    ):

        self.read_training_output()

        self.clear_stop_file()

        stopped = (
            self.stop_requested
        )

        success = (
            exit_code == 0
            and exit_status
            == QProcess.NormalExit
            and self.transport_failure is None
            and not self.webots_closed
        )

        if success:

            status = (
                "stopped"
                if stopped
                else "completed"
            )

        else:

            status = "failed"

        self.finish_run(
            status=status,
            exit_code=exit_code,
            details={
                "stopped_by_user":
                    stopped,

                "transport_failure":
                    self.transport_failure,

                "webots_closed":
                    self.webots_closed,
            },
        )

        self.training_process = None

        if success:

            if stopped:

                self.status_label.setText(
                    "Treinamento interrompido "
                    "e modelo salvo."
                )

            else:

                self.status_label.setText(
                    "Treinamento finalizado "
                    "e modelo salvo."
                )

        else:

            self.status_label.setText(
                f"Treinamento falhou "
                f"(código {exit_code}). "
                "Consulte o log."
            )

        self.finish_operation()

    def training_error(self, error):
        self.process_error("training_process", error)

    def read_training_output(self):
        if self.training_process is None:
            return

        data = self.training_process.readAllStandardOutput()

        text = bytes(data).decode(
            "utf-8",
            errors="replace",
        )

        self.record_process_output(text)
        self.log_output.insertPlainText(text)

        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def check_compatibility(self):
        self.check_webots_health()
        if self.operation is not None:
            return

        if self.launcher is None:
            self.status_label.setText(
                "Inicie o Webots antes de verificar compatibilidade."
            )
            return

        if (
            self.compatibility_process is not None
            and self.compatibility_process.state()
            != QProcess.NotRunning
        ):
            self.status_label.setText(
                "Verificação já está em execução."
            )
            return

        config = self.get_config()

        model_path = (
            config["model_path"].strip()
        )

        if not model_path:
            self.status_label.setText(
                "Nenhum modelo selecionado."
            )
            return

        python_exe = (
            Path(config["venv_path"])
            / "Scripts"
            / "python.exe"
        )

        checker_path = (
            TRAINING_PATH
            / "check_compatibility.py"
        )

        ports = self.launcher.get_ports()

        ports_text = ",".join(
            str(port)
            for port in ports
        )

        arguments = [
            "-u",
            str(checker_path),

            "--ports",
            ports_text,

            "--model-path",
            model_path,

            "--algorithm",
            config["algorithm"],
        ]

        self.compatibility_process = QProcess(
            self
        )

        self.compatibility_process.setProgram(
            str(python_exe)
        )

        self.compatibility_process.setArguments(
            arguments
        )

        self.compatibility_process.setWorkingDirectory(
            str(PROJECT_ROOT)
        )

        environment = (
            QProcessEnvironment.systemEnvironment()
        )

        environment.insert(
            "WEBOTS_HOME",
            config["webots_home"],
        )

        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        self.compatibility_process.setProcessEnvironment(
            environment
        )

        self.compatibility_process.setProcessChannelMode(
            QProcess.MergedChannels
        )

        self.compatibility_process.readyReadStandardOutput.connect(
            self.read_compatibility_output
        )

        self.compatibility_process.finished.connect(
            self.compatibility_finished
        )

        self.model_compatible = None
        self.compatibility_output = ""


        self.log_output.append(
            "\n=== VERIFICANDO COMPATIBILIDADE ===\n"
        )

        self.status_label.setText(
            "Verificando compatibilidade..."
        )

        self.compatibility_process.errorOccurred.connect(
            lambda error: self.process_error("compatibility_process", error)
        )
        self.prepare_operation()
        self.operation = "checking"
        self.sync_ui_state()
        self.compatibility_process.start()

    def read_compatibility_output(self):

        if self.compatibility_process is None:
            return

        data = (
            self.compatibility_process
            .readAllStandardOutput()
        )

        text = bytes(data).decode(
            "utf-8",
            errors="replace",
        )

        self.record_process_output(text)
        self.compatibility_output += text

        self.log_output.insertPlainText(
            text
        )

        scrollbar = (
            self.log_output.verticalScrollBar()
        )

        scrollbar.setValue(
            scrollbar.maximum()
        )

    def compatibility_finished(self, exit_code, exit_status):
        self.read_compatibility_output()
        tags = {line.split("|", 1)[0] for line in self.compatibility_output.splitlines()}
        ok = exit_code == 0 and exit_status == QProcess.NormalExit
        self.model_compatible = bool(ok and tags.intersection({"COMPATIBLE", "NEW"}))
        self.status_label.setText(
            "Modelo pronto para treinamento." if self.model_compatible
            else "Verificação falhou. Consulte o log."
        )
        self.compatibility_process = None
        self.finish_operation()

    def invalidate_compatibility(self):
        self.model_compatible = None
        self.sync_ui_state()

    # ------------------------------------------------------------------
    # Webots
    # ------------------------------------------------------------------

    def start_webots(self):
        if self.operation is not None:
            return
        if not self.preflight_check():
            return

        if self.launcher is not None:
            self.status_label.setText(
                "Webots já estão em execução."
            )
            return

        config = self.get_config()
        save_config(config)

        try:
            self.launcher = WebotsLauncher(config)

            ports = self.launcher.get_ports()
            self.launcher.start_webots_instances()

            ports_text = ", ".join(
                str(port)
                for port in ports
            )

            self.status_label.setText(
                f"✓ Webots iniciados nas portas: {ports_text}"
            )



        except Exception as error:
            if self.launcher is not None:
                self.launcher.stop_all()

            self.launcher = None

            self.status_label.setText(
                f"✗ Erro ao iniciar Webots:\n{error}"
            )
        self.sync_ui_state()

    def stop_webots(self):
        if self.operation is not None:
            return
        if (
            self.training_process is not None
            and self.training_process.state() != QProcess.NotRunning
        ):
            self.status_label.setText(
                "Pare o treinamento antes de fechar os Webots."
            )
            return

        if self.launcher is None:
            self.status_label.setText(
                "Nenhuma instância do Webots foi iniciada pela interface."
            )
            return

        self.launcher.stop_all()
        self.launcher = None

        self.status_label.setText(
            "Webots encerrados."
        )


        self.model_compatible = None
        self.sync_ui_state()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        central_widget = QWidget()

        scroll_area.setWidget(
            central_widget
        )

        self.setCentralWidget(
            scroll_area
        )

        main_layout = QVBoxLayout(
            central_widget
        )

        main_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )

        main_layout.setSpacing(6)

        form = QFormLayout()

        form.setVerticalSpacing(5)

        # Robot
        self.robot_input = QComboBox()
        self.robot_input.addItem("Spot", "spot")

        form.addRow(
            "Robô:",
            self.robot_input,
        )

        # Algorithm
        self.algorithm_input = QComboBox()

        for algorithm in self.config[
            "available_algorithms"
        ]:
            self.algorithm_input.addItem(
                algorithm.upper(),
                algorithm,
            )

        form.addRow(
            "Algoritmo:",
            self.algorithm_input,
        )

        # Model
        self.model_input = QLineEdit()

        self.model_input.editingFinished.connect(
            self.inspect_selected_model
        )

        self.model_input.textChanged.connect(
            self.invalidate_compatibility
        )

        model_row = QHBoxLayout()
        model_row.addWidget(self.model_input)

        model_button = QPushButton(
            "Carregar..."
        )
        model_button.clicked.connect(
            self.select_model
        )

        model_row.addWidget(model_button)

        form.addRow(
            "Modelo:",
            model_row,
        )

        # Output model
        self.output_model_input = QLineEdit()

        output_model_row = QHBoxLayout()
        output_model_row.addWidget(
            self.output_model_input
        )

        output_model_button = QPushButton(
            "Salvar como..."
        )
        output_model_button.clicked.connect(
            self.select_output_model
        )

        output_model_row.addWidget(
            output_model_button
        )

        form.addRow(
            "Modelo de saída:",
            output_model_row,
        )

        self.model_status_label = QLabel(
            "Nenhum modelo carregado."
        )

        form.addRow(
            "Status do modelo:",
            self.model_status_label,
        )

        # Webots
        self.webots_input = QLineEdit()
        webots_row = QHBoxLayout()
        webots_row.addWidget(self.webots_input)

        webots_button = QPushButton("Procurar...")
        webots_button.clicked.connect(self.select_webots)
        webots_row.addWidget(webots_button)

        form.addRow(
            "Instalação do Webots:",
            webots_row,
        )

        # World
        self.world_input = QLineEdit()
        world_row = QHBoxLayout()
        world_row.addWidget(self.world_input)

        world_button = QPushButton("Procurar...")
        world_button.clicked.connect(self.select_world)
        world_row.addWidget(world_button)

        form.addRow("World:", world_row)

        # Venv
        self.venv_input = QLineEdit()
        venv_row = QHBoxLayout()
        venv_row.addWidget(self.venv_input)

        venv_button = QPushButton("Procurar...")
        venv_button.clicked.connect(self.select_venv)
        venv_row.addWidget(venv_button)

        form.addRow("Python env:", venv_row)

        # Controller
        self.controller_input = QLineEdit()
        controller_row = QHBoxLayout()
        controller_row.addWidget(self.controller_input)

        controller_button = QPushButton("Procurar...")
        controller_button.clicked.connect(
            self.select_controller
        )
        controller_row.addWidget(controller_button)

        form.addRow("Controller:", controller_row)
        self.transport_buttons = [webots_button, world_button, venv_button, controller_button]

        # Instances
        self.instances_input = QSpinBox()
        self.instances_input.setRange(1, 32)
        self.instances_input.valueChanged.connect(
            self.update_rollout
        )
        form.addRow("Instâncias:", self.instances_input)

        # Base port
        self.port_input = QSpinBox()
        self.port_input.setRange(1024, 65500)
        form.addRow("Porta inicial:", self.port_input)

        # Timesteps
        self.timesteps_input = QSpinBox()
        self.timesteps_input.setRange(
            1,
            2_000_000_000,
        )
        form.addRow(
            "Total timesteps:",
            self.timesteps_input,
        )

        # Learning rate
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setDecimals(7)
        self.learning_rate_input.setRange(
            0.0000001,
            1.0,
        )
        self.learning_rate_input.setSingleStep(
            0.00001
        )
        form.addRow(
            "Learning rate:",
            self.learning_rate_input,
        )

        # n_steps
        self.n_steps_input = QSpinBox()
        self.n_steps_input.setRange(1, 65536)
        self.n_steps_input.valueChanged.connect(
            self.update_rollout
        )
        form.addRow(
            "n_steps:",
            self.n_steps_input,
        )

        # Batch size
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(
            1,
            1_000_000,
        )

        form.addRow(
            "Batch size:",
            self.batch_size_input,
        )

        # N epochs
        self.n_epochs_input = QSpinBox()
        self.n_epochs_input.setRange(
            1,
            1000,
        )

        form.addRow(
            "N epochs:",
            self.n_epochs_input,
        )

        # Gamma
        self.gamma_input = QDoubleSpinBox()
        self.gamma_input.setDecimals(6)
        self.gamma_input.setRange(
            0.0,
            1.0,
        )
        self.gamma_input.setSingleStep(
            0.001
        )

        form.addRow(
            "Gamma:",
            self.gamma_input,
        )

        self.test_episodes_input = QSpinBox()
        self.test_episodes_input.setRange(1, 10000)
        form.addRow("Episódios de teste:", self.test_episodes_input)

        # Seed
        self.seed_input = QSpinBox()
        self.seed_input.setRange(
            0,
            2_147_483_647,
        )

        self.random_seed_input = QCheckBox(
            "Aleatória"
        )

        self.random_seed_input.toggled.connect(
            self.update_seed_mode
        )

        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed_input)
        seed_row.addWidget(self.random_seed_input)

        form.addRow(
            "Seed:",
            seed_row,
        )

        self.config_panel = QWidget()
        self.config_panel.setLayout(form)
        main_layout.addWidget(self.config_panel)

        self.rollout_label = QLabel()
        main_layout.addWidget(self.rollout_label)

        self.algorithm_input.currentIndexChanged.connect(
            self.update_rollout
        )

        # Config buttons
        config_buttons = QHBoxLayout()

        save_button = QPushButton(
            "Salvar configuração"
        )
        save_button.clicked.connect(self.save)

        check_button = QPushButton(
            "Verificar configuração"
        )
        check_button.clicked.connect(
            self.preflight_check
        )

        self.compatibility_button = QPushButton(
            "Verificar compatibilidade"
        )

        self.compatibility_button.setEnabled(False)

        self.compatibility_button.clicked.connect(
            self.check_compatibility
        )

        config_buttons.addWidget(save_button)
        config_buttons.addWidget(check_button)
        config_buttons.addWidget(
            self.compatibility_button
        )

        main_layout.addLayout(config_buttons)

        # Webots buttons
        webots_buttons = QHBoxLayout()

        self.start_webots_button = QPushButton(
            "▶ Iniciar Webots"
        )
        self.stop_webots_button = QPushButton(
            "■ Parar Webots"
        )

        self.stop_webots_button.setEnabled(False)

        self.start_webots_button.clicked.connect(
            self.start_webots
        )
        self.stop_webots_button.clicked.connect(
            self.stop_webots
        )

        webots_buttons.addWidget(
            self.start_webots_button
        )
        webots_buttons.addWidget(
            self.stop_webots_button
        )

        main_layout.addLayout(webots_buttons)

        # Training buttons
        training_buttons = QHBoxLayout()

        self.start_training_button = QPushButton(
            "▶ Iniciar treinamento"
        )
        self.stop_training_button = QPushButton(
            "■ Parar treinamento"
        )

        self.start_training_button.setEnabled(False)
        self.stop_training_button.setEnabled(False)

        self.start_training_button.clicked.connect(
            self.start_training
        )
        self.stop_training_button.clicked.connect(
            self.stop_training
        )

        training_buttons.addWidget(
            self.start_training_button
        )
        training_buttons.addWidget(
            self.stop_training_button
        )

        main_layout.addLayout(training_buttons)

        test_buttons = QHBoxLayout()
        self.start_test_button = QPushButton("▶ Testar modelo")
        self.stop_test_button = QPushButton("■ Parar teste")
        self.start_test_button.clicked.connect(self.start_test)
        self.stop_test_button.clicked.connect(self.stop_test)
        test_buttons.addWidget(self.start_test_button)
        test_buttons.addWidget(self.stop_test_button)
        main_layout.addLayout(test_buttons)

        # Log
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(120)
        main_layout.addWidget(self.log_output)

        self.status_label = QLabel(
            "Aguardando verificação."
        )
        main_layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def load_values(self):
        self.test_episodes_input.setValue(self.config.get("test_episodes", 3))

        self.webots_input.setText(
            self.config["webots_home"]
        )

        self.world_input.setText(
            self.config["world_path"]
        )

        self.venv_input.setText(
            self.config["venv_path"]
        )

        self.controller_input.setText(
            self.config["controller_path"]
        )

        self.model_input.setText(
            self.config["model_path"]
        )

        self.output_model_input.setText(
            self.config.get(
                "output_model_path",
                self.config["model_path"],
            )
        )

        # Robot
        robot_index = (
            self.robot_input.findData(
                self.config["robot"]
            )
        )

        if robot_index >= 0:
            self.robot_input.setCurrentIndex(
                robot_index
            )

        # Algorithm
        algorithm_index = (
            self.algorithm_input.findData(
                self.config["algorithm"]
            )
        )

        if algorithm_index >= 0:
            self.algorithm_input.setCurrentIndex(
                algorithm_index
            )

        self.instances_input.setValue(
            self.config["instances"]
        )

        self.port_input.setValue(
            self.config["base_port"]
        )

        self.timesteps_input.setValue(
            self.config["total_timesteps"]
        )

        self.seed_input.setValue(
            self.config["seed"]
        )

        self.random_seed_input.setChecked(
            self.config.get(
                "random_seed",
                False,
            )
        )

        self.update_seed_mode()

        # PPO
        ppo_config = self.config[
            "algorithm_settings"
        ]["ppo"]

        self.learning_rate_input.setValue(
            ppo_config["learning_rate"]
        )

        self.n_steps_input.setValue(
            ppo_config["n_steps"]
        )

        self.batch_size_input.setValue(
            ppo_config["batch_size"]
        )

        self.n_epochs_input.setValue(
            ppo_config["n_epochs"]
        )

        self.gamma_input.setValue(
            ppo_config["gamma"]
        )

        self.update_rollout()

    def get_config(self):

        # Preserva configurações que ainda
        # não estão expostas na interface.
        config = self.config.copy()

        config[
            "algorithm_settings"
        ] = {
            name: settings.copy()
            for name, settings
            in self.config[
                "algorithm_settings"
            ].items()
        }

        config["test_episodes"] = self.test_episodes_input.value()

        config["robot"] = (
            self.robot_input.currentData()
        )

        config["algorithm"] = (
            self.algorithm_input.currentData()
        )

        config["webots_home"] = (
            self.webots_input.text()
        )

        config["world_path"] = (
            self.world_input.text()
        )

        config["venv_path"] = (
            self.venv_input.text()
        )

        config["controller_path"] = (
            self.controller_input.text()
        )

        config["model_path"] = (
            self.model_input.text()
        )

        config["output_model_path"] = (
            self.output_model_input.text()
        )

        config["instances"] = (
            self.instances_input.value()
        )

        config["base_port"] = (
            self.port_input.value()
        )

        config["total_timesteps"] = (
            self.timesteps_input.value()
        )

        config["seed"] = (
            self.seed_input.value()
        )

        config["random_seed"] = (
            self.random_seed_input.isChecked()
        )

        ppo = config[
            "algorithm_settings"
        ]["ppo"]

        ppo["learning_rate"] = (
            self.learning_rate_input.value()
        )

        ppo["n_steps"] = (
            self.n_steps_input.value()
        )

        ppo["batch_size"] = (
            self.batch_size_input.value()
        )

        ppo["n_epochs"] = (
            self.n_epochs_input.value()
        )

        ppo["gamma"] = (
            self.gamma_input.value()
        )

        return config

    def save(self):
        config = self.get_config()
        save_config(config)

        self.status_label.setText(
            "Configuração salva."
        )

    def update_seed_mode(
        self,
        checked=None,
    ):

        if checked is None:
            checked = (
                self.random_seed_input.isChecked()
            )

        self.seed_input.setEnabled(
            not checked
        )

    def update_rollout(self):

        algorithm = (
            self.algorithm_input.currentData()
        )

        if algorithm != "ppo":
            self.rollout_label.setText(
                "Rollout total: "
                "não aplicável."
            )
            return

        instances = (
            self.instances_input.value()
        )

        n_steps = (
            self.n_steps_input.value()
        )

        total = (
            instances
            * n_steps
        )

        self.rollout_label.setText(
            f"Rollout total: "
            f"{instances} × "
            f"{n_steps} = "
            f"{total}"
        )

    def preflight_check(self):
        config = self.get_config()
        errors = []

        # Webots
        webots_home = Path(config["webots_home"])

        webots_exe = (
            webots_home
            / "msys64"
            / "mingw64"
            / "bin"
            / "webots.exe"
        )

        controller_exe = (
            webots_home
            / "msys64"
            / "mingw64"
            / "bin"
            / "webots-controller.exe"
        )

        if not webots_exe.exists():
            errors.append(
                "webots.exe não encontrado."
            )

        if not controller_exe.exists():
            errors.append(
                "webots-controller.exe não encontrado."
            )

        # World
        if not Path(config["world_path"]).exists():
            errors.append(
                "World não encontrado."
            )

        # Python env
        python_exe = (
            Path(config["venv_path"])
            / "Scripts"
            / "python.exe"
        )

        if not python_exe.exists():
            errors.append(
                "Python do venv não encontrado."
            )

        # Controller
        controller_path = Path(
            config["controller_path"]
        )

        controller_required_files = [
            "spot_env.py",
            "motors.py",
            "sensors.py",
            "reward.py",
            "parallel_worker.py",
        ]

        for filename in controller_required_files:
            file_path = controller_path / filename

            if not file_path.exists():
                errors.append(
                    f"{filename} não encontrado no controller."
                )

        # Training
        training_required_files = [
            "train_parallel.py",
            "webots_vec_env.py",
            "check_compatibility.py",
            "test_model.py",
        ]

        for filename in training_required_files:
            file_path = TRAINING_PATH / filename

            if not file_path.exists():
                errors.append(
                    f"{filename} não encontrado em training."
                )

        if errors:
            text = (
                "Falha no pre-flight:\n\n"
                + "\n".join(
                    f"✗ {error}"
                    for error in errors
                )
            )

            self.status_label.setText(text)
            return False

        self.status_label.setText(
            "✓ Configuração válida. Pronto para o launcher."
        )
        return True

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def select_model(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione um modelo",
            str(PROJECT_ROOT / "models"),
            "Stable-Baselines3 (*.zip)",
        )

        if not path:
            return

        self.model_input.setText(path)

        self.inspect_selected_model()

    def select_output_model(self):

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar modelo treinado como",
            str(PROJECT_ROOT / "models"),
            "Stable-Baselines3 (*.zip)",
        )

        if not path:
            return

        if not path.lower().endswith(".zip"):
            path += ".zip"

        self.output_model_input.setText(
            path
        )

    def inspect_selected_model(self):

        path = self.model_input.text().strip()

        if not path:
            self.model_status_label.setText(
                "Nenhum modelo selecionado."
            )
            return

        algorithm = (
            self.algorithm_input.currentData()
        )

        result = inspect_model(
            path,
            algorithm,
        )

        if not result.get("exists"):
            self.model_status_label.setText(
                "Novo modelo — arquivo ainda não existe."
            )

            self.log_output.append(
                "\nModelo não encontrado."
            )

            self.log_output.append(
                "Um novo modelo será criado "
                "neste caminho ao iniciar o treino.\n"
            )

            return

        if not result.get("valid"):
            self.model_status_label.setText(
                "✗ Modelo inválido."
            )

            self.log_output.append(
                "\nFalha ao carregar modelo:"
            )

            self.log_output.append(
                result.get(
                    "error",
                    "Erro desconhecido.",
                )
            )

            return

        self.model_status_label.setText(
            f"✓ {result['algorithm'].upper()} | "
            f"{result['observation_space']} | "
            f"{result['action_space']}"
        )

        self.log_output.append(
            "\n=== MODELO CARREGADO ==="
        )

        self.log_output.append(
            f"Arquivo: {path}"
        )

        self.log_output.append(
            f"Algoritmo: "
            f"{result['algorithm'].upper()}"
        )

        self.log_output.append(
            f"Observation space: "
            f"{result['observation_space']}"
        )

        self.log_output.append(
            f"Action space: "
            f"{result['action_space']}"
        )

        # Preenche os parâmetros PPO
        if algorithm == "ppo":

            if "learning_rate" in result:
                self.learning_rate_input.setValue(
                    float(
                        result["learning_rate"]
                    )
                )

            if "n_steps" in result:
                self.n_steps_input.setValue(
                    int(
                        result["n_steps"]
                    )
                )

            if "batch_size" in result:
                self.batch_size_input.setValue(
                    int(
                        result["batch_size"]
                    )
                )

            if "n_epochs" in result:
                self.n_epochs_input.setValue(
                    int(
                        result["n_epochs"]
                    )
                )

            if "gamma" in result:
                self.gamma_input.setValue(
                    float(
                        result["gamma"]
                    )
                )

        self.update_rollout()

        self.log_output.append(
            "\nParâmetros carregados na interface."
        )

        self.log_output.append(
            "Você pode alterá-los antes "
            "de continuar o treinamento.\n"
        )

    def select_webots(self):
        if self.launcher is not None or self.operation is not None:
            return
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecione a pasta do Webots",
        )

        if path:
            self.webots_input.setText(path)

    def select_world(self):
        if self.launcher is not None or self.operation is not None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione o world",
            "",
            "Webots World (*.wbt)",
        )

        if path:
            self.world_input.setText(path)

    def select_venv(self):
        if self.launcher is not None or self.operation is not None:
            return
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecione o ambiente Python",
        )

        if path:
            self.venv_input.setText(path)

    def select_controller(self):
        if self.launcher is not None or self.operation is not None:
            return
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecione a pasta do controller",
        )

        if path:
            self.controller_input.setText(path)

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.operation is not None:
            self.close_pending = True
            event.ignore()
            if self.operation == "training":
                self.stop_training()
            elif self.operation == "testing":
                self.stop_test()
            else:
                self.status_label.setText("Aguardando a verificação terminar para fechar.")
            return
        if self.launcher is not None:
            self.launcher.stop_all()
            self.launcher = None
        event.accept()


def main():
    app = QApplication(sys.argv)

    window = SpotManager()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
