import sys
import secrets
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment
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
)

from config import load_config, save_config
from launcher import WebotsLauncher
from model_inspector import inspect_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_PATH = PROJECT_ROOT / "training"
STOP_FILE = TRAINING_PATH / ".stop_training"


class SpotManager(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Webots RL Trainer")
        self.resize(750, 650)

        self.config = load_config()

        self.launcher = None
        self.training_process = None
        self.test_process = None


        # Verificação de compatibilidade
        self.compatibility_process = None
        self.model_compatible = None

        self._build_ui()
        self.load_values()

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

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
        ]

        self.training_process = QProcess(self)
        self.training_process.setProgram(str(python_exe))
        self.training_process.setArguments(arguments)
        self.training_process.setWorkingDirectory(str(PROJECT_ROOT))

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert(
            "WEBOTS_HOME",
            config["webots_home"],
        )

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

        self.log_output.clear()
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
            f"Modelo: {config['model_path']}"
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

        self.training_process.start()

        self.start_webots_button.setEnabled(
            False
        )

        self.stop_webots_button.setEnabled(
            False
        )

        self.start_training_button.setEnabled(
            False
        )

        self.stop_training_button.setEnabled(
            True
        )

        self.compatibility_button.setEnabled(
            False
        )

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

        # Evita pedidos repetidos enquanto o trainer encerra.
        self.stop_training_button.setEnabled(False)

    def training_finished(self, exit_code, exit_status):
        self.clear_stop_file()

        if exit_code == 0:
            self.status_label.setText(
                "Treinamento finalizado corretamente."
            )
        else:
            self.status_label.setText(
                f"Treinamento finalizado com código {exit_code}."
            )

        self.start_training_button.setEnabled(
            self.launcher is not None
        )
        self.stop_training_button.setEnabled(False)
        self.start_webots_button.setEnabled(
            self.launcher is None
        )
        self.stop_webots_button.setEnabled(
            self.launcher is not None
        )

        self.training_process = None

    def training_error(self, error):
        self.status_label.setText(
            f"Erro no processo de treinamento: {error}"
        )

    def read_training_output(self):
        if self.training_process is None:
            return

        data = self.training_process.readAllStandardOutput()

        text = bytes(data).decode(
            "utf-8",
            errors="replace",
        )

        self.log_output.insertPlainText(text)

        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def check_compatibility(self):

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

        self.compatibility_button.setEnabled(False)
        self.start_training_button.setEnabled(False)

        self.log_output.append(
            "\n=== VERIFICANDO COMPATIBILIDADE ===\n"
        )

        self.status_label.setText(
            "Verificando compatibilidade..."
        )

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

    def compatibility_finished(
        self,
        exit_code,
        exit_status,
    ):

        output = self.compatibility_output

        if "INCOMPATIBLE|" in output:

            self.model_compatible = False

            self.status_label.setText(
                "✗ Modelo incompatível com o ambiente."
            )

            self.start_training_button.setEnabled(
                False
            )

        elif "COMPATIBLE|" in output:

            self.model_compatible = True

            self.status_label.setText(
                "✓ Modelo compatível com o ambiente."
            )

            self.start_training_button.setEnabled(
                True
            )

        elif "NEW|" in output:

            self.model_compatible = True

            self.status_label.setText(
                "✓ Modelo novo. Pronto para treinar."
            )

            self.start_training_button.setEnabled(
                True
            )

        else:

            self.model_compatible = None

            self.status_label.setText(
                "Não foi possível verificar "
                "a compatibilidade."
            )

            self.start_training_button.setEnabled(
                False
            )

        self.compatibility_button.setEnabled(
            self.launcher is not None
        )

        self.compatibility_process = None

    def invalidate_compatibility(self):

        self.model_compatible = None

        if self.launcher is not None:
            self.start_training_button.setEnabled(
                False
            )

            self.compatibility_button.setEnabled(
                True
            )

    # ------------------------------------------------------------------
    # Webots
    # ------------------------------------------------------------------

    def start_webots(self):
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

            self.start_webots_button.setEnabled(False)
            self.stop_webots_button.setEnabled(True)
            self.start_training_button.setEnabled(False)

            self.compatibility_button.setEnabled(True)

        except Exception as error:
            if self.launcher is not None:
                self.launcher.stop_all()

            self.launcher = None

            self.status_label.setText(
                f"✗ Erro ao iniciar Webots:\n{error}"
            )

    def stop_webots(self):
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

        self.start_webots_button.setEnabled(True)
        self.stop_webots_button.setEnabled(False)
        self.start_training_button.setEnabled(False)
        self.stop_training_button.setEnabled(False)

        self.compatibility_button.setEnabled(False)
        self.model_compatible = None

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        form = QFormLayout()

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

        main_layout.addLayout(form)

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

        # Log
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(220)
        main_layout.addWidget(self.log_output)

        self.status_label = QLabel(
            "Aguardando verificação."
        )
        main_layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def load_values(self):

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
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecione a pasta do Webots",
        )

        if path:
            self.webots_input.setText(path)

    def select_world(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione o world",
            "",
            "Webots World (*.wbt)",
        )

        if path:
            self.world_input.setText(path)

    def select_venv(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecione o ambiente Python",
        )

        if path:
            self.venv_input.setText(path)

    def select_controller(self):
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
        """
        Ao fechar a aplicação inteira, tenta primeiro solicitar parada
        graciosa do trainer. Se ele não finalizar em alguns segundos,
        encerra o processo à força. Em seguida fecha as instâncias do
        Webots abertas pelo manager.
        """
        if (
            self.training_process is not None
            and self.training_process.state() != QProcess.NotRunning
        ):
            try:
                STOP_FILE.touch()
            except OSError:
                pass

            finished = self.training_process.waitForFinished(
                5000
            )

            if not finished:
                self.training_process.kill()
                self.training_process.waitForFinished(3000)

            self.training_process = None

        self.clear_stop_file()

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
