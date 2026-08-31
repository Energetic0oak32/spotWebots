from PySide6.QtCore import (
    QProcess,
    QProcessEnvironment
)

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget
)

from config import load_config, save_config

from launcher import WebotsLauncher

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


class SpotManager(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Spot Training Manager"
        )

        self.resize(750, 500)

        self.config = load_config()

        self.launcher = None
        self.training_process = None

        self._build_ui()

        self.load_values()


    def start_training(self):

        if self.launcher is None:

            self.status_label.setText(
                "Inicie as instâncias do Webots primeiro."
            )

            return

        if (
            self.training_process is not None
            and self.training_process.state()
            != QProcess.NotRunning
        ):

            self.status_label.setText(
                "Treinamento já está em execução."
            )

            return

        config = self.get_config()

        save_config(config)

        python_exe = (
            Path(config["venv_path"])
            / "Scripts"
            / "python.exe"
        )

        trainer_path = (
            PROJECT_ROOT
            / "training"
            / "train_parallel.py"
        )

        ports = (
            self.launcher.get_ports()
        )

        ports_text = ",".join(
            str(port)
            for port in ports
        )

        arguments = [
            "-u",
            str(trainer_path),

            "--ports",
            ports_text,

            "--timesteps",
            str(config["total_timesteps"]),

            "--learning-rate",
            str(config["learning_rate"]),

            "--n-steps",
            str(config["n_steps"])
        ]

        self.training_process = QProcess(
            self
        )

        self.training_process.setProgram(
            str(python_exe)
        )

        self.training_process.setArguments(
            arguments
        )

        self.training_process.setWorkingDirectory(
            str(PROJECT_ROOT)
        )

        # Passa WEBOTS_HOME para o trainer
        environment = (
            QProcessEnvironment.systemEnvironment()
        )

        environment.insert(
            "WEBOTS_HOME",
            config["webots_home"]
        )

        self.training_process.setProcessEnvironment(
            environment
        )

        # Junta stdout + stderr
        self.training_process.setProcessChannelMode(
            QProcess.MergedChannels
        )

        # Quando aparecer texto...
        self.training_process.readyReadStandardOutput.connect(
            self.read_training_output
        )

        # Quando terminar...
        self.training_process.finished.connect(
            self.training_finished
        )

        self.log_output.clear()

        self.log_output.append(
            "Iniciando treinamento...\n"
        )

        self.log_output.append(
            f"Portas: {ports_text}"
        )

        self.log_output.append(
            f"Instâncias: {config['instances']}"
        )

        self.log_output.append(
            f"Rollout: "
            f"{config['instances']} × "
            f"{config['n_steps']} = "
            f"{config['instances'] * config['n_steps']}\n"
        )

        self.training_process.start()

        self.start_training_button.setEnabled(
            False
        )

        self.stop_training_button.setEnabled(
            True
        )

        self.start_webots_button.setEnabled(
            False
        )

        self.stop_webots_button.setEnabled(
            False
        )

        self.status_label.setText(
            "Treinamento em execução."
        )


    def start_webots(self):

        # Verifica tudo antes de iniciar
        if not self.preflight_check():
            return

        # Evita abrir outra leva por acidente
        if self.launcher is not None:

            self.status_label.setText(
                "Webots já estão em execução."
            )

            return

        config = self.get_config()

        # Salva automaticamente a configuração usada
        save_config(config)

        try:

            self.launcher = WebotsLauncher(
                config
            )

            ports = self.launcher.get_ports()

            self.launcher.start_webots_instances()

            ports_text = ", ".join(
                str(port)
                for port in ports
            )

            self.status_label.setText(
                f"✓ Webots iniciados nas portas: "
                f"{ports_text}"
            )

            self.start_webots_button.setEnabled(
                False
            )

            self.stop_webots_button.setEnabled(
                True
            )

            self.start_training_button.setEnabled(
                True
            )

        except Exception as error:

            if self.launcher is not None:
                self.launcher.stop_all()

            self.launcher = None

            self.status_label.setText(
                f"✗ Erro ao iniciar Webots:\n"
                f"{error}"
            )


    def stop_webots(self):

        if (
            self.training_process is not None
            and self.training_process.state()
            != QProcess.NotRunning
        ):

            self.status_label.setText(
                "Pare o treinamento antes de "
                "fechar os Webots."
            )

            return

        if self.launcher is None:

            self.status_label.setText(
                "Nenhuma instância do Webots "
                "foi iniciada pela interface."
            )

            return

        self.launcher.stop_all()

        self.launcher = None

        self.status_label.setText(
            "Webots encerrados."
        )

        self.start_webots_button.setEnabled(
            True
        )

        self.stop_webots_button.setEnabled(
            False
        )

        self.start_training_button.setEnabled(
            False
        )


    def _build_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(
            central_widget
        )

        main_layout = QVBoxLayout(
            central_widget
        )

        form = QFormLayout()

        # -------------------------
        # WEBOTS
        # -------------------------

        self.webots_input = QLineEdit()

        webots_row = QHBoxLayout()

        webots_row.addWidget(
            self.webots_input
        )

        webots_button = QPushButton(
            "Procurar..."
        )

        webots_button.clicked.connect(
            self.select_webots
        )

        webots_row.addWidget(
            webots_button
        )

        form.addRow(
            "Instalação do Webots:",
            webots_row
        )

        # -------------------------
        # WORLD
        # -------------------------

        self.world_input = QLineEdit()

        world_row = QHBoxLayout()

        world_row.addWidget(
            self.world_input
        )

        world_button = QPushButton(
            "Procurar..."
        )

        world_button.clicked.connect(
            self.select_world
        )

        world_row.addWidget(
            world_button
        )

        form.addRow(
            "World:",
            world_row
        )

        # -------------------------
        # VENV
        # -------------------------

        self.venv_input = QLineEdit()

        venv_row = QHBoxLayout()

        venv_row.addWidget(
            self.venv_input
        )

        venv_button = QPushButton(
            "Procurar..."
        )

        venv_button.clicked.connect(
            self.select_venv
        )

        venv_row.addWidget(
            venv_button
        )

        form.addRow(
            "Python env:",
            venv_row
        )

        # -------------------------
        # CONTROLLER
        # -------------------------

        self.controller_input = QLineEdit()

        controller_row = QHBoxLayout()

        controller_row.addWidget(
            self.controller_input
        )

        controller_button = QPushButton(
            "Procurar..."
        )

        controller_button.clicked.connect(
            self.select_controller
        )

        controller_row.addWidget(
            controller_button
        )

        form.addRow(
            "Controller:",
            controller_row
        )

        # -------------------------
        # INSTÂNCIAS
        # -------------------------

        self.instances_input = QSpinBox()

        self.instances_input.setRange(
            1,
            32
        )

        self.instances_input.valueChanged.connect(
            self.update_rollout
        )

        form.addRow(
            "Instâncias:",
            self.instances_input
        )

        # -------------------------
        # PORTA
        # -------------------------

        self.port_input = QSpinBox()

        self.port_input.setRange(
            1024,
            65500
        )

        form.addRow(
            "Porta inicial:",
            self.port_input
        )

        # -------------------------
        # TIMESTEPS
        # -------------------------

        self.timesteps_input = QSpinBox()

        self.timesteps_input.setRange(
            1,
            2_000_000_000
        )

        form.addRow(
            "Total timesteps:",
            self.timesteps_input
        )

        # -------------------------
        # LEARNING RATE
        # -------------------------

        self.learning_rate_input = (
            QDoubleSpinBox()
        )

        self.learning_rate_input.setDecimals(
            7
        )

        self.learning_rate_input.setRange(
            0.0000001,
            1.0
        )

        self.learning_rate_input.setSingleStep(
            0.00001
        )

        form.addRow(
            "Learning rate:",
            self.learning_rate_input
        )

        # -------------------------
        # N_STEPS
        # -------------------------

        self.n_steps_input = QSpinBox()

        self.n_steps_input.setRange(
            1,
            65536
        )

        self.n_steps_input.valueChanged.connect(
            self.update_rollout
        )

        form.addRow(
            "n_steps:",
            self.n_steps_input
        )

        main_layout.addLayout(
            form
        )

        # -------------------------
        # ROLLOUT
        # -------------------------

        self.rollout_label = QLabel()

        main_layout.addWidget(
            self.rollout_label
        )

        # -------------------------
        # BOTÕES
        # -------------------------

        buttons = QHBoxLayout()

        save_button = QPushButton(
            "Salvar configuração"
        )

        save_button.clicked.connect(
            self.save
        )

        check_button = QPushButton(
            "Verificar configuração"
        )

        check_button.clicked.connect(
            self.preflight_check
        )

        buttons.addWidget(
            save_button
        )

        buttons.addWidget(
            check_button
        )

        main_layout.addLayout(
            buttons
        )

        webots_buttons = QHBoxLayout()

        self.start_webots_button = QPushButton(
            "▶ Iniciar Webots"
        )

        self.stop_webots_button = QPushButton(
            "■ Parar Webots"
        )

        self.stop_webots_button.setEnabled(
            False
        )

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

        main_layout.addLayout(
            webots_buttons
        )

        training_buttons = QHBoxLayout()

        self.start_training_button = QPushButton(
            "▶ Iniciar treinamento"
        )

        self.stop_training_button = QPushButton(
            "■ Parar treinamento"
        )

        self.start_training_button.setEnabled(
            False
        )

        self.stop_training_button.setEnabled(
            False
        )

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

        main_layout.addLayout(
            training_buttons
        )

        # -------------------------
        # STATUS
        # -------------------------

        self.log_output = QTextEdit()

        self.log_output.setReadOnly(
            True
        )

        self.log_output.setMinimumHeight(
            200
        )

        main_layout.addWidget(
            self.log_output
        )

        self.status_label = QLabel(
            "Aguardando verificação."
        )

        main_layout.addWidget(
            self.status_label
        )


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

        self.instances_input.setValue(
            self.config["instances"]
        )

        self.port_input.setValue(
            self.config["base_port"]
        )

        self.timesteps_input.setValue(
            self.config["total_timesteps"]
        )

        self.learning_rate_input.setValue(
            self.config["learning_rate"]
        )

        self.n_steps_input.setValue(
            self.config["n_steps"]
        )

        self.update_rollout()


    def get_config(self):

        return {
            "webots_home":
                self.webots_input.text(),

            "world_path":
                self.world_input.text(),

            "venv_path":
                self.venv_input.text(),

            "controller_path":
                self.controller_input.text(),

            "instances":
                self.instances_input.value(),

            "base_port":
                self.port_input.value(),

            "total_timesteps":
                self.timesteps_input.value(),

            "learning_rate":
                self.learning_rate_input.value(),

            "n_steps":
                self.n_steps_input.value()
        }


    def save(self):

        config = self.get_config()

        save_config(config)

        self.status_label.setText(
            "Configuração salva."
        )


    def read_training_output(self):

        if self.training_process is None:
            return

        data = (
            self.training_process
            .readAllStandardOutput()
        )

        text = bytes(
            data
        ).decode(
            "utf-8",
            errors="replace"
        )

        self.log_output.insertPlainText(
            text
        )

        scrollbar = (
            self.log_output
            .verticalScrollBar()
        )

        scrollbar.setValue(
            scrollbar.maximum()
        )


    def closeEvent(self, event):

        if (
            self.training_process is not None
            and self.training_process.state()
            != QProcess.NotRunning
        ):

            self.training_process.kill()

            self.training_process.waitForFinished(
                3000
            )

            self.training_process = None

        if self.launcher is not None:

            self.launcher.stop_all()

            self.launcher = None

        event.accept()


    def update_rollout(self):

        instances = (
            self.instances_input.value()
        )

        n_steps = (
            self.n_steps_input.value()
        )

        total = instances * n_steps

        self.rollout_label.setText(
            f"Rollout total: "
            f"{instances} × "
            f"{n_steps} = "
            f"{total}"
        )


    def preflight_check(self):

        config = self.get_config()

        errors = []

        # -------------------------
        # WEBOTS
        # -------------------------

        webots_home = Path(
            config["webots_home"]
        )

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
                "webots-controller.exe "
                "não encontrado."
            )

        # -------------------------
        # WORLD
        # -------------------------

        if not Path(
            config["world_path"]
        ).exists():

            errors.append(
                "World não encontrado."
            )

        # -------------------------
        # PYTHON ENV
        # -------------------------

        python_exe = (
            Path(config["venv_path"])
            / "Scripts"
            / "python.exe"
        )

        if not python_exe.exists():

            errors.append(
                "Python do venv "
                "não encontrado."
            )

        # -------------------------
        # CONTROLLER
        # -------------------------

        controller_path = Path(
            config["controller_path"]
        )

        controller_required_files = [
            "spot_env.py",
            "motors.py",
            "sensors.py",
            "reward.py",
            "parallel_worker.py"
        ]

        for filename in controller_required_files:

            file_path = (
                controller_path
                / filename
            )

            if not file_path.exists():

                errors.append(
                    f"{filename} "
                    f"não encontrado no controller."
                )

        # -------------------------
        # TRAINING
        # -------------------------

        training_path = (
            PROJECT_ROOT
            / "training"
        )

        training_required_files = [
            "train_parallel.py",
            "webots_vec_env.py"
        ]

        for filename in training_required_files:

            file_path = (
                training_path
                / filename
            )

            if not file_path.exists():

                errors.append(
                    f"{filename} "
                    f"não encontrado em training."
                )

        # -------------------------
        # RESULTADO
        # -------------------------

        if errors:

            text = (
                "Falha no pre-flight:\n\n"
                + "\n".join(
                    f"✗ {error}"
                    for error in errors
                )
            )

            self.status_label.setText(
                text
            )

            return False

        self.status_label.setText(
            "✓ Configuração válida. "
            "Pronto para o launcher."
        )

        return True


    def select_webots(self):

        path = (
            QFileDialog.getExistingDirectory(
                self,
                "Selecione a pasta do Webots"
            )
        )

        if path:

            self.webots_input.setText(
                path
            )


    def select_world(self):

        path, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Selecione o world",
                "",
                "Webots World (*.wbt)"
            )
        )

        if path:

            self.world_input.setText(
                path
            )


    def select_venv(self):

        path = (
            QFileDialog.getExistingDirectory(
                self,
                "Selecione o ambiente Python"
            )
        )

        if path:

            self.venv_input.setText(
                path
            )


    def select_controller(self):

        path = (
            QFileDialog.getExistingDirectory(
                self,
                "Selecione a pasta do controller"
            )
        )

        if path:

            self.controller_input.setText(
                path
            )


    def stop_training(self):

        if self.training_process is None:
            return

        if (
            self.training_process.state()
            == QProcess.NotRunning
        ):
            return

        stop_file = (
            PROJECT_ROOT
            / "training"
            / ".stop_training"
        )

        try:

            stop_file.touch()

        except OSError as error:

            self.status_label.setText(
                f"Erro ao solicitar parada: {error}"
            )

            return

        self.status_label.setText(
            "Solicitando parada do treinamento..."
        )

        self.stop_training_button.setEnabled(
            False
        )


    def training_finished(
        self,
        exit_code,
        exit_status
    ):

        stop_file = (
            PROJECT_ROOT
            / "training"
            / ".stop_training"
        )

        if stop_file.exists():

            try:
                stop_file.unlink()
            except OSError:
                pass

        self.status_label.setText(
            f"Treinamento finalizado. "
            f"Código: {exit_code}"
        )

        self.start_training_button.setEnabled(
            True
        )

        self.stop_training_button.setEnabled(
            False
        )

        self.stop_webots_button.setEnabled(
            True
        )

        self.training_process = None


def main():

    app = QApplication(
        sys.argv
    )

    window = SpotManager()

    window.show()

    sys.exit(
        app.exec()
    )

if __name__ == "__main__":
    main()