import numpy as np
import gymnasium as gym

from gymnasium import spaces
from controller import Supervisor


class SpotWebotsEnv(gym.Env):

    def __init__(self):

        super().__init__()

        # =====================================================
        # WEBOTS
        # =====================================================

        self.robot = Supervisor()

        self.time_step = int(
            self.robot.getBasicTimeStep()
        )

        self.dt = self.time_step / 1000.0


        # =====================================================
        # NÓ DO SPOT
        # =====================================================

        self.robot_node = self.robot.getSelf()

        self.translation_field = (
            self.robot_node.getField("translation")
        )

        self.rotation_field = (
            self.robot_node.getField("rotation")
        )

        self.initial_translation = (
            self.translation_field.getSFVec3f()
        )

        self.initial_rotation = (
            self.rotation_field.getSFRotation()
        )


        # =====================================================
        # JUNTAS
        # =====================================================
        # TODO:
        # substituir pelos nomes exatos depois que você mandar
        # os arquivos do Spot.

        self.JOINT_NAMES = [
            "front left shoulder abduction motor",
            "front left shoulder rotation motor",
            "front left elbow motor",

            "front right shoulder abduction motor",
            "front right shoulder rotation motor",
            "front right elbow motor",

            "rear left shoulder abduction motor",
            "rear left shoulder rotation motor",
            "rear left elbow motor",

            "rear right shoulder abduction motor",
            "rear right shoulder rotation motor",
            "rear right elbow motor",
        ]

        self.motors = []
        self.position_sensors = []

        for name in self.JOINT_NAMES:

            motor = self.robot.getDevice(name)

            self.motors.append(motor)

            sensor = motor.getPositionSensor()
            sensor.enable(self.time_step)

            self.position_sensors.append(sensor)


        # =====================================================
        # LIMITES DAS JUNTAS
        # =====================================================

        self.joint_min = np.array([
            motor.getMinPosition()
            for motor in self.motors
        ], dtype=np.float32)

        self.joint_max = np.array([
            motor.getMaxPosition()
            for motor in self.motors
        ], dtype=np.float32)


        # =====================================================
        # POSE NEUTRA
        # =====================================================
        # TODO:
        # vamos pegar a pose correta do controller oficial.
        #
        # Não assuma que zero é necessariamente uma pose em pé
        # válida para o Spot.

        self.neutral_pose = np.array([
            -0.1, 0.0, 0.0,
            0.1, 0.0, 0.0,
            -0.1, 0.0, 0.0,
            0.1, 0.0, 0.0,
        ], dtype=np.float32)

        self.target_positions = (
            self.neutral_pose.copy()
        )


        # =====================================================
        # AÇÕES
        # =====================================================
        # Uma saída contínua para cada junta.

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )


        # =====================================================
        # OBSERVAÇÃO MÍNIMA
        # =====================================================
        #
        # Por enquanto:
        #
        # 12 posições articulares
        # 12 velocidades articulares
        #
        # Total = 24
        #
        # Depois você pode adicionar:
        # - roll/pitch/yaw
        # - velocidade do corpo
        # - altura
        # - contato dos pés
        # etc.

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32
        )


        # =====================================================
        # ESTADO INTERNO
        # =====================================================

        self.last_joint_positions = None

        # Quanto uma ação pode alterar o target em um step.
        self.JOINT_DELTA = 0.03

        self.current_step = 0
        self.MAX_STEPS = 1000


    # =========================================================
    # LEITURA DAS JUNTAS
    # =========================================================

    def _get_joint_positions(self):

        return np.array([
            sensor.getValue()
            for sensor in self.position_sensors
        ], dtype=np.float32)


    def _get_obs(self):

        positions = self._get_joint_positions()

        if self.last_joint_positions is None:

            velocities = np.zeros(
                12,
                dtype=np.float32
            )

        else:

            velocities = (
                positions
                - self.last_joint_positions
            ) / self.dt

        self.last_joint_positions = (
            positions.copy()
        )

        obs = np.concatenate([
            positions,
            velocities
        ])

        return obs.astype(np.float32)


    # =========================================================
    # APLICAÇÃO DA AÇÃO
    # =========================================================

    def _apply_action(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        # Controle incremental:
        #
        # action = -1 -> reduz target
        # action =  0 -> mantém
        # action = +1 -> aumenta target

        self.target_positions += (
            action
            * self.JOINT_DELTA
        )

        self.target_positions = np.clip(
            self.target_positions,
            self.joint_min,
            self.joint_max
        )

        for motor, target in zip(
            self.motors,
            self.target_positions
        ):

            motor.setPosition(
                float(target)
            )


    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        self.current_step = 0


        # -----------------------------------------------------
        # Corpo volta ao spawn
        # -----------------------------------------------------

        self.translation_field.setSFVec3f(
            self.initial_translation
        )

        self.rotation_field.setSFRotation(
            self.initial_rotation
        )

        self.robot_node.resetPhysics()


        # -----------------------------------------------------
        # Juntas voltam à pose neutra
        # -----------------------------------------------------

        self.target_positions = (
            self.neutral_pose.copy()
        )

        for motor, target in zip(
            self.motors,
            self.target_positions
        ):

            motor.setPosition(
                float(target)
            )


        # Deixa a física estabilizar um pouco.

        for _ in range(5):

            self.robot.step(
                self.time_step
            )


        self.last_joint_positions = None

        obs = self._get_obs()

        return obs, {}


    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):

        self.current_step += 1

        self._apply_action(action)

        status = self.robot.step(
            self.time_step
        )

        if status == -1:

            return (
                self._get_obs(),
                0.0,
                True,
                False,
                {}
            )


        obs = self._get_obs()


        # =====================================================
        # REWARD
        # =====================================================
        #
        # DELIBERADAMENTE vazio por enquanto.
        #
        # Essa é uma das partes que você vai construir.

        reward = 0.0


        # =====================================================
        # TERMINAÇÃO
        # =====================================================
        #
        # Depois adicionaremos detecção de queda.

        terminated = False

        truncated = (
            self.current_step
            >= self.MAX_STEPS
        )


        info = {}


        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )


    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        for motor in self.motors:

            motor.setVelocity(0.0)