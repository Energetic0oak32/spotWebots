import numpy as np


class SpotSensors:

    def __init__(self, robot, time_step):

        self.robot = robot
        self.time_step = time_step

        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.inertial_unit = self.robot.getDevice("inertial unit")
        self.inertial_unit.enable(self.time_step)

        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.time_step)

        self.previous_position = None
        self.dt = self.time_step / 1000.0


    def get_orientation(self):
        """
        Retorna as orientações do robo, Roll, Pitch, Yaw, em  em radianos em um np array.
        """

        orientation = (
            self.inertial_unit.getRollPitchYaw()
        )

        return np.array(
            orientation,
            dtype=np.float32
        )

    def get_orientation_features(self):
        """
        Retorna as orientações do robo, Roll, Pitch, Yaw separados em cos e sin.
        """

        roll, pitch, yaw = (
            self.inertial_unit.getRollPitchYaw()
        )

        features = np.array([
            np.sin(roll), #indice 0
            np.cos(roll),

            np.sin(pitch),
            np.cos(pitch),

            np.sin(yaw),
            np.cos(yaw), #indice 5
        ], dtype=np.float32)

        return features

    def get_angular_velocity(self):
        """
        Retorna velocidade angular em X, Y e Z, em rad/s
        """

        angular_velocity = self.gyro.getValues()

        return np.array(
            angular_velocity,
            dtype=np.float32
        )

    def get_upright(self, orientation_features):
        """
        Retorna uma medida da orientação vertical do robô.

        1  -> aproximadamente em pé
        0  -> aproximadamente de lado
        -1 -> aproximadamente invertido
        """

        cos_roll = orientation_features[1]
        cos_pitch = orientation_features[3]

        upright = cos_roll * cos_pitch

        return upright

    def get_position(self):
        """
        Retorna a posição global X, Y e Z do robô em metros.
        """

        position = self.gps.getValues()

        return np.array(
            position,
            dtype=np.float32
        )

    def get_linear_velocity(self, current_position):
        """
        Retorna os valores de deslocamento no eixo X, Y e Z em m/s
        """

        if self.previous_position is None:

            self.previous_position = current_position.copy()

            return np.zeros(
                3,
                dtype=np.float32
            )

        velocity = (
            current_position
            - self.previous_position
        ) / self.dt

        self.previous_position = current_position.copy()

        return velocity.astype(np.float32)

    def get_local_linear_velocity(self, linear_velocity, orientation_features):
        """
        Retorna a velocidade relativa ao robô, Velocidade relativa ao Spot: [frente, lateral, vertical]
        """

        vx = linear_velocity[0]
        vy = linear_velocity[1]
        vz = linear_velocity[2]

        sin_yaw = orientation_features[4]
        cos_yaw = orientation_features[5]

        forward_velocity = (
            vx * cos_yaw
            + vy * sin_yaw
        )

        lateral_velocity = (
            -vx * sin_yaw
            + vy * cos_yaw
        )

        return np.array([
            forward_velocity,
            lateral_velocity,
            vz
        ], dtype=np.float32)