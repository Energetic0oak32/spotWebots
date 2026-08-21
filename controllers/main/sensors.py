import numpy as np


class SpotSensors:

    def __init__(self, robot, time_step):

        self.robot = robot
        self.time_step = time_step

        self.gyro = self.robot.getDevice("gyro")

        self.gyro.enable(self.time_step)

        self.inertial_unit = self.robot.getDevice("inertial unit")

        self.inertial_unit.enable(self.time_step)

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
            np.sin(roll),
            np.cos(roll),

            np.sin(pitch),
            np.cos(pitch),

            np.sin(yaw),
            np.cos(yaw),
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