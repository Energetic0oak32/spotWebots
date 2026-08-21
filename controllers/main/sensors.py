import numpy as np


class SpotSensors:

    def __init__(self, robot, time_step):

        self.robot = robot
        self.time_step = time_step

        self.inertial_unit = self.robot.getDevice(
            "inertial unit"
        )

        self.inertial_unit.enable(
            self.time_step
        )

    def get_orientation(self):
        """
        Retorna as orientações do robo, Roll, Pitch, Yaw em um np array.
        """

        orientation = (
            self.inertial_unit.getRollPitchYaw()
        )

        return np.array(
            orientation,
            dtype=np.float32
        )