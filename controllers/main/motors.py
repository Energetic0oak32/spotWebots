import numpy as np

JOINT_NAMES = [
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


class SpotMotors:

    def __init__(self, robot):

        self.robot = robot

        self.motors = []

        for name in JOINT_NAMES:

            motor = self.robot.getDevice(name)

            self.motors.append(motor)


        self.joint_min = np.array([motor.getMinPosition() for motor in self.motors], dtype=np.float32)


        self.joint_max = np.array([motor.getMaxPosition() for motor in self.motors], dtype=np.float32)


    def set_motor_position(self, index, position):

        motor = self.motors[index]

        position = np.clip(position, self.joint_min[index], self.joint_max[index])

        motor.setPosition(float(position))