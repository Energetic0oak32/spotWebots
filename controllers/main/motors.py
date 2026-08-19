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

    def __init__(self, robot, time_step):

        self.robot = robot
        self.time_step = time_step

        self.motors = []
        self.position_sensors = []

        for name in JOINT_NAMES:

            motor = self.robot.getDevice(name)

            self.motors.append(motor)

            sensor = motor.getPositionSensor()

            sensor.enable(self.time_step)

            self.position_sensors.append(sensor)


        self.joint_min = np.array([motor.getMinPosition() for motor in self.motors], dtype=np.float32)


        self.joint_max = np.array([motor.getMaxPosition() for motor in self.motors], dtype=np.float32)



    def set_single_motor_position(self, index, position):
        """
        Teste de documentaçao
        """

        motor = self.motors[index]

        position = np.clip(position, self.joint_min[index], self.joint_max[index])

        motor.setPosition(float(position))

    def set_motor_positions(self, positions):

        positions = np.clip(
            positions,
            self.joint_min,
            self.joint_max
        )

        for motor, position in zip(
            self.motors,
            positions
        ):
            motor.setPosition(float(position))

    def get_single_motor_position(self, index):

        sensor = self.position_sensors[index]

        position = sensor.getValue()

        return position

    def get_motor_positions(self):

        positions = np.array([sensor.getValue() for sensor in self.position_sensors], dtype=np.float32)

        return positions