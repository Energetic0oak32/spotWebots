"""debuging controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
from motors import SpotMotors

# create the Robot instance.
robot = Supervisor()   #instancia o robo supervisor

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

spot_motors = SpotMotors(robot, timestep)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    print(spot_motors.get_motor_positions())
    pass

# Enter here exit cleanup code.
