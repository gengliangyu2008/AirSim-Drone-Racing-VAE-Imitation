from __future__ import print_function

import msgpackrpc  # install as admin: pip install msgpack-rpc-python
import numpy as np  # pip install numpy
import msgpack
import time
import math
import logging
import socket
import threading
import time

from airsimdroneracingvae import DrivetrainType, YawMode, RCData, MultirotorState, ImageResponse

from stats import Stats
#from tello import ImageResponse, CollisionInfo
from tello import Tello
from tello_control_ui import TelloUI
# from .TelloClient import *

class VehicleClient:
    def __init__(self, ip="", port=8889):

        self.MAX_TIME_OUT = 15.0

        '''
        self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port),
                                        timeout=self.MAX_TIME_OUT,
                                        pack_encoding='utf-8',
                                        unpack_encoding='utf-8')
        '''

        self.tello = Tello()
        '''
        self.local_ip = ''
        self.local_port = 8889
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket.bind((self.local_ip, self.local_port))

        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        self.tello_ip = '192.168.10.1'
        self.tello_port = 8889
        self.tello_adderss = (self.tello_ip, self.tello_port)
        self.log = []
        '''


# -----------------------------------  Multirotor APIs ---------------------------------------------
class MultirotorClient:

    def __init__(self):
        self.tello = Tello()
        #drone = tello.Tello('', 8889)
        self.vplayer = TelloUI(self.tello, "./img/")
        # start the Tkinter mainloop
        self.vplayer.root.mainloop()

    def enableApiControl(self, is_enabled, vehicle_name=''):
        self.tello.send_command("command")
        # return self.client.call('enableApiControl', is_enabled, vehicle_name)

    # camera control, simGetImage returns compressed png in array of bytes
    # image_type uses one of the ImageType members
    def simGetImages(self, requests, vehicle_name=''):
        # responses_raw = self.client.call('simGetImages', requests, vehicle_name)
        responses_raw = self.client.call('simGetImages', requests, vehicle_name)
        return [ImageResponse.from_msgpack(response_raw) for response_raw in responses_raw]

    def telloGetImages(self):
        # responses_raw = self.client.call('simGetImages', requests, vehicle_name)
        responses_raw = self.tello.read()
        return [ImageResponse.from_msgpack(response_raw) for response_raw in responses_raw]

    def takeoffAsync(self, timeout_sec=20, vehicle_name=''):
        return self.client.call_async('takeoff', timeout_sec, vehicle_name)

    def landAsync(self, timeout_sec=60, vehicle_name=''):
        return self.client.call_async('land', timeout_sec, vehicle_name)

    def goHomeAsync(self, timeout_sec=3e+38, vehicle_name=''):
        return self.client.call_async('goHome', timeout_sec, vehicle_name)

    # APIs for control
    def moveByAngleZAsync(self, pitch, roll, z, yaw, duration, vehicle_name=''):
        return self.client.call_async('moveByAngleZ', pitch, roll, z, yaw, duration, vehicle_name)

    def moveByAngleThrottleAsync(self, pitch, roll, throttle, yaw_rate, duration, vehicle_name=''):
        return self.client.call_async('moveByAngleThrottle', pitch, roll, throttle, yaw_rate, duration, vehicle_name)

    def moveByVelocityAsync(self, a, b, c, d):

        # rc a b c d
        return self.tello.send_command("rc " + a + " " + b + " " + c + " " +d)
        # nt.call_async('moveByVelocity', vx, vy, vz, duration, drivetrain, yaw_mode, vehicle_name)

    def moveByVelocityAsync(self, vx, vy, vz, duration, drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode=YawMode(), vehicle_name=''):
        return self.client.call_async('moveByVelocity', vx, vy, vz, duration, drivetrain, yaw_mode, vehicle_name)

    def moveByVelocityZAsync(self, vx, vy, z, duration, drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                             yaw_mode=YawMode(), vehicle_name=''):
        return self.client.call_async('moveByVelocityZ', vx, vy, z, duration, drivetrain, yaw_mode, vehicle_name)

    def moveOnPathAsync(self, path, velocity, timeout_sec=3e+38, drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=YawMode(),
                        lookahead=-1, adaptive_lookahead=1, vehicle_name=''):
        return self.client.call_async('moveOnPath', path, velocity, timeout_sec, drivetrain, yaw_mode, lookahead,
                                      adaptive_lookahead, vehicle_name)

    def moveOnSplineAsync(self, path, vel_max, acc_max, add_curr_odom_position_constraint=True,
                          add_curr_odom_velocity_constraint=False, viz_traj=True, vehicle_name=''):
        return self.client.call_async('moveOnSpline', path, add_curr_odom_position_constraint,
                                      add_curr_odom_velocity_constraint, vel_max, acc_max, viz_traj, vehicle_name)

    def moveOnSplineVelConstraintsAsync(self, path, velocities, vel_max, acc_max,
                                        add_curr_odom_position_constraint=True, add_curr_odom_velocity_constraint=False,
                                        viz_traj=True, vehicle_name=''):
        return self.client.call_async('moveOnSplineVelConstraints', path, velocities, add_curr_odom_position_constraint,
                                      add_curr_odom_velocity_constraint, vel_max, acc_max, viz_traj, vehicle_name)

    def setTrajectoryTrackerGains(self, gains, vehicle_name=''):
        self.client.call('setTrajectoryTrackerGains', gains, vehicle_name)

    def moveToPositionAsync(self, x, y, z, velocity, timeout_sec=3e+38, drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode=YawMode(),
                            lookahead=-1, adaptive_lookahead=1, vehicle_name=''):
        return self.client.call_async('moveToPosition', x, y, z, velocity, timeout_sec, drivetrain, yaw_mode, lookahead,
                                      adaptive_lookahead, vehicle_name)

    def moveToZAsync(self, z, velocity, timeout_sec=3e+38, yaw_mode=YawMode(), lookahead=-1, adaptive_lookahead=1,
                     vehicle_name=''):
        return self.client.call_async('moveToZ', z, velocity, timeout_sec, yaw_mode, lookahead, adaptive_lookahead,
                                      vehicle_name)

    def moveByManualAsync(self, vx_max, vy_max, z_min, duration, drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                          yaw_mode=YawMode(), vehicle_name=''):
        """Read current RC state and use it to control the vehicles.

        Parameters sets up the constraints on velocity and minimum altitude while flying. If RC state is detected to violate these constraints
        then that RC state would be ignored.

        :param vx_max: max velocity allowed in x direction
        :param vy_max: max velocity allowed in y direction
        :param vz_max: max velocity allowed in z direction
        :param z_min: min z allowed for vehicle position
        :param duration: after this duration vehicle would switch back to non-manual mode
        :param drivetrain: when ForwardOnly, vehicle rotates itself so that its front is always facing the direction of travel. If MaxDegreeOfFreedom then it doesn't do that (crab-like movement)
        :param yaw_mode: Specifies if vehicle should face at given angle (is_rate=False) or should be rotating around its axis at given rate (is_rate=True)
        """
        return self.client.call_async('moveByManual', vx_max, vy_max, z_min, duration, drivetrain, yaw_mode,
                                      vehicle_name)

    def rotateToYawAsync(self, yaw, timeout_sec=3e+38, margin=5, vehicle_name=''):
        return self.client.call_async('rotateToYaw', yaw, timeout_sec, margin, vehicle_name)

    def rotateByYawRateAsync(self, yaw_rate, duration, vehicle_name=''):
        return self.client.call_async('rotateByYawRate', yaw_rate, duration, vehicle_name)

    def hoverAsync(self, vehicle_name=''):
        return self.client.call_async('hover', vehicle_name)

    def moveByRC(self, rcdata=RCData(), vehicle_name=''):
        return self.client.call('moveByRC', rcdata, vehicle_name)

    def plot_tf(self, pose_list, duration=10.0, vehicle_name=''):
        self.client.call('plot_tf', pose_list, duration, vehicle_name)

        # query vehicle state

    def getMultirotorState(self, vehicle_name=''):
        return MultirotorState.from_msgpack(self.client.call('getMultirotorState', vehicle_name))

    getMultirotorState.__annotations__ = {'return': MultirotorState}
