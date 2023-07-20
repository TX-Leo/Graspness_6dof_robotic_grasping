#coding=utf8
import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import scipy as sc
from collections import namedtuple
# import utils
import copy
import socket
import select
import struct
import numpy as np
import math
import sys
from real.robotiq_gripper import RobotiqGripper
# from real.realsenseD435 import RealsenseD435
from real.realsenseD415 import Camera
#ros
sys.path.append("/home/randy/catkin_UR5/devel/include/ur_planning")
sys.path.append("/home/randy/catkin_UR5/src/universal_robot/ur_planning")
import rospy
from ur_planning.srv import grasp_pose,grasp_poseRequest,grasp_poseResponse


class UR_Robot:
    def __init__(self, tcp_host_ip="192.168.50.100", tcp_port=30003, workspace_limits=None,
                 is_use_robotiq85=True, is_use_camera=True):
        # Init varibles
        if workspace_limits is None:
            workspace_limits = [[-0.7, 0.7], [-0.7, 0.7], [0.00, 0.6]]
        self.workspace_limits = workspace_limits
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port
        self.is_use_robotiq85 = is_use_robotiq85
        self.is_use_camera = is_use_camera


        # UR5 robot configuration
        # Default joint/tool speed configuration
        self.joint_acc = 1.4  # Safe: 1.4   8
        self.joint_vel = 1.05  # Safe: 1.05  3

        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        # Default tool speed configuration
        self.tool_acc = 0.5  # Safe: 0.5
        self.tool_vel = 0.2  # Safe: 0.2

        # Tool pose tolerance for blocking calls
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

        # robotiq85 gripper configuration
        if(self.is_use_robotiq85):
            # reference https://gitlab.com/sdurobotics/ur_rtde
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.tcp_host_ip, 63352)  # don't change the 63352 port
            self.gripper._reset()
            print("Activating gripper...")
            self.gripper.activate()
            time.sleep(1.5)
        
        # realsense configuration
        self.cam_intrinsics = np.array([615.284,0,309.623,0,614.557,247.967,0,0,1]).reshape(3,3)
        if (self.is_use_camera):
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera(width=1280, height=720)
            self.cam_intrinsics = self.camera.intrinsics  # get camera intrinsics
        # # Load camera pose (from running calibrate.py), intrinsics and depth scale
        self.cam_pose = np.loadtxt('real/cam_pose/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('real/cam_pose/camera_depth_scale.txt', delimiter=' ')
   

        # Default robot home joint configuration (the robot is up to air)
        self.home_joint_config = [-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             -(0 / 360.0) * 2 * np.pi, 0.0]

        # ros
        #rospy.init_node("moveit_control_client")
        # # 创建请求对象
        #self.client = rospy.ServiceProxy("moveit_grasp", grasp_pose)

        # test
        self.testRobot()
    # Test for robot controlmove_and_wait_for_pos
    def testRobot(self):
        try:
            print("Test for robot...")
            # self.grasp([0.3, 0.3, 0.1],[0, -np.pi, 0])
            #self.move_j_p([0.4, 0, 0.4, -np.pi, 0, 0])
            # self.move_j_p([0.3,0.4,0.3,0,-np.pi,0],r=0.01)
            # self.move_j_p([0.3, 0.2, 0.25, 0, -np.pi, 0])
            # time.sleep(3)
            # self.gripper.move_and_wait_for_pos(255)
            # self.log_gripper_info()

            # self.move_j([-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  -(0 / 360.0) * 2 * np.pi, 0.0])
            # self.move_j([(57.04 / 360.0) * 2 * np.pi, (-65.26/ 360.0) * 2 * np.pi,
            #                  (73.52/ 360.0) * 2 * np.pi, (-100.89/ 360.0) * 2 * np.pi,
            #                  (-86.93/ 360.0) * 2 * np.pi, (-0.29/360)*2*np.pi])
            # self.open_gripper()
            # self.move_j([(57.03 / 360.0) * 2 * np.pi, (-56.67 / 360.0) * 2 * np.pi,
            #                   (88.72 / 360.0) * 2 * np.pi, (-124.68 / 360.0) * 2 * np.pi,
            #                   (-86.96/ 360.0) * 2 * np.pi, (-0.3/ 360) * 2 * np.pi])
            # self.close_gripper()
            # self.move_j([(57.04 / 360.0) * 2 * np.pi, (-65.26 / 360.0) * 2 * np.pi,
            #                   (73.52 / 360.0) * 2 * np.pi, (-100.89 / 360.0) * 2 * np.pi,
            #                   (-86.93 / 360.0) * 2 * np.pi, (-0.29 / 360) * 2 * np.pi])
            # self.move_j([-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  -(0 / 360.0) * 2 * np.pi, 0.0])
            # self.move_j_p([0.3,0,0.3,np.pi/2,0,0],0.5,0.5)
            # for i in range(10):
            #     self.move_j_p([0.3, 0, 0.3, np.pi, 0, i*0.1], 0.5, 0.5)
            #     time.sleep(1)
            # self.move_j_p([0.3, 0, 0.3, -np.pi, 0, 0],0.5,0.5)
            # self.move_p([0.3, 0.3, 0.3, -np.pi, 0, 0],0.5,0.5)
            # self.move_l([0.2, 0.2, 0.3, -np.pi, 0, 0],0.5,0.5)
            # self.plane_grasp([0.3, 0.3, 0.1])
            # self.plane_push([0.3, 0.3, 0.1])
            rot_z = np.array([[math.cos(3.1415926), -math.sin(3.1415926), 0],
                          [math.sin(3.1415926), math.cos(3.1415926), 0],
                          [0, 0, 1]])
            print(self.cam_pose[0:3][0:3])
            R =np.dot(rot_z,self.cam_pose[0:3][0:3])
           
            print(R)
            rpy = self.R2rpy(R)
            print(rpy) #[-2.85184905  0.06696862 -1.36472784]
            p =np.dot(rot_z,[0.554368,0.596036,0.855069])
            print(p)
        except:
            print("Test fail! ")
    
    # joint control
    '''
    input:joint_configuration = joint angle
    '''
    def move_j(self, joint_configuration,k_acc=1,k_vel=1,t=0,r=0):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        # command: movej([joint_configuration],a,v,t,r)\n
        tcp_command = "movej([%f" % joint_configuration[0]  #"movej([]),a=,v=,\n"
        for joint_idx in range(1,6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f,t=%f,r=%f)\n" % (k_acc*self.joint_acc, k_vel*self.joint_vel,t,r)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            state_data = self.tcp_socket.recv(1500)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)
        self.tcp_socket.close()
        return True
    # joint control
    '''
    move_j_p(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0)
    input:tool_configuration=[x y z r p y]
    其中x y z为三个轴的目标位置坐标，单位为米
    r p y ，单位为弧度
    '''
    def move_j_p(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movej_p([{tool_configuration}])")
        # tool_configuration[3:] = self.rpy2rotating_vector([tool_configuration[3],tool_configuration[4],tool_configuration[5]])
        # command: movej([joint_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command +=" array = rpy2rotvec([%f,%f,%f])\n" %(tool_configuration[3],tool_configuration[4],tool_configuration[5])
        tcp_command += "movej(get_inverse_kin(p[%f,%f,%f,array[0],array[1],array[2]]),a=%f,v=%f,t=%f,r=%f)\n" % (tool_configuration[0],
            tool_configuration[1],tool_configuration[2],k_acc * self.joint_acc, k_vel * self.joint_vel,t,r ) # "movej([]),a=,v=,\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        start = time.time()
        delaytime = 10
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while (not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]) and (time.time()-start<delaytime)):
            state_data = self.tcp_socket.recv(1500)
            # print(f"tool_position_error{actual_tool_positions - tool_configuration}")
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        if (time.time() -start)>=delaytime:
            print(f'{time.time()},start:{start}')
            return False
        self.tcp_socket.close()
        time.sleep(0.5)
        return True


    # Usually, We don't use move_p
    # move_p is mean that the robot keep the same speed moving
    def move_p(self, tool_configuration,k_acc=1,k_vel=1,r=0):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movep([{tool_configuration}])")
        # command: movep([tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " array = rpy2rotvec([%f,%f,%f])\n" % (
        tool_configuration[3], tool_configuration[4], tool_configuration[5])
        tcp_command += "movep(p[%f,%f,%f,array[0],array[1],array[2]],a=%f,v=%f,r=%f)\n" % (
        tool_configuration[0],tool_configuration[1], tool_configuration[2],
        k_acc * self.joint_acc,k_vel * self.joint_vel,r)  # "movep([]),a=,v=,\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in
                       range(3)]):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        time.sleep(1.5)
        self.tcp_socket.close()


    # move_l is mean that the robot keep running in a straight line
    def move_l(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0):
        print(f"movel([{tool_configuration}])")
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        # command: movel([tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " array = rpy2rotvec([%f,%f,%f])\n" % (
            tool_configuration[3], tool_configuration[4], tool_configuration[5])
        tcp_command += "movel(p[%f,%f,%f,array[0],array[1],array[2]],a=%f,v=%f,t=%f,r=%f)\n" % (
            tool_configuration[0], tool_configuration[1], tool_configuration[2],
            k_acc * self.joint_acc, k_vel * self.joint_vel,t,r)  # "movel([]),a=,v=,\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        start = time.time()
        delaytime = 10
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while (not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]) and time.time()-start<delaytime ):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        if (time.time() - start) >= delaytime:
            return False
        time.sleep(0.5)
        self.tcp_socket.close()
        return True

    # Usually, We don't use move_c
    # move_c is mean that the robot move circle
    # mode 0: Unconstrained mode. Interpolate orientation from current pose to target pose (pose_to)
    #      1: Fixed mode. Keep orientation constant relative to the tangent of the circular arc (starting from current pose)
    def move_c(self,pose_via,tool_configuration,k_acc=1,k_vel=1,r=0,mode=0):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movec([{pose_via},{tool_configuration}])")
        # command: movec([pose_via,tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " via_pose = rpy2rotvec([%f,%f,%f])\n" % (
        pose_via[3],pose_via[4] ,pose_via[5] )
        tcp_command += " tool_pose = rpy2rotvec([%f,%f,%f])\n" % (
        tool_configuration[3], tool_configuration[4], tool_configuration[5])
        tcp_command = f" movec([{pose_via[0]},{pose_via[1]},{pose_via[2]},via_pose[0],via_pose[1],via_pose[2]], \
                [{tool_configuration[0]},{tool_configuration[1]},{tool_configuration[2]},tool_pose[0],tool_pose[1],tool_pose[2]], \
                a={k_acc * self.tool_acc},v={k_vel * self.tool_vel},r={r})\n"
        tcp_command += "end\n"

        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        self.tcp_socket.close()
        time.sleep(1.5)

    def go_home(self):
        self.move_j(self.home_joint_config)

    def restartReal(self):
        self.go_home()
        # robotiq85 gripper configuration
        if (self.is_use_robotiq85):
            # reference https://gitlab.com/sdurobotics/ur_rtde
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.tcp_host_ip, 63352)  # don't change the 63352 port
            self.gripper._reset()
            print("Activating gripper...")
            self.gripper.activate()
            time.sleep(1.5)

        # realsense configuration
        if (self.is_use_camera):
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera()
            # self.cam_intrinsics = self.camera.intrinsics  # get camera intrinsics
            self.cam_intrinsics = self.camera.color_intr
            # # Load camera pose (from running calibrate.py), intrinsics and depth scale
            self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
            self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

    # get robot current state and information
    def get_state(self):
        self.tcp_cket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(1500)
        self.tcp_socket.close()
        return state_data
    
    # get robot current joint angles and cartesian pose
    def parse_tcp_state_data(self, data, subpasckage):
        dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
               'I target': '6d',
               'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
               'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
               'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
               'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
               'Tool Accelerometer values': '3d',
               'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd',
               'softwareOnly2': 'd',
               'V main': 'd',
               'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
               'Elbow position': 'd', 'Elbow velocity': '3d'}
        ii = range(len(dic))
        for key, i in zip(dic, ii):
            fmtsize = struct.calcsize(dic[key])
            data1, data = data[0:fmtsize], data[fmtsize:]
            fmt = "!" + dic[key]
            dic[key] = dic[key], struct.unpack(fmt, data1)

        if subpasckage == 'joint_data':  # get joint data
            q_actual_tuple = dic["q actual"]
            joint_data= np.array(q_actual_tuple[1])
            return joint_data
        elif subpasckage == 'cartesian_info':
            Tool_vector_actual = dic["Tool vector actual"]  # get x y z rx ry rz
            cartesian_info = np.array(Tool_vector_actual[1])
            return cartesian_info

    def rpy2rotating_vector(self,rpy):
        # rpy to R
        R = self.rpy2R(rpy)
        # R to rotating_vector
        return self.R2rotating_vector(R)

    def rpy2R(self,rpy): # [r,p,y] 单位rad
        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                          [0, math.sin(rpy[0]), math.cos(rpy[0])]])
        rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                          [0, 1, 0],
                          [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
        rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                          [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                          [0, 0, 1]])
        R = np.dot(rot_z, np.dot(rot_y, rot_x))
        return R

    def R2rotating_vector(self,R):
        theta = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
        print(f"theta:{theta}")
        rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
        ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
        rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
        return np.array([rx, ry, rz]) * theta

    def R2rpy(self,R):
    # assert (isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])


    ## robotiq85 gripper
    # get gripper position [0-255]  open:0 ,close:255
    def get_current_tool_pos(self):
        return self.gripper.get_current_position()       

    def log_gripper_info(self):
        print(f"Pos: {str(self.gripper.get_current_position())}")

    def close_gripper(self,speed=255,force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(255, speed, force)
        print("gripper had closed!")
        time.sleep(1.2)
        self.log_gripper_info()

    def open_gripper(self,speed=255,force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(0, speed, force)
        print("gripper had opened!")
        time.sleep(1.2)
        self.log_gripper_info()
    

    ## get camera data 
    def get_camera_data(self):
        color_img, depth_img = self.camera.get_data()
        return color_img, depth_img

    # Note: must be preceded by close_gripper()
    def check_grasp(self):
        # if the robot grasp object ,then the gripper is not open
        return self.get_current_tool_pos()>220

    def plane_grasp(self, position, yaw=0, open_size=0.65, k_acc=0.8,k_vel=0.8,speed=255, force=125):

        # 判定抓取的位置是否处于工作空间
        rpy = [-np.pi, 0, 1.57-yaw]
        for i in range(3):
            position[i] = min(max(position[i],self.workspace_limits[i][0]),self.workspace_limits[i][1])
        # 判定抓取的角度RPY是否在规定范围内 [-pi,pi]
        for i in range(3):
            if rpy[i] > np.pi:
                rpy[i] -= 2*np.pi
            elif rpy[i] < -np.pi:
                rpy[i] += 2*np.pi
        print('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)' \
              % (position[0], position[1], position[2],rpy[0],rpy[1],rpy[2]))

        # pre work
        grasp_home = [0.4,0,0.4,-np.pi,0,0]  # you can change me
        self.move_j_p(grasp_home,k_acc,k_vel)
        open_pos = int(-258*open_size +230)  # open size:0~0.85cm --> open pos:230~10
        self.gripper.move_and_wait_for_pos(open_pos, speed, force)
        print("gripper open size:")
        self.log_gripper_info()

        # Firstly, achieve pre-grasp position
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.1  # z axis
        # print(pre_position)
        self.move_j_p(pre_position + rpy,k_acc,k_vel)

        # Second，achieve grasp position
        self.move_l(position+rpy,0.6*k_acc,0.6*k_vel)
        self.close_gripper(speed,force)
        self.move_l(pre_position + rpy, 0.6*k_acc,0.6*k_vel)
        if(self.check_grasp()):
            print("Check grasp fail! ")
            self.move_j_p(grasp_home)
            return False
        # Third,put the object into box
        box_position = [0.63,0,0.25,-np.pi,0,0]  # you can change me!
        self.move_j_p(box_position,k_acc,k_vel)
        box_position[2] = 0.1  # down to the 10cm
        self.move_j_p(box_position, k_acc, k_vel)
        self.open_gripper(speed,force)
        box_position[2] = 0.25
        self.move_j_p(box_position, k_acc, k_vel)
        self.move_j_p(grasp_home)
        print("grasp success!")
        return True

    def plane_push(self, position, move_orientation=0, length=0.1):
        for i in range(2):
            position[i] = min(max(position[i],self.workspace_limits[i][0]+0.1),self.workspace_limits[i][1]-0.1)
        position[2] = min(max(position[2],self.workspace_limits[2][0]),self.workspace_limits[2][1])
        print('Executing: push at (%f, %f, %f) and the orientation is %f' % (position[0], position[1], position[2],move_orientation))

        push_home = [0.4, 0, 0.4, -np.pi, 0, 0]
        self.move_j_p(push_home,k_acc=1, k_vel=1)  # pre push position(push home)
        # self.close_gripper()

        self.move_j_p([position[0],position[1],position[2]+0.1,-np.pi,0,0],k_acc=1,k_vel=1)
        self.move_j_p([position[0], position[1], position[2], -np.pi, 0, 0], k_acc=0.6, k_vel=0.6)

        # compute the destination pos
        destination_pos = [position[0] + length * math.cos(move_orientation),position[1] + length * math.sin(move_orientation),position[2]]
        self.move_l(destination_pos+[-np.pi, 0, 0], k_acc=0.5, k_vel=0.5)
        self.move_j_p([destination_pos[0],destination_pos[1],destination_pos[2]+0.1,-np.pi,0,0],k_acc=0.6, k_vel=0.6)

        # go back push-home
        self.move_j_p(push_home, k_acc=1, k_vel=1)

    def grasp(self, position, rpy=None, open_size=0.85, k_acc=1, k_vel=1, speed=255, force=200):

        # 判定抓取的位置是否处于工作空间
        if rpy is None:
            rpy = [-np.pi, 0, 0]
        for i in range(3):
            position[i] = min(max(position[i], self.workspace_limits[i][0]), self.workspace_limits[i][1])
        # 判定抓取的角度RPY是否在规定范围内 [0.5*pi,1.5*pi]
        for i in range(3):
            if rpy[i] > np.pi:
                rpy[i] -= 2 * np.pi
            elif rpy[i] < -np.pi:
                rpy[i] += 2 * np.pi
        print('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)' \
              % (position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]))

        # pre work
        grasp_home = [0.4, 0, 0.4, -np.pi, 0, 0]  # you can change me
        self.move_j_p(grasp_home, k_acc, k_vel)
        # open_pos = int(-300 * open_size + 255)  # open size:0~0.85cm --> open pos:255~0
        open_pos = int(-258 * open_size + 230)  # open size:0~0.85cm --> open pos:230~10
        self.gripper.move_and_wait_for_pos(open_pos, speed, force)
        self.log_gripper_info()

        # Firstly, achieve pre-grasp position
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.1  # z axis
        print(pre_position)
        no_ik_result = self.move_j_p(pre_position + rpy, k_acc, k_vel)
        if not no_ik_result:
            print("no ik reutlt!")
            return False
        # Second，achieve grasp position
        no_ik_result = self.move_l(position + rpy, 0.6 * k_acc, 0.6 * k_vel)
        if not no_ik_result:
            print("no ik reutlt!")
            self.move_j_p(grasp_home, k_acc, k_vel)
            return False
        self.close_gripper(speed, force)
        self.move_j_p(pre_position + rpy, 0.6 * k_acc, 0.6 * k_vel)
        if (self.check_grasp()):
            print("Check grasp fail! ")
            self.move_j_p(grasp_home)
            return False
        # Third,put the object into box
        #self.move_j_p([0.3, 0.4, 0.3, -np.pi, 0, 0], k_acc, k_vel) # zhong jian dian
        box_position = [0.53, 0, 0.25, -np.pi, 0, 0]  # you can change me!
        self.move_j_p(box_position, k_acc, k_vel)
        box_position[2] = 0.15  # down to the 10cm
        self.move_j_p(box_position, k_acc, k_vel)
        self.open_gripper(speed, force)
        box_position[2] = 0.25
        self.move_j_p(box_position, k_acc, k_vel)
        self.move_j_p(grasp_home)
        print("grasp success!")
        return True


    def grasp_ros(self, position, rpy=None, open_size=0.85, k_acc=0.8, k_vel=0.8, speed=255, force=125):

        # 判定抓取的位置是否处于工作空间
        if rpy is None:
            rpy = [-np.pi, 0, 0]
        for i in range(3):
            position[i] = min(max(position[i], self.workspace_limits[i][0]), self.workspace_limits[i][1])
        # 判定抓取的角度RPY是否在规定范围内 [0.5*pi,1.5*pi]
        for i in range(3):
            if rpy[i] > np.pi:
                rpy[i] -= 2 * np.pi
            elif rpy[i] < -np.pi:
                rpy[i] += 2 * np.pi
        print('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)' \
              % (position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]))

        rospy.wait_for_service("moveit_grasp")
        
        # pre work
        grasp_home = [0.4, 0, 0.4, -np.pi, 0, 0]  # you can change me
        self.move_j_p(grasp_home, k_acc, k_vel)
        # open_pos = int(-300 * open_size + 255)  # open size:0~0.85cm --> open pos:255~0
        open_pos = int(-258 * open_size + 230)  # open size:0~0.85cm --> open pos:230~10
        self.gripper.move_and_wait_for_pos(open_pos, speed, force)
        self.log_gripper_info()
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.15  # z axis
        # Firstly, achieve pre-grasp position
        req =grasp_poseRequest()
        req.grasppose_x,req.grasppose_y,req.grasppose_z=position[0], position[1], position[2]+0.1
        req.grasppose_R, req.grasppose_P, req.grasppose_Y = rpy[0], rpy[1], rpy[2]
        result = self.client.call(req)
        # Second，achieve grasp position
        req =grasp_poseRequest()
        req.grasppose_x,req.grasppose_y,req.grasppose_z=position[0], position[1], position[2]
        req.grasppose_R, req.grasppose_P, req.grasppose_Y = rpy[0], rpy[1], rpy[2]
        result = self.client.call(req)

        self.close_gripper(speed, force)
        no_ik_result =self.move_j_p(pre_position + rpy, 0.6 * k_acc, 0.6 * k_vel)
        if not no_ik_result:
            print("no ik reutlt!")
            self.move_j_p(grasp_home, k_acc, k_vel)
            return False
        if (self.check_grasp()):
            print("Check grasp fail! ")
            self.move_j_p(grasp_home)
            return False
        # Third,put the object into box
        #self.move_j_p([0.3, 0.4, 0.3, -np.pi, 0, 0], k_acc, k_vel) # zhong jian dian
        box_position = [0.53, 0, 0.35, -np.pi, 0, 0]  # you can change me!
        self.move_j_p(box_position, k_acc, k_vel)
        box_position[2] = 0.15  # down to the 10cm
        self.move_j_p(box_position, k_acc, k_vel)
        self.open_gripper(speed, force)
        box_position[2] = 0.25
        self.move_j_p(box_position, k_acc, k_vel)
        self.move_j_p(grasp_home)
        print("grasp success!")
        return True


        return result.success


if __name__ =="__main__":
    ur_robot = UR_Robot(is_use_robotiq85=False,is_use_camera=False)

