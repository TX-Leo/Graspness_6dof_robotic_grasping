import numpy as np
import math
import cv2
print('Saving...')
# np.savetxt('camera_depth_scale.txt',[1,2], delimiter=' ')


def rpy2R(rpy):  # [r,p,y] 单位rad
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


def R2rotating_vector(R):
    theta = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
    print(f"theta:{theta}")
    rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
    return np.array([rx, ry, rz])

if __name__ =="__main__":
    R =  rpy2R([3.14,1.57,0])
    print(R)
    v1 = R2rotating_vector(R)
    v2 = cv2.Rodrigues(R)

    print(v1)
    print(v2[0])