import numpy as np 
from glob import glob
import os
import os.path
import cv2
import math
from time import sleep

from visual_odo import PinholeCamera, VisualOdometry

#cam = PinholeCamera(640.0, 480.0, 458.654, 457.296, 367.215, 248.375)
cam = PinholeCamera(1280.0, 720.0, 458.654, 457.296, 640.0, 360.0)
vo = VisualOdometry(cam)

traj = np.zeros((1000,1000,3), dtype=np.uint8)

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R)) 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

cap = cv2.VideoCapture('SampleVideos/harder_challenge_video.mp4')

if not (cap.isOpened()):
        print('Could not open video device')

for img_id in range(4540):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if img is not None:
            vo.update(img, img_id)

        cur_t = vo.cur_t
        cur_R = vo.cur_R
        anlges = np.array([0,0,0])
        n=0.
        e=0.
        d=0.
        if(img_id > 2):
	        x, y, z = cur_t[0], cur_t[1], cur_t[2]
	        #get angles from rotation matrix
	        angles = rotationMatrixToEulerAngles(cur_R)
	        n = angles[0]
	        e = angles[1]
	        d = anlges[2]
        else:
	        x, y, z = 0., 0., 0.
        draw_x, draw_y = int(x)+650, int(z*-1)+650

        cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm '\n' n=%2fm e==%2fm d=%2fm "%(x,y,z,n,e,d)
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        cv2.imshow('Camera view', img)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)

cv2.imwrite('trajectory.png', traj)
