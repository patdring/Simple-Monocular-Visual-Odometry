import numpy as np 
from glob import glob
import os
import os.path
import cv2
import math
from time import sleep

STAGE_FIRST_FRAME = 0
STAGE_DEFAULT_FRAME = 1
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
 
  if not singular :
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], sy)
    z = math.atan2(R[1,0], R[0,0])
  else :
    x = math.atan2(-R[1,2], R[1,1])
    y = math.atan2(-R[2,0], sy)
    z = 0

  return np.array([x, y, z])

class Camera:
  def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
    self.width = width
    self.height = height
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.distortion = (abs(k1) > 0.0000001)
    self.d = [k1, k2, p1, p2, k3]

class SMV:
  def __init__(self, cam):
    self.cam = cam
    self.focal = cam.fx
    self.pp = (cam.cx, cam.cy)
    self.orb = cv2.ORB_create(nfeatures=25)
    self.frame_stage = STAGE_FIRST_FRAME
    self.frame_ref = None
    self.keypoints_ref = None 
    self.descriptors_ref = None
    self.keypoints_cur = None
    self.descriptors_cur = None
  
  def processFirstFrame(self, frame):
    self.keypoints_ref, self.descriptors_ref = self.orb.detectAndCompute(frame, None)
    self.frame_ref = frame

    #sleep(1)

  def processFrame(self, frame): 
    self.keypoints_cur, self.descriptors_cur = self.orb.detectAndCompute(frame, None)
    frame_with_kpts_ref = cv2.drawKeypoints(self.frame_ref, self.keypoints_ref, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Frame Ref.',frame_with_kpts_ref)
    frame_with_kpts_cur = cv2.drawKeypoints(frame, self.keypoints_cur, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Frame Cur.',frame_with_kpts_cur)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(self.descriptors_ref, self.descriptors_cur)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([self.keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([self.keypoints_cur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(dst_pts, src_pts, focal=1.0, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=10.0)   
    _, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, focal=1.0, pp=self.pp)

    t = t + R.dot(t)
    R = R.dot(R)

    x, y, z = t[0], t[1], t[2]
    angles = rotationMatrixToEulerAngles(R)
    n = angles[0]
    e = angles[1]
    d = angles[2]
    
    draw_x, draw_y = int(x)+250, int(z*-1)+250
    print(y)
    #cv2.circle(traj, (draw_x,draw_y), 1, (0,0,255), 1)
    #cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    #text = "Coordinates: x=%2fm y=%2fm z=%2fm '\n' n=%2fm e==%2fm d=%2fm "%(x,y,z,n,e,d)
    #cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    #cv2.imshow('Trajectory', traj)

    self.keypoints_ref = self.keypoints_cur 
    self.descriptors_ref = self.descriptors_cur
    frame_ref = frame
    #sleep(1)

cap = cv2.VideoCapture('SampleVideos/harder_challenge_video.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.
fps = 0   
if int(major_ver)  < 3 :
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
else :
  fps = cap.get(cv2.CAP_PROP_FPS)

print(fps)

ring_buffer_size = 1
i = 0

cam = Camera(1280.0, 720.0, 458.654, 457.296, 640.0, 360.0)
smv = SMV(cam)
# Read until video is completed
while(cap.isOpened()):
  # Capture not every frame
  i = (i + 1) % ring_buffer_size
  if i < ring_buffer_size-1:
    continue

  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame is not None:
      if smv.frame_stage == STAGE_FIRST_FRAME:
        smv.processFirstFrame(frame)
        smv.frame_stage = STAGE_DEFAULT_FRAME
      else:
        smv.processFrame(frame)    
   
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

