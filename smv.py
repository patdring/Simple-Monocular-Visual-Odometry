import numpy as np 
from glob import glob
import os
import os.path
import cv2
import math
from time import sleep

STAGE_FIRST_FRAME = 0
STAGE_DEFAULT_FRAME = 1

class SMV:
  def __init__(self):
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
    sleep(1)

  def processFrame(self, frame): 
    self.keypoints_cur, self.descriptors_cur = self.orb.detectAndCompute(frame, None)
    
    frame_with_kpts_ref = cv2.drawKeypoints(self.frame_ref, self.keypoints_ref, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Frame Ref.',frame_with_kpts_ref)

    frame_with_kpts_cur = cv2.drawKeypoints(frame, self.keypoints_cur, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Frame Cur.',frame_with_kpts_cur)
    self.keypoints_ref = self.keypoints_cur 
    self.descriptors_ref = self.descriptors_cur
    frame_ref = frame
    sleep(1)

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

ring_buffer_size = fps
i = 0

smv = SMV()
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

