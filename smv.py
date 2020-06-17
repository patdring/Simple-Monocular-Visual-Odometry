import numpy as np 
from glob import glob
import os
import os.path
import cv2
import math
from time import sleep

def processFrame(frame):
  orb = cv2.ORB_create(nfeatures=25)
  keypoints, descriptors = orb.detectAndCompute(frame, None)
  frame_with_kpts = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  # Display the resulting frame 
  cv2.imshow('Frame',frame_with_kpts)
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
        processFrame(frame)      
   
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

