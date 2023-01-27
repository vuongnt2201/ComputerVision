from collections import deque
from math import radians
from cv2 import blur, contourArea
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math

cap = cv2.VideoCapture('lemon.MOV')
Lower = (25, 80, 30)
Upper = (70, 255, 255)
pts = []
vid = []
out_video_name = 'lemon_detected.mp4'

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
count = 0
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #b = cv2.resize(frame,(648, 1024),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    #b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)

    frame = imutils.resize(frame, width=600)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    blurred = cv2.filter2D(frame, ddepth=-1, kernel=kernel)
    
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, Lower, Upper)
    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.dilate(mask, None, iterations=10)
    #cv2.imshow('mask', mask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    if(len(cnts) > 0):
        c = max(cnts, key = cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if(radius > 30):
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    cv2.imshow('image', frame)
    #cv2.imwrite('mask/frame%d.jpg' % count, mask)
    count += 1
    vid.append(frame) 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

frame = vid[0]
size = list(frame.shape)
del size[2]
size.reverse()

video = cv2.VideoWriter(out_video_name, cv2_fourcc, 24, size)

for i in range(len(vid)):
    video.write(vid[i])

video.release()

cap.release()

# Closes all the frames
cv2.destroyAllWindows()


