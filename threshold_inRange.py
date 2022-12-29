from __future__ import print_function
import cv2 as cv
import argparse
import random as rng
import numpy as np

max_value = 255
max_value_H = 360//2
low_H = 100
high_H = 114

low_S = 85
high_S = 162


low_V = 78
high_V = 255

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


# function called by trackbar, sets the next frame to be read
def getFrame(frame_nr):
    global video
    video.set(cv.CAP_PROP_POS_FRAMES, frame_nr)

#  function called by trackbar, sets the speed of playback
def setSpeed(val):
    global playSpeed
    playSpeed = max(val,1)

## [low]
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
## [low]

## [high]
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
## [high]

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
    
def scale_contour(cnt, scale):
    M = cv.moments(cnt)
    if M['m00'] > 0:
	    cx = int(M['m10']/M['m00'])
	    cy = int(M['m01']/M['m00'])
	    cnt_norm = cnt - [cx, cy]
	    cnt_scaled = cnt_norm * scale
	    cnt_scaled = cnt_scaled + [cx, cy]
	    #cnt_scaled = cnt_scaled.astype(int32)
	    return cnt_scaled

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
parser.add_argument('--file', help='Camera divide number.', type=str)
args = parser.parse_args()

# set wait for each frame, determines playbackspeed
playSpeed = 30




## [cap]
cap = cv.VideoCapture(args.file)

nr_of_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))


## [cap]

## [window]
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
## [window]

# add trackbar
cv.createTrackbar("Frame", window_detection_name, 0,nr_of_frames,getFrame)
cv.createTrackbar("Speed", window_detection_name, playSpeed,100,setSpeed)


## [trackbar]
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
## [trackbar]

while True:
    ## [while]
    ret, frame = cap.read()
    if frame is None:
        break
        
    scale_percent = 40 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)

    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 50))
    frame_threshold = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)
    
    contours, hierarchy = cv.findContours(frame_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    
    
    outl = 15;
    for i in range(len(contours)):
        color = (0, 255, 0)
        cv.rectangle(frame, (int(boundRect[i][0]) - outl , int(boundRect[i][1]) - outl ), \
          (int(boundRect[i][0]+boundRect[i][2] + outl)  , int(boundRect[i][1]+boundRect[i][3]) + outl ) , color, 1)

    #cv.drawContours(frame, contours, -1, (0,255,0), 1)
    ## [while]

    ## [show]
    cv.setTrackbarPos("Frame","Video", int(cap.get(cv.CAP_PROP_POS_FRAMES)))


    # display frame for 'playSpeed' ms, detect key input
    key = cv.waitKey(playSpeed)

    # stop playback when q is pressed
    if key == ord('q'):
        break
    
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    ## [show]

    key = cv.waitKey(10)
    if key == ord('q') or key == 27:
        break
