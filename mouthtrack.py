from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2
import pyautogui

MOUTH_AR_THRESH = 0.79
SCALE_FACTOR = 3.0
def mouth_aspect_ratio(mouth):
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55
	return (A + B) / (2.0 * C)

EYE_AR_THRESH = 0.2
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5]) # 44, 48; one top/bottom pair
	B = dist.euclidean(eye[2], eye[4]) # 53, 57; other top-bottom pair
	C = dist.euclidean(eye[0], eye[3]) # 49, 55; left-right extremes
	return (A + B) / (2.0 * C)

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68) # mouth
(eStart, eEnd) = (36, 48) # eyes
nose_i = 30 # nose

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=1).start() #NOTE- src=0 for webcam, 1 for ext camera
time.sleep(1.0)

frame_width = 640
frame_height = 360

time.sleep(1.0)
prev_x, prev_y = (0,0)
first_loop = True
mouth_open = False
# loop over frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth = shape[mStart:mEnd]
        eyes = shape[36:48]
        left_eye = shape[42:48]
        nose = shape[nose_i]

        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)
        cv2.circle(frame, nose, 2, (0, 0, 255), -1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, ":O", (30,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

        if first_loop:
            prev_x, prev_y = nose
            first_loop = False
        delta_x, delta_y = nose[0] - prev_x, nose[1] - prev_y
        prev_x, prev_y = nose[0], nose[1]

        # compute the convex hull for the mouth, then
        # visualize the mouth
        mouthHull = cv2.convexHull(mouth)

        cv2.circle(frame, nose, 2, (0, 0, 255), -1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        lear = eye_aspect_ratio(left_eye)
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, ":O", (30,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            if not mouth_open: pyautogui.mouseDown() 
            mouth_open = True
        elif mouth_open:
            pyautogui.mouseUp()
            mouth_open = False
        elif (lear < EYE_AR_THRESH):
            print(lear)
            pyautogui.click(button='right')
        pyautogui.moveRel(SCALE_FACTOR*-delta_x, SCALE_FACTOR*delta_y)
        

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
