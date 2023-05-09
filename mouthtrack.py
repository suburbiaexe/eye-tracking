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

######## SETTINGS ########
#thresholds for marking facial expressions
MOUTH_AR_THRESH = 0.79
SCALE_FACTOR = 3.0
EYE_AR_THRESH = 0.2
DRAW_MODE = True # True turns off right click
DEMO_VIDEO_MODE = False # True turns off clicking entirely
DETECT_MULTIPLE = False # If true, will detect multiple faces

# dlib detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

# frame vars
frame_width = 640
frame_height = 360


######## HELPERS ########
def mouth_aspect_ratio(mouth):
    """calculates the aspect ratio of the mouth

    Args:
        mouth (list): list of coordinates for points defining the mouth

    Returns:
        float: mouth aspect ratio
    """
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8]) # 53, 57
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55
    return (A + B) / (2.0 * C)

def eye_aspect_ratio(eye):
    """calculates the aspect ratio of the eye

    Args:
        eye (list): list of coordinates for points defining the eye

    Returns:
        float: eye aspect ratio
    """
    A = dist.euclidean(eye[1], eye[5]) # 44, 48; one top/bottom pair
    B = dist.euclidean(eye[2], eye[4]) # 53, 57; other top-bottom pair
    C = dist.euclidean(eye[0], eye[3]) # 49, 55; left-right extremes
    return (A + B) / (2.0 * C)


######## LIVE TRACKING ########
# start the video stream thread
vs = VideoStream(src=0).start() #NOTE- src=0 for webcam, 1 for ext camera
time.sleep(1.0)
time.sleep(1.0)

# loop vars
prev_x, prev_y = (0,0)
first_loop = True
mouth_open = False

# loop over frames from the webcam video
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=frame_width)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ### replaced below w/
    rects = detector(frame, 0)
    if not DETECT_MULTIPLE: rects = rects[:1]
    for rect in rects: 

        # get landmark points 
        # indices always map to the same facial point (correspondences are here: https://www.researchgate.net/figure/Sixty-eight-facial-landmarks-obtained-by-the-Dlib-facial-landmark-predictor-Kazemi-and_fig1_343753489)
        landmark_pts = face_utils.shape_to_np(predictor(frame, rect)) #NOTE- change first back to grey?
        mouth = landmark_pts[49:68]
        eyes = landmark_pts[36:48]
        left_eye, right_eye = landmark_pts[42:48], landmark_pts[36:42]
        nose = landmark_pts[30]

        # mouth
        mar = mouth_aspect_ratio(mouth)
        mouthHull = cv2.convexHull(mouth)
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, ":O", (30,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
    
        # nose
        if first_loop:
            prev_x, prev_y = nose
            first_loop = False
        delta_x, delta_y = nose[0] - prev_x, nose[1] - prev_y
        prev_x, prev_y = nose[0], nose[1]
        cv2.circle(frame, nose, 2, (0, 0, 255), -1)

        # eyes
        l_ear = eye_aspect_ratio(left_eye)
        r_ear = eye_aspect_ratio(right_eye)
        for coord in eyes:
            cv2.circle(frame, coord, 2, (255, 0, 0), -1)

        # move mouse
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, ":O", (30,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            if not mouth_open and not DEMO_VIDEO_MODE: pyautogui.mouseDown() 
            mouth_open = True
        elif mouth_open and not DEMO_VIDEO_MODE:
            pyautogui.mouseUp()
            mouth_open = False
        elif (r_ear < EYE_AR_THRESH) and (l_ear > EYE_AR_THRESH) and not DRAW_MODE and not DEMO_VIDEO_MODE:
            pyautogui.click(button='right')
        if not DEMO_VIDEO_MODE:
            pyautogui.moveRel(SCALE_FACTOR*-delta_x, SCALE_FACTOR*delta_y)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
