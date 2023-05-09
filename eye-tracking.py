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

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255,255,255)

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


def position_delta(p): # keep track of how a position has changed over time.
    global first_loop, prev_x, prev_y
    if first_loop:
        prev_x, prev_y = p
        first_loop = False
    delta_x, delta_y = p[0] - prev_x, p[1] - prev_y
    prev_x, prev_y = p
    return delta_x, delta_y

######## LIVE TRACKING ########
# start the video stream thread
vs = VideoStream(src=0).start() #NOTE- src=0 for webcam, 1 for ext camera
time.sleep(2.0)

# loop vars
prev_x, prev_y = (0,0)
first_loop = True
mouth_open = False

# loop over frames from the webcam video
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=frame_width)
    rects = detector(frame, 0)

    # Option between detecting multiple faces or just one
    # (in the current state, the program is very buggy with DETECT_MULTIPLE=True)
    if not DETECT_MULTIPLE: rects = rects[:1]
    for rect in rects: 

        # get landmark points 
        # indices always map to the same facial point (correspondences are here: https://www.researchgate.net/figure/Sixty-eight-facial-landmarks-obtained-by-the-Dlib-facial-landmark-predictor-Kazemi-and_fig1_343753489)
        landmark_pts = face_utils.shape_to_np(predictor(frame, rect)) 
        mouth = landmark_pts[49:68]
        eyes = landmark_pts[36:48]
        left_eye, right_eye = landmark_pts[42:48], landmark_pts[36:42]
        nose = landmark_pts[30]

        # MOUTH
        # draw the green contour border
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, GREEN, 1)
    
        # NOSE
        delta_x, delta_y = position_delta(nose)
        # red point on nose
        cv2.circle(frame, nose, 2, RED, -1)

        # eyes
        l_ear = eye_aspect_ratio(left_eye)
        r_ear = eye_aspect_ratio(right_eye)
        for coord in eyes:
            cv2.circle(frame, coord, 2, BLUE, -1)

        # change mouse state:
        if mouth_aspect_ratio(mouth) > MOUTH_AR_THRESH:
            cv2.putText(frame, ":O", (30,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE,2)
            if not mouth_open and not DEMO_VIDEO_MODE: pyautogui.mouseDown() 
            mouth_open = True
        elif mouth_open and not DEMO_VIDEO_MODE: # if the mouth was open but is now closed
            pyautogui.mouseUp()
            mouth_open = False
        elif (r_ear < EYE_AR_THRESH) and (l_ear > EYE_AR_THRESH) and not DRAW_MODE and not DEMO_VIDEO_MODE:
            pyautogui.click(button='right')
        if not DEMO_VIDEO_MODE: # move mouse based on change of nose position
            pyautogui.moveRel(SCALE_FACTOR*-delta_x, SCALE_FACTOR*delta_y)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()