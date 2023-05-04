
import cv2
import dlib
import numpy as np
import pyautogui

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

SCALE_FACTOR = 3.0 # changed from 2.0 in order to cover the entire screen more easily

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass

# isabelle added
NOD_THRESH = 30
def is_nod(delta_x, delta_y):
    if abs(delta_y) > NOD_THRESH:
        print('nodded')
        return True
    else:
        return False
# end isa added

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

prev_x, prev_y = (0,0)
first_loop = True

x_deltas = []
y_deltas = []

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        
        for (x, y) in shape[36:48]:
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        avg_coord = np.mean(shape, axis=0)
        
        (nose_x,nose_y) = shape[30]
        cv2.circle(img, (nose_x,nose_y), 2, (0, 0, 255), -1)
        if first_loop: 
            prev_x, prev_y = nose_x, nose_y
            first_loop = False
        delta_x, delta_y = nose_x - prev_x, nose_y - prev_y
        prev_x, prev_y = nose_x, nose_y
        
        ##isabelle added
        x_deltas.append(delta_x)
        y_deltas.append(delta_y)
        if is_nod(delta_x, delta_y):
            pyautogui.click()
        ## end isabelle added
        else:
            # print(delta_x,delta_y)
            pyautogui.moveRel(SCALE_FACTOR*-delta_x, SCALE_FACTOR*delta_y)

    # show the image with the face detections + facial landmarks
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()