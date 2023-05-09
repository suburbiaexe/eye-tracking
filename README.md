# eye-tracking

For this project, we wanted to implement face
tracking to control the computer cursor. 
We were able to identify major facial landmarks to
use for webcam position tracking, but we weren’t able to
get eye (pupil) tracking to work properly, as we couldn’t get
granular enough feature points that allowed for accuracy
and consistency. As a result, we used the popular 68-point 
model from Dlib to identify facial landmarks, and then used 
the change in those landmarks to map to the cursor’s position.

To run the program, run python mouthtrack.py. A videostream should 
show up with your points highlighted on your eyes, nose, and mouth.
For full functionality, ensure the DRAW_MODE and DEMO_VIDEO_MODE 
variables (both in mouthtrack.py) are both set to False. 

As you move your head, the cursor will move. If you open (and 
close) your mouth, it will click, and if you open your mouth and 
move your head, it will click and drag. You can right click by 
winking your right eye.

This implementation is less accessible than eye tracking control 
of a mouse, but we believe our model not only fit better within 
the scope of the project, but also allows for more accuracy and 
robustness in registering the mouse movements, because the
facial movements to control the cursor are much bigger than
pupil movements. 

We believe that with more time, we could create a custom feature 
detection model built off of the one we ended up using that 
would allow us to implement pupil detection (and thus gaze 
tracking) to control the cursor