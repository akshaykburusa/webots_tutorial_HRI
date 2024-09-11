"""tutorial1_tracker controller."""

# Noa Baijens s1010113
# Micha Lobbezoo s1019806

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Keyboard, Display, Motion
import numpy as np
import cv2


class MyRobot(Robot):
    def __init__(self, ext_camera_flag):
        super(MyRobot, self).__init__()
        print('> Starting robot controller')
        
        self.timeStep = 32 # Milisecs to process the data (loop frequency) - Use int(self.getBasicTimeStep()) for default
        self.state = 0 # Idle starts for selecting different states
        
        # Sensors init
        self.gps = self.getGPS('gps')
        self.gps.enable(self.timeStep)
        
        self.step(self.timeStep) # Execute one step to get the initial position
        
        self.ext_camera = ext_camera_flag        
        self.displayCamExt = self.getDisplay('CameraExt')
        
        self.bottom_cam = self.getCamera('CameraBottom')
        self.bottom_cam.enable(self.timeStep)
                
        #external camera
        if self.ext_camera:
            self.cameraExt = cv2.VideoCapture(0)
            
        self.haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
         
        # Actuators init
        self.head_yaw = self.getDevice('HeadYaw') 
        print(self.head_yaw.getMinPosition(), self.head_yaw.getMaxPosition()) 
        self.head_pitch = self.getDevice('HeadPitch')
        print(self.head_pitch.getMinPosition(), self.head_pitch.getMaxPosition())
            
        # Keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()
        
        # Motions
        self.forwards = Motion("../../motions/Forwards.motion")
        self.backwards = Motion("../../motions/Backwards.motion")
        self.shoot = Motion("../../motions/Shoot.motion")
        self.wave = Motion("../../motions/HandWave.motion")

        
    # Captures the external camera frames 
    # Returns the image downsampled by 2   
    def camera_read_external(self):
        img = []
        if self.ext_camera:
            # Capture frame-by-frame
            ret, frame = self.cameraExt.read()
            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # From openCV BGR to RGB
            img = cv2.resize(img, None, fx=0.5, fy=0.5) # image downsampled by 2
                        
        return img
            
    # Displays the image on the webots camera display interface
    def image_to_display(self, img):
        if self.ext_camera:
            height, width, channels = img.shape
            imageRef = self.displayCamExt.imageNew(cv2.transpose(img).tolist(), Display.RGB, width, height)
            self.displayCamExt.imagePaste(imageRef, 0, 0)
    
    def print_gps(self):
        gps_data = self.gps.getValues();
        print('----------gps----------')
        print(' [x y z] =  [' + str(gps_data[0]) + ',' + str(gps_data[1]) + ',' + str(gps_data[2]) + ']' )
        
    def printHelp(self):
        print(
            'Commands:\n'
            ' H for displaying the commands\n'
            ' G for print the gps\n'
            ' < for head turn left\n'
            ' > for head turn right\n'
            ' ^ for walking forwards\n'
            ' v for walking backwards\n'
            ' s for stop\n'
        )
        
    def move_head(self, velocity):
        self.head_yaw.setPosition(float("inf"))
        self.head_yaw.setVelocity(velocity)
    
    def walk(self, direction):
        direction.play()
        direction.setLoop(True)
        
    def stop(self):
        self.move_head(0)
        self.forwards.stop()
        self.backwards.stop()
        print('stop')
    
    def run_keyboard(self):
    
        self.printHelp()
        previous_message = ''

        # Main loop.
        while True:
            # Deal with the pressed keyboard key.
            k = self.keyboard.getKey()
            message = ''
            if k == ord('G'):
                self.print_gps() 
            elif k == ord('H'):
                self.printHelp()
            elif k == Keyboard.LEFT:
                self.move_head(-1)
            elif k == Keyboard.RIGHT:
                self.move_head(1)
            elif k == Keyboard.UP:
                self.walk(self.forwards)
            elif k == Keyboard.DOWN:
                self.walk(self.backwards)
            elif k == ord('S'):
                self.stop()

            # Perform a simulation step, quit the loop when
            # Webots is about to quit.
            if self.step(self.timeStep) == -1:
                break
                
        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release() 
    
    # Facial recognition
    def recognize_face(self, image):
        faces = self.haar.detectMultiScale(image) #will detect all faces in image
        return faces
    
    def look_at(self, x, y):  
        # see if x and y fall within bounds of headyaw and headpitch
        if self.head_yaw.getMinPosition() < x < self.head_yaw.getMaxPosition():
            self.head_yaw.setPosition(x)
        if self.head_pitch.getMinPosition() < y < self.head_pitch.getMaxPosition():
            self.head_pitch.setPosition(y)
                
    # Face following main function
    def run_face_follower(self):
        # main control loop: perform simulation steps of self.timeStep milliseconds
        # and leave the loop when the simulation is over
        #initialize head in the midddle
        self.head_yaw.setPosition(0)
        self.head_pitch.setPosition(0)
        while self.step(self.timeStep) != -1:
            # Write your controller here
            image = self.camera_read_external()
            w_camera = self.displayCamExt.getWidth()
            h_camera = self.displayCamExt.getHeight()
            faces = self.recognize_face(image) 
            for (x,y,w,h) in faces:
                # draw rectangle around each face
                image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            #see if face was detected and select only the first one
            if len(faces)>0:
                (x1,y1,w1,h1) = faces[0]
                # find centre of face rectangle
                centre_x = (x1 + (w1/2))
                centre_y = (y1 + (h1/2))
                # determine where Nao should look
                focus_x = (centre_x - w_camera/2)/w_camera
                focus_y = (centre_y - h_camera/2)/w_camera
                print('focus_x: ' + str(focus_x))
                print('focus_y: ' + str(focus_y)) 
                self.look_at(focus_x, focus_y)            
            self.image_to_display(image)
        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release()   
    
    
    def get_ball_contours(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([50,100,100]) 
        upper = np.array([75,255,255]) 
        mask = cv2.inRange(hsv, lower, upper)
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
        
    
    def run_ball_follower(self):
        w_camera = self.bottom_cam.getWidth()
        h_camera = self.bottom_cam.getHeight()
        while self.step(self.timeStep) != -1:
            image = np.uint8(self.bottom_cam.getImageArray())
            cx, cy = 0, 0
            contours = self.get_ball_contours(image)
            for contour in contours:
                m = cv2.moments(contour)
                if m["m00"]>0:
                    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            K=0.2 #put higher than the instructions to make head movements more visible
            dx = K * ((cx / w_camera) - 0.5)
            dy = K * ((cy / h_camera) - 0.5)
            self.look_at(-dy,dx)
    
    def run_hri(self):
        time_count = 0
        seen_face = False
        seen_ball = False
        while self.step(self.timeStep) != -1:
            # reset after about 6 seconds
            if time_count >= 200:
                seen_face = False
                seen_ball = False
                time_count = 0
            # detect ball and shoot
            image_bot = np.uint8(self.bottom_cam.getImageArray())
            if len(self.get_ball_contours(image_bot)) > 0 and not seen_ball:
                seen_ball = True
                self.shoot.play()
            # detect face and wave
            else:  
                image_ext = self.camera_read_external()
                self.image_to_display(image_ext)
                if len(self.recognize_face(image_ext)) > 0 and not seen_face:
                    seen_face = True
                    self.wave.play()
            time_count+=1
        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release()            

    
# create the Robot instance and run the controller
#robot = MyRobot(ext_camera_flag = False)
robot = MyRobot(ext_camera_flag = True)
#robot.run_keyboard()
#robot.run_face_follower()
#robot.run_ball_follower()
robot.run_hri()


