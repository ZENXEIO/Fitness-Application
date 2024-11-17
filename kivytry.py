import cv2
import mediapipe as mp
import numpy as np
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivy.graphics.texture import Texture
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


# Exercise logic functions
def arm_curl(landmarks):
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    # Calculate angle
    angle = calculate_angle(shoulder, elbow, wrist)
    
    return angle, elbow


def squat(landmarks):
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    # Calculate angle
    angle = calculate_angle(hip, knee, ankle)
    
    return angle, knee


class ExerciseApp(MDApp):
    def build(self):
        # Layout for buttons and video stream
        self.layout = MDBoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Button for selecting Arm Curls
        self.arm_curl_button = MDRaisedButton(text="Arm Curls", pos_hint={'center_x': 0.5})
        self.arm_curl_button.bind(on_press=self.start_arm_curls)
        
        # Button for selecting Squats
        self.squat_button = MDRaisedButton(text="Squats", pos_hint={'center_x': 0.5})
        self.squat_button.bind(on_press=self.start_squats)
        
        # Label to display reps
        self.reps_label = MDLabel(text="Reps: 0", halign="center", theme_text_color="Primary")
        
        # Image widget for video feed
        self.image = Image(size_hint=(1, 0.8))

        # Add widgets to layout
        self.layout.add_widget(self.image)
        self.layout.add_widget(self.arm_curl_button)
        self.layout.add_widget(self.squat_button)
        self.layout.add_widget(self.reps_label)
        
        # Initialize some variables
        self.counter = 0
        self.stage = None
        self.exercise_type = None
        
        # OpenCV video capture with reduced resolution for performance
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower the resolution to 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize Mediapipe Pose once (outside the loop for efficiency)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Skip every 2 frames to reduce processing
        self.frame_skip = 2
        self.current_frame = 0
        
        # Schedule the update function to run repeatedly
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS
        
        return self.layout

    def start_arm_curls(self, instance):
        self.exercise_type = "arm_curl"

    def start_squats(self, instance):
        self.exercise_type = "squat"

    def exercise_logic(self, exercise, landmarks):
        if exercise == "arm_curl":
            return arm_curl(landmarks)
        elif exercise == "squat":
            return squat(landmarks)

    def update(self, dt):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1

            # Skip every nth frame (based on frame_skip value) to improve performance
            if self.current_frame % self.frame_skip != 0:
                return
            
            # Recolor the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Mediapipe pose detection (pose is initialized outside the loop)
            results = self.pose.process(image)
            
            image.flags.writeable = True
            
            # Draw pose landmarks and connections
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                
                landmarks = results.pose_landmarks.landmark
                if self.exercise_type:
                    angle, joint = self.exercise_logic(self.exercise_type, landmarks)

                    # Rep counting logic
                    if self.exercise_type == "arm_curl":
                        if angle > 160:
                            self.stage = "down"
                        if angle < 30 and self.stage == "down":
                            self.stage = "up"
                            self.counter += 1
                    elif self.exercise_type == "squat":
                        if angle > 170:
                            self.stage = "up"
                        if angle < 90 and self.stage == "up":
                            self.stage = "down"
                            self.counter += 1

                    # Update the UI with reps
                    self.reps_label.text = f"Reps: {self.counter}"
            
            # Convert to Kivy texture
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for Kivy
            buf = cv2.flip(image, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def on_stop(self):
        # Release the video capture when the app stops
        self.cap.release()


if __name__ == '__main__':
    ExerciseApp().run()
