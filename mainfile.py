import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


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

# Switch-case like function for exercises
def exercise_logic(exercise, landmarks):
    exercises = {
        "arm_curl": arm_curl,
        "squat": squat
    }
    return exercises[exercise](landmarks)

# Function to get user input for exercise selection
def get_user_choice():
    print("Choose your exercise:")
    print("1. Arm Curls")
    print("2. Squats")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        return "arm_curl"
    elif choice == '2':
        return "squat"
    else:
        print("Invalid input. Defaulting to Arm Curls.")
        return "arm_curl"

# Main function for video capture and exercise tracking
def main():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    
    # Get user input for the exercise type
    exercise_type = get_user_choice()  # Choose between 'arm_curl' and 'squat'

    # Mediapipe pose detection setup
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make pose detection
            results = pose.process(image)
            
            # Recolor back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks and apply exercise logic
            try:
                landmarks = results.pose_landmarks.landmark
                angle, joint = exercise_logic(exercise_type, landmarks)

                # Rep counting logic based on the exercise type
                if exercise_type == "arm_curl":
                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage == "down":
                        stage = "up"
                        counter += 1
                        print(f'Arm Curl Reps: {counter}')
                elif exercise_type == "squat":
                    if angle > 170:
                        stage = "up"
                    if angle < 90 and stage == "up":
                        stage = "down"
                        counter += 1
                        print(f'Squat Reps: {counter}')
                
                # Display the angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(joint, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display the reps and stage
                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                cv2.putText(image, 'REPS', (15, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'STAGE', (65, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (60, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass

            # Render the pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
            # Display the image
            cv2.imshow('Mediapipe Feed', image)

            # Break the loop with 'q' key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
