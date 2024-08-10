import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    bc = c - b

    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # In radians
    return np.degrees(angle)  # Convert to degrees

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load video
cap = cv2.VideoCapture('arshad_gold.mp4')

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        hip_knee_angle = calculate_angle(hip, knee, ankle)
        angle = calculate_angle(shoulder, elbow, wrist)

        # Prepare text for overlay
        angle_text = f'Elbow Angle: {angle:.2f}'
        hip_knee_text = f'Hip-Knee Angle: {hip_knee_angle:.2f}'

        # Create a background rectangle for the text
        overlay_height = 80  # Height of the box
        angle_text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        hip_knee_text_size = cv2.getTextSize(hip_knee_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        box_width = max(angle_text_size[0], hip_knee_text_size[0]) + 20

        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + overlay_height), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # Create a semi-transparent effect

        # Draw text in black color
        cv2.putText(frame, angle_text, (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)  # Black text
        cv2.putText(frame, hip_knee_text, (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)  # Black text

    # Write the processed frame to the video
    out.write(frame)

    # Display the frame
    cv2.imshow('Javelin Throw Analysis', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the VideoWriter object
out.release()

cap.release()
cv2.destroyAllWindows()
