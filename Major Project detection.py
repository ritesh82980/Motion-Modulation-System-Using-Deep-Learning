# Importing Libraries
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
from deepface import DeepFace  # Emotion detection

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Initialize variables for motion detection
ret, prev_frame = cap.read()
if not ret:
    print("Error: Unable to access the camera.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
motion_detected = False

while True:
    # Read video frame by frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image for hand tracking
    process = hands.process(frame_rgb)
    landmark_list = []

    # If hands are present in the image (frame)
    if process.multi_hand_landmarks:
        for handlm in process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                # Store height and width of image
                height, width, _ = frame.shape
                # Calculate and append x, y coordinates of hand landmarks
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmark_list.append([_id, x, y])
            # Draw Landmarks
            mp_drawing.draw_landmarks(frame, handlm, mp_hands.HAND_CONNECTIONS)

    # Adjust brightness using hand landmarks
    if landmark_list:
        # Store x, y coordinates of (tip of) thumb
        x_1, y_1 = landmark_list[4][1], landmark_list[4][2]
        # Store x, y coordinates of (tip of) index finger
        x_2, y_2 = landmark_list[8][1], landmark_list[8][2]

        # Draw circle on thumb and index finger tip
        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)

        # Draw line from tip of thumb to tip of index finger
        cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)

        # Calculate the distance between thumb and index finger
        distance = hypot(x_2 - x_1, y_2 - y_1)

        # Map the distance to brightness levels
        brightness_level = np.interp(distance, [15, 220], [0, 100])
        # Set brightness
        sbc.set_brightness(int(brightness_level))

    # Detect emotions using DeepFace
    try:
        # Convert frame to RGB for DeepFace analysis
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze emotions (focus on the face area)
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)

        # Handle cases where a list is returned instead of a dictionary
        if isinstance(analysis, list):
            analysis = analysis[0]  # Take the first detected face's emotion analysis

        # Extract dominant emotion
        dominant_emotion = analysis.get('dominant_emotion', 'None')

        # Display the detected emotion
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    except Exception as e:
        print(f"Emotion detection skipped: {e}")

    # Detect faces and count humans
    face_results = face_detection.process(frame_rgb)
    if face_results.detections:
        human_count = len(face_results.detections)
        cv2.putText(frame, f"Humans Detected: {human_count}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw bounding boxes around faces
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)
    else:
        cv2.putText(frame, "Humans Detected: 0", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Motion Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff_frame = cv2.absdiff(prev_gray, gray)
    _, threshold_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
    motion_detected = threshold_frame.sum() > 5000  # Arbitrary threshold

    if motion_detected:
        cv2.putText(frame, "Motion Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    # Update previous frame
    prev_gray = gray.copy()

    # Display the video feed
    cv2.imshow('Human Tracking System', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
