import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,  # Adjusted to detect two hands
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam for video capture
cap = cv2.VideoCapture(0)

# Initialize the position and size of the basketball
circle_center = (320, 240)  # Initial position at the center of the screen
circle_radius = 30

# Initialize counters for balls in right and left hands
balls_in_right_hand = 0
balls_in_left_hand = 0

# Initialize flags to track whether the ball is currently in each hand
ball_in_right_hand = False
ball_in_left_hand = False

# Main loop for capturing and processing video frames
while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Flip the frame horizontally to correct mirroring
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks and connections on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Determine if it's a right hand or left hand
            if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                dot_color = (0, 0, 255)  # Red color for right hand
                if not ball_in_right_hand:
                    balls_in_right_hand += 1  # Increment counter for right hand
                    ball_in_right_hand = True
            else:
                dot_color = (0, 255, 0)  # Green color for left hand
                if not ball_in_left_hand:
                    balls_in_left_hand += 1  # Increment counter for left hand
                    ball_in_left_hand = True
            
            # Get the positions of the fingertip landmarks (Landmarks 4, 8, 12, 16, and 20)
            fingertips = []
            for landmark_id in [4, 8, 12, 16, 20]:
                x, y = int(hand_landmarks.landmark[landmark_id].x * frame.shape[1]), \
                       int(hand_landmarks.landmark[landmark_id].y * frame.shape[0])
                fingertips.append((x, y))
            
            # Update the position and size of the basketball based on the average position of the fingertips
            if fingertips:
                circle_center = (int(np.mean([x for x, y in fingertips])), int(np.mean([y for x, y in fingertips])))

            # Draw all landmarks as filled circles with the determined color
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, dot_color, -1)  # Draw filled circle with determined color
            
            # Draw connections between landmarks on the same hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_draw.DrawingSpec(color=dot_color, thickness=1, circle_radius=1))

    # Draw the basketball on the frame
    # Outer orange circle
    cv2.circle(frame, circle_center, circle_radius, (0, 140, 255), -1)
    # Black lines to simulate basketball texture
    for angle in range(0, 360, 30):
        x1 = int(circle_center[0] + circle_radius * np.cos(np.radians(angle)))
        y1 = int(circle_center[1] + circle_radius * np.sin(np.radians(angle)))
        x2 = int(circle_center[0] + (circle_radius - 3) * np.cos(np.radians(angle)))
        y2 = int(circle_center[1] + (circle_radius - 3) * np.sin(np.radians(angle)))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # Add text annotations for the counters
    cv2.putText(frame, f'Balls in Right Hand: {balls_in_right_hand}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Balls in Left Hand: {balls_in_left_hand}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the processed image with hand landmarks, connections, and the basketball
    cv2.imshow('Hand Gesture Recognition', frame)

    # Reset the flag indicating whether the ball is currently in each hand
    ball_in_right_hand = False
    ball_in_left_hand = False

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
