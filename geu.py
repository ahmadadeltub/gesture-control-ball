import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, mark the image as not writeable
    image.flags.writeable = False
    
    # Process the image and detect hands
    results = hands.process(image)

    # Draw the hand annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Gesture recognition logic (simplified for demonstration)
    # Count the number of fingers extended and map it to keyboard inputs
    if results.multi_hand_landmarks:
        # Example: If thumb is extended, press the right arrow key
        thumb_tip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_MCP].x
        if thumb_tip < thumb_mcp: # Thumb is extended (for right hand)
            pyautogui.press('right')
    
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
