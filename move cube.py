import cv2
import mediapipe as mp
import turtle
import threading
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Turtle Graphics setup
window = turtle.Screen()
window.title("Hand Gesture Controlled Game")
window.bgcolor("black")
window.setup(width=600, height=600)

player = turtle.Turtle("circle")
player.color("blue")
player.penup()
player.speed(0)

# Movement flags
move_right_flag = False
move_left_flag = False

def update_movement_flags(right_hand_detected, left_hand_detected):
    global move_right_flag, move_left_flag
    move_right_flag = right_hand_detected
    move_left_flag = left_hand_detected

def gesture_control():
    cap = cv2.VideoCapture(0)

    while True:
        _, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image)

        right_hand_detected = False
        left_hand_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label
                if hand_type == "Right":
                    right_hand_detected = True
                elif hand_type == "Left":
                    left_hand_detected = True

        update_movement_flags(right_hand_detected, left_hand_detected)

        if cv2.waitKey(5) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def update_player_position():
    global move_right_flag, move_left_flag
    x, y = player.xcor(), player.ycor()
    if move_right_flag:
        player.goto(x + 10, y)
    elif move_left_flag:
        player.goto(x - 10, y)
    # Reset the flags after moving
    move_right_flag = False
    move_left_flag = False


def game_loop():
    while True:
        window.update()
        update_player_position()
        time.sleep(0.1)  # Use time.sleep() for delays

# Start gesture control in a separate thread
thread = threading.Thread(target=gesture_control, daemon=True)
thread.start()

game_loop()
turtle.done()
