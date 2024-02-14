import cv2
import mediapipe as mp
from random import randrange
from turtle import *
from freegames import square, vector

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)  # Initialize webcam

food = vector(0, 0)
snake = [vector(10, 0)]
aim = vector(0, -10)

def process_gestures():
    global aim
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get wrist landmark for basic gesture recognition
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            if wrist.x < 0.4:  # Hand is on the left side
                aim = vector(-10, 0)
            elif wrist.x > 0.6:  # Hand is on the right side
                aim = vector(10, 0)
            elif wrist.y < 0.4:  # Hand is up
                aim = vector(0, 10)
            elif wrist.y > 0.6:  # Hand is down
                aim = vector(0, -10)

def change(x, y):
    """Change snake direction based on hand gestures."""
    global aim
    aim.x = x
    aim.y = y

def inside(head):
    """Return True if head inside boundaries."""
    return -200 < head.x < 190 and -200 < head.y < 190

def move():
    process_gestures()  # Include a simple debug print inside to confirm it's called
    head = snake[-1].copy()
    head.move(aim)
    
    if not inside(head) or head in snake:
        # Reset the game for debugging
        snake.clear()
        snake.append(vector(10, 0))
        return
    
    snake.append(head)
    
    # Simplify food logic for debugging
    if head == food:
        food.x = randrange(-15, 15) * 10
        food.y = randrange(-15, 15) * 10
    else:
        snake.pop(0)
    
    clear()
    for body in snake:
        square(body.x, body.y, 9, 'black')
    square(food.x, food.y, 9, 'green')
    update()
    ontimer(move, 100)

    # Existing game logic...
    # Make sure to include the rest of the move() function here.

setup(420, 420, 370, 0)
hideturtle()
tracer(False)
move()
done()

cap.release()
