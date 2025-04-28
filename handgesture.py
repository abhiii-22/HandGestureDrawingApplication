#This is the code 
import cv2
import numpy as np
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0) #for some computers webcam will be accesses by using VideoCapture(1) or VideoCapture(2)

# Set frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create a canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Colors for drawing
colors = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
    (0, 128, 255),   # Sky Blue
    (255, 0, 128)    # Pink
]
current_color_index = 0
current_color = colors[current_color_index]

# Previous finger position
prev_x, prev_y = 0, 0

# Variable for erasing
erase_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with mediapipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Detect finger states
            fingers = []
            fingers.append(lm_list[4][0] > lm_list[3][0])  # Thumb
            for tip_id in [8, 12, 16, 20]:
                fingers.append(lm_list[tip_id][1] < lm_list[tip_id - 2][1])  # Other fingers

            # Drawing Mode: Only Index Finger Up
            if fingers[1] and not any(fingers[2:]):
                cx, cy = lm_list[8]

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy

                # Draw with more opacity (alpha blending)
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), current_color, 3)
                prev_x, prev_y = cx, cy
            else:
                prev_x, prev_y = 0, 0

            # Change color: Thumb and Index Finger close together
            distance = np.hypot(lm_list[4][0] - lm_list[8][0], lm_list[4][1] - lm_list[8][1])
            if distance < 30:  # Make it a more deliberate gesture
                current_color_index = (current_color_index + 1) % len(colors)
                current_color = colors[current_color_index]

            # Palm gesture to erase: All 5 fingers up (detect if all tips are above base)
            if all(fingers):
                erase_mode = True
            else:
                erase_mode = False

            # Erase mode: if palm detected, clear drawing at current position
            if erase_mode:
                canvas = np.zeros_like(canvas)  # Clear the canvas

            # Draw landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Merge canvas and frame
    frame_with_canvas = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.rectangle(frame, (5, 5), (40, 40), current_color, -1)  # Color box
    cv2.putText(frame, f'Color: {current_color}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with canvas in a separate window
    cv2.imshow("Finger Painting - Webcam", frame)
    cv2.imshow("Drawing Canvas", canvas)

    # Press 'ESC' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
