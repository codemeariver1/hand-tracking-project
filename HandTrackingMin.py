import cv2
import mediapipe as mp
import time

# print(cv2.__version__)
capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for handId, landmark in enumerate(handLandmarks.landmark):
                # print(handId, landmark)

                # Initialize the height, width, and channel
                h, w, ch = img.shape

                # Find the center's position
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                print(handId, cx, cy)
                # Track any part of the hand
                #if handId == 4:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Draw the line connections in the hands
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    # Get FPS
    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    # Display FPS
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display Image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
