import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

prevTime = 0
currTime = 0
capture = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = capture.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img)
    if len(landmarkList) != 0:
        print(landmarkList[4])

    # Get FPS
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    # Display FPS
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display Image
    cv2.imshow("Image", img)
    cv2.waitKey(1)