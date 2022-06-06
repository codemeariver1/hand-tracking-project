import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.complexity, self.detectionConf, self.trackingConf
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Draw the line connections in the hands
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for handId, landmark in enumerate(myHand.landmark):
                # print(handId, landmark)
                # Initialize the height, width, and channel
                h, w, ch = img.shape
                # Find the center's position
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                # print(handId, cx, cy)
                landmarkList.append([handId, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return landmarkList

def main():
    prevTime = 0
    currTime = 0
    capture = cv2.VideoCapture(0)
    detector = handDetector()

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


# Define function that inherits module
if __name__ == "__main__":
    main()