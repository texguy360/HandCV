import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def findHands(img, draw=True):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    return img

def findPosition(img, handNo=0, draw=True):
    lmlist = []
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)
    return lmlist

def main(img):
    cap = cv2.VideoCapture(0)
    detector1 = findPosition(img)
    detector2 = findHands(img)

    while True:
        success, img = cap.read()
        img = detector1.findHands(img)
        lmlist = detector2.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
