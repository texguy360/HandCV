import cv2
import mediapipe as mp
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        #self.pow = math.pow
        #self.sqrt = math.sqrt

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                lmlist.append([id, cx, cy])
        return lmlist
    def index_base(self, lmlist, handNo=0):
        iblist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                xib = lmlist[5][1] - lmlist[0][1]
                yib = lmlist[5][2] - lmlist[0][2]
                iblist.append([xib, yib])
        return iblist
    def index_finger(self, lmlist, handNo=0):
        iflist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                xib = lmlist[5][1] - lmlist[0][1]
                yib = lmlist[5][2] - lmlist[0][2]
                iflist.append([xib, yib])
        return iflist



def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        iblist = detector.index_base(lmlist)
        iflist = detector.index_finger(lmlist)

        if len(lmlist) != 0:
           print(iblist, iflist)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
