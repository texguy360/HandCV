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

    def indexFinger(self, lmlist):
        while True:
            index_base05 = self.sqrt(self.pow(lmlist[5][1] - lmlist[0][1], 2) + self.pow(lmlist[5][2] - lmlist[0][2], 2))
            index_tip85 = self.sqrt(self.pow(lmlist[8][1] - lmlist[5][1], 2) + self.pow(lmlist[8][2] - lmlist[5][2], 2))
            index_bend = int(index_tip85)/int(index_base05)
        return index_bend


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)

        if len(lmlist) != 0:
            index_base0_5 = math.sqrt(
                math.pow(lmlist[5][1] - lmlist[0][1], 2) + math.pow(lmlist[5][2] - lmlist[0][2], 2))
            index_tip8_0 = math.sqrt(math.pow(lmlist[8][1] - lmlist[0][1], 2) + math.pow(lmlist[8][2] - lmlist[0][2], 2))
            index_bend = index_tip8_0/index_base0_5
            if index_bend > 1.7:
                index = 0
            elif 1 < index_bend < 1.7:
                index = 1
            elif index_bend < 1:
                index = 2
            print(index)




        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
