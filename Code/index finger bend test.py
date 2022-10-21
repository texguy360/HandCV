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

    #def indexFinger(self, lmlist):
        #while True:
            #index_base05 = self.sqrt(self.pow(lmlist[5][1] - lmlist[0][1], 2) + self.pow(lmlist[5][2] - lmlist[0][2], 2))
            #index_tip85 = self.sqrt(self.pow(lmlist[8][1] - lmlist[5][1], 2) + self.pow(lmlist[8][2] - lmlist[5][2], 2))
            #index_bend = int(index_tip85)/int(index_base05)
        #return index_bend


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

            middle_base0_9 = math.sqrt(
                math.pow(lmlist[9][1] - lmlist[0][1], 2) + math.pow(lmlist[9][2] - lmlist[0][2], 2))
            middle_tip12_0 = math.sqrt(math.pow(lmlist[12][1] - lmlist[0][1], 2) + math.pow(lmlist[12][2] - lmlist[0][2], 2))
            middle_bend = middle_tip12_0/middle_base0_9

            ring_base0_13 = math.sqrt(
                math.pow(lmlist[13][1] - lmlist[0][1], 2) + math.pow(lmlist[13][2] - lmlist[0][2], 2))
            ring_tip16_0 = math.sqrt(
                math.pow(lmlist[16][1] - lmlist[0][1], 2) + math.pow(lmlist[16][2] - lmlist[0][2], 2))
            ring_bend = ring_tip16_0/ring_base0_13

            small_base0_17 = math.sqrt(
                math.pow(lmlist[17][1] - lmlist[0][1], 2) + math.pow(lmlist[17][2] - lmlist[0][2], 2))
            small_tip20_0 = math.sqrt(
                math.pow(lmlist[20][1] - lmlist[0][1], 2) + math.pow(lmlist[20][2] - lmlist[0][2], 2))
            small_bend = small_tip20_0/small_base0_17

            thumb_base1_0 = math.sqrt(
                math.pow(lmlist[2][1] - lmlist[0][1], 2) + math.pow(lmlist[2][2] - lmlist[0][2], 2))
            thumb_mid4_17 = math.sqrt(
                math.pow(lmlist[4][1] - lmlist[17][1], 2) + math.pow(lmlist[4][2] - lmlist[17][2], 2))
            thumb_rotate = thumb_mid4_17/thumb_base1_0

            thumb_tip4_2 = math.sqrt(
                math.pow(lmlist[4][1] - lmlist[2][1], 2) + math.pow(lmlist[4][2] - lmlist[2][2], 2))
            thumb_base3_2 = math.sqrt(
                math.pow(lmlist[3][1] - lmlist[2][1], 2) + math.pow(lmlist[3][2] - lmlist[2][2], 2))
            thumb_bend = thumb_tip4_2/thumb_base3_2 #1.67

            print(index_bend, middle_bend, ring_bend, small_bend, thumb_rotate, thumb_bend)


        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
