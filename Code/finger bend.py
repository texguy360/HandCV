import cv2
import mediapipe as mp
import math


class Detector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_confidence,
                                        self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img, draw=True):
        self.results = self.hands.process(img)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0):
        lmlist = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                lmlist.append([id, cx, cy])
        return lmlist


def main():
    cap = cv2.VideoCapture(0)
    detector = Detector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmlist = detector.find_position(img)

        if len(lmlist) != 0:
            index_base0_5 = math.sqrt(
                math.pow(lmlist[5][1] - lmlist[0][1], 2) + math.pow(lmlist[5][2] - lmlist[0][2], 2))
            index_tip8_0 = math.sqrt(
                math.pow(lmlist[8][1] - lmlist[0][1], 2) + math.pow(lmlist[8][2] - lmlist[0][2], 2))
            index_bend = index_tip8_0/index_base0_5

            middle_base0_9 = math.sqrt(
                math.pow(lmlist[9][1] - lmlist[0][1], 2) + math.pow(lmlist[9][2] - lmlist[0][2], 2))
            middle_tip12_0 = math.sqrt(
                math.pow(lmlist[12][1] - lmlist[0][1], 2) + math.pow(lmlist[12][2] - lmlist[0][2], 2))
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
            thumb_bend = thumb_tip4_2/thumb_base3_2

            hand_rotate = (math.sqrt(
                math.pow(lmlist[5][1] - lmlist[2][1], 2) + math.pow(lmlist[5][2] - lmlist[2][2], 2)))/(math.sqrt(
                math.pow(lmlist[5][1] - lmlist[17][1], 2) + math.pow(lmlist[5][2] - lmlist[17][2], 2)))

            hand_bend = (math.sqrt(
                math.pow(lmlist[17][1] - lmlist[0][1], 2) + math.pow(lmlist[17][2] - lmlist[0][2], 2)))/(math.sqrt(
                math.pow(lmlist[5][1] - lmlist[2][1], 2) + math.pow(lmlist[5][2] - lmlist[2][2], 2)))

            print(hand_rotate)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
