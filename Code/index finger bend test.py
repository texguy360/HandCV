import cv2
import mediapipe as mp
import numpy as np


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

    def index_vector(self, img):
        list_x = []
        index_vector = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(hand.landmark):
                # list_x.append(lm.x)
                if id == 0 or id == 5 or id == 6 or id == 7 or id == 8:
                    index_vector.append(lm.x)
                    index_vector.append(lm.y)
        return index_vector

    # def index_x(self, list_x):
    #     index_x = [list_x[0], list_x[5], list_x[6], list_x[7], list_x[8]]
    #     return index_x

    # def find_y(self, img):
    #     # list_y = []
    #     index_y = []
    #     if self.results.multi_hand_landmarks:
    #         hand = self.results.multi_hand_landmarks[0]
    #         for id, lm in enumerate(hand.landmark):
    #             # list_y.append(lm.y)
    #             if id == 0 or id == 5 or id == 6 or id == 7 or id == 8:
    #                 index_y.append(lm.x)
    #     return index_y

    # def find_position(self, img, hand_number=0):
    #     lmlist = []
    #     if self.results.multi_hand_landmarks:
    #         hand = self.results.multi_hand_landmarks[hand_number]
    #         for id, lm in enumerate(hand.landmark):
    #             h, w, c = img.shape
    #             cx = int(lm.x * w)
    #             cy = int(lm.y * h)
    #             lmlist.append([id, cx, cy])
    #     return lmlist



def main():
    cap = cv2.VideoCapture(0)
    detector = Detector()
    index = 0

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        # list_x = detector.find_x(img)
        # list_y = detector.find_y(img)
        index = detector.index_vector(img)
        # lmlist = detector.find_position(img)

        if len(index) != 0:
            print(index, ', ')

        # if len(lmlist) != 0:
        #     vector = [lmlist[8][1] - lmlist[5][1], lmlist[8][2] - lmlist[5][2]]
            # vi76 = [lmlist[7][1] - lmlist[6][1], lmlist[7][2] - lmlist[6][2]]
            # vi50 = [lmlist[5][1] - lmlist[0][1], lmlist[5][2] - lmlist[0][2]]
            # d1 = np.dot(vi87, vi50)
            # d2 = np.dot(vi76, vi50)
            # print(vector)
            # if d2 < 0:
            #     print(2)
            # elif d1 < 0:
            #     print(1)
            # else:
            #     print(0)


        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
