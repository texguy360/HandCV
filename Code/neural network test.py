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

    def index_vector(self, lmlist):
        wi87 = [lmlist[8][1] - lmlist[7][1], lmlist[8][2] - lmlist[7][2]]
        wi76 = [lmlist[7][1] - lmlist[6][1], lmlist[7][2] - lmlist[6][2]]
        wi65 = [lmlist[6][1] - lmlist[5][1], lmlist[6][2] - lmlist[5][2]]
        wi50 = [lmlist[5][1] - lmlist[0][1], lmlist[5][2] - lmlist[0][2]]
        wi85 = [lmlist[8][1] - lmlist[5][1], lmlist[8][2] - lmlist[5][2]]


def main():
    cap = cv2.VideoCapture(0)
    detector = Detector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmlist = detector.find_position(img)



        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
