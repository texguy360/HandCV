# import cv2
# import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), ])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate



    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors


def main():
    input_vectors = np.array(
        [
            [0.25193852186203003, 0.7692977786064148, 0.3237171173095703, 0.578625500202179, 0.3505420386791229,
             0.5040644407272339, 0.3638669550418854, 0.45671844482421875, 0.37281155586242676, 0.4153592586517334],
            [0.24933961033821106, 0.7638492584228516, 0.318428635597229, 0.573274552822113, 0.3470381200313568,
             0.5007309317588806, 0.36179953813552856, 0.4532376527786255, 0.37181422114372253, 0.4129672646522522],
            [0.2511405646800995, 0.7549431324005127, 0.3184404671192169, 0.5670883059501648, 0.3476926386356354,
             0.4945875406265259, 0.36158502101898193, 0.4472615122795105, 0.3705624043941498, 0.4067414402961731],
            [0.25300079584121704, 0.7444130182266235, 0.31868284940719604, 0.5604948401451111, 0.34759944677352905,
             0.48979651927948, 0.3622899055480957, 0.4430158734321594, 0.3727712035179138, 0.4025464653968811],

            [0.5359541177749634, 0.899268627166748, 0.58026123046875, 0.7421433329582214, 0.6145501136779785,
             0.7219982743263245, 0.6316317319869995, 0.7499102354049683, 0.6358980536460876, 0.7805922031402588],
            [0.5355429649353027, 0.8999814987182617, 0.5795508027076721, 0.7435020208358765, 0.6128666996955872,
             0.719552755355835, 0.6304481625556946, 0.7485392689704895, 0.6349132657051086, 0.779306948184967],
            [0.5332090854644775, 0.9023879766464233, 0.5765767097473145, 0.7441398501396179, 0.6096667647361755,
             0.7218216061592102, 0.6265652775764465, 0.7538039088249207, 0.6294001340866089, 0.7873400449752808],
            [0.5294579267501831, 0.9043954014778137, 0.5715833902359009, 0.7397657632827759, 0.6111900210380554,
             0.711862325668335, 0.6326424479484558, 0.7474554181098938, 0.6389825940132141, 0.7841379046440125],
            [0.5236472487449646, 0.9100868105888367, 0.5630386471748352, 0.7398107647895813, 0.6039892435073853,
             0.7043879628181458, 0.6293343305587769, 0.7363298535346985, 0.6411401629447937, 0.7715740203857422],
            [0.5248044729232788, 0.9096633195877075, 0.5649393796920776, 0.7377301454544067, 0.5993365049362183,
             0.6843036413192749, 0.6268956661224365, 0.6679976582527161, 0.6482567191123962, 0.6595897078514099],
            [0.5199020504951477, 0.9094165563583374, 0.5597013831138611, 0.7415160536766052, 0.6015750765800476,
             0.7070472240447998, 0.6261689066886902, 0.7341768741607666, 0.6381100416183472, 0.765846312046051],
            [0.516589343547821, 0.9123899936676025, 0.5589199066162109, 0.7400632500648499, 0.5926723480224609,
             0.6855558753013611, 0.6178646087646484, 0.669927716255188, 0.6372005343437195, 0.6615186929702759],
            [0.5121497511863708, 0.9076277017593384, 0.551276445388794, 0.7351791858673096, 0.5879266858100891,
             0.6823524236679077, 0.6132745146751404, 0.6661125421524048, 0.6326683163642883, 0.6574450135231018],


            [0.25207391381263733, 0.732280969619751, 0.32135704159736633, 0.5530893206596375, 0.3529872000217438,
             0.483390212059021, 0.3685600161552429, 0.43740713596343994, 0.3798524737358093, 0.3978192210197449],
            [0.2501662075519562, 0.7228423953056335, 0.3230522572994232, 0.5467614531517029, 0.356703519821167,
             0.4771071970462799, 0.3732253909111023, 0.4311860203742981, 0.38522830605506897, 0.3916011154651642],
            [0.24754802882671356, 0.7123114466667175, 0.3240247666835785, 0.5392216444015503, 0.3597675561904907,
             0.47133755683898926, 0.3780512511730194, 0.4256804287433624, 0.391924649477005, 0.38588035106658936],
            [0.24489322304725647, 0.6995850205421448, 0.3274460434913635, 0.5326871871948242, 0.36417096853256226,
             0.4639800786972046, 0.38205868005752563, 0.418022096157074, 0.3947073817253113, 0.37866029143333435],
            [0.24389693140983582, 0.6881539821624756, 0.32953688502311707, 0.5249399542808533, 0.365725040435791,
             0.4598803222179413, 0.38306817412376404, 0.4165644645690918, 0.394561231136322, 0.37994110584259033],
            [0.23940883576869965, 0.6765248775482178, 0.331582635641098, 0.51824951171875, 0.3693504333496094,
             0.4559922218322754, 0.387122243642807, 0.413699746131897, 0.3989405333995819, 0.3781530261039734],
            [0.24363543093204498, 0.6427454352378845, 0.3431381285190582, 0.4965827763080597, 0.3826615512371063,
             0.44092732667922974, 0.40279898047447205, 0.40176963806152344, 0.41728952527046204, 0.3688059449195862],
            [0.24345728754997253, 0.6248077154159546, 0.34474343061447144, 0.4885914623737335, 0.38619810342788696,
             0.43524324893951416, 0.4071538746356964, 0.3979584276676178, 0.42166587710380554, 0.3670971393585205],

        ]
    )

    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    learning_rate = 0.1

    neural_network = NeuralNetwork(learning_rate)

    training_error = neural_network.train(input_vectors, targets, 10000)

    weights = NeuralNetwork.weights()

    print(weights)

    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.savefig("cumulative_error.png")


if __name__ == '__main__':
    main()


#
#
# class Detector:
#     def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, track_confidence=0.5):
#         self.mode = mode
#         self.max_hands = max_hands
#         self.detection_confidence = detection_confidence
#         self.track_confidence = track_confidence
#         self.model_complexity = model_complexity
#
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_confidence,
#                                         self.track_confidence)
#         self.mpDraw = mp.solutions.drawing_utils
#         self.results = None
#
#     def find_hands(self, img, draw=True):
#         self.results = self.hands.process(img)
#
#         if self.results.multi_hand_landmarks:
#             for handLms in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
#         return img
#
#     def find_position(self, img, hand_number=0):
#         lmlist = []
#         if self.results.multi_hand_landmarks:
#             hand = self.results.multi_hand_landmarks[hand_number]
#             for id, lm in enumerate(hand.landmark):
#                 h, w, c = img.shape
#                 cx = int(lm.x * w)
#                 cy = int(lm.y * h)
#                 lmlist.append([id, cx, cy])
#         return lmlist
#
#     def index_vector(self, lmlist):
#         index_list = []
#         wi87 = [lmlist[8][1] - lmlist[7][1], lmlist[8][2] - lmlist[7][2]]
#         wi76 = [lmlist[7][1] - lmlist[6][1], lmlist[7][2] - lmlist[6][2]]
#         wi65 = [lmlist[6][1] - lmlist[5][1], lmlist[6][2] - lmlist[5][2]]
#         wi50 = [lmlist[5][1] - lmlist[0][1], lmlist[5][2] - lmlist[0][2]]
#         wi85 = [lmlist[8][1] - lmlist[5][1], lmlist[8][2] - lmlist[5][2]]
#         index_list.append([wi87, wi76, wi65, wi85, wi50])
#         return index_list
#
#
# def main():
#     cap = cv2.VideoCapture(0)
#     detector = Detector()
#
#     while True:
#         success, img = cap.read()
#         img = detector.find_hands(img)
#         lmlist = detector.find_position(img)
#         index_list = detector.index_vector(lmlist)
#
#         if len(index_list) != 0:
#             print(index_list)
#
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)
#
#
# if __name__ == '__main__':
#     main()
