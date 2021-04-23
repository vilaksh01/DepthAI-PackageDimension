# Control your system functionalities using OAK-D vision systema and DepthAI realtime vision processing with Mediapipe AI models

## Hand tracking and hand landmark detection to detect gestures

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the BGR image to RGB before processing.
        self.results = self.hands.process(imgRGB)      # magic happens here
        # print(results.multi_hand_landmarks)  #check if your landmarks are detected or not

        if self.results.multi_hand_landmarks:
            for self.handMK in self.results.multi_hand_landmarks:            
                if draw:
                    # Draw the hand annotations on the image.
                    self.mpDraw.draw_landmarks(img, self.handMK, self.mpHands.HAND_CONNECTIONS)

        return img
        
<img src="https://google.github.io/mediapipe/images/mobile/hand_crops.png">
        
    def findPositions(self, img, handNo = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(self.handMK.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                # if id == 4:
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 40, 90), cv2.FILLED)
        # returns the list for hand landmarks
        return lmList

<img src="https://google.github.io/mediapipe/images/mobile/hand_landmarks.png">


## Reqirements:

- depthai	2.2.1.0
- mediapipe	0.8.3.1
- numpy	1.20.2
- opencv-python	4.5.1.48
- pip	21.0.1
- pkg-resources	0.0.0	
