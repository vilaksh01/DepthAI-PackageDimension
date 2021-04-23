import cv2
import numpy as np
import depthai as dai
import time
import math
import handTrackModule as htm
from subprocess import call

# For windows system
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#
# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = cast(interface, POINTER(IAudioEndpointVolume))
# print(volume.GetVolumeRange())
# volume.SetMasterVolumeLevel(%use it to set vol, None)

pTime = 0
cTime = 0
detector = htm.HandDetector(detectConf=0.7)
color = (0, 0, 255)
minVol = 0
maxVol = 100
vol = 0
volBar = 400
# Defining a pipeline
p = dai.Pipeline()

# Define a source - color camera
camRgb = p.createColorCamera()
camRgb.setPreviewSize(640, 480)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(True)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Create output
xoutPreview = p.createXLinkOut()
xoutPreview.setStreamName("preview")

camRgb.preview.link(xoutPreview.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    #Start pipeline
    device.startPipeline()

    while True:
        #Get preview frames
        preview = device.getOutputQueue('preview').get()
        img = preview.getFrame()
        img = detector.findHands(img)
        lmList = detector.findPositions(img)

        if len(lmList) != 0:
            # print(lmList[4], lmList[8])
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 5, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            if length < 50:
                cv2.circle(img, (cx, cy), 10, (225, 0, 220), cv2.FILLED)
            # hand range 50 - 300
            # vol range 0 - 100
            vol = int(np.interp(length, [50, 300], [minVol, maxVol]))
            volBar = int(np.interp(length, [100, 250], [400, 150]))

            if vol >= minVol and vol <= maxVol:
                call(["amixer", "-D", "pulse", "sset", "Master", str(vol) + "%"])

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, volBar), (85, 400), (0, 255, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("image", img)
        cv2.waitKey(1)



