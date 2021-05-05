import cv2
import numpy as np
import imutils
import time
from imutils import perspective
from imutils import contours
import depthai as dai
import streamlit as st
from scipy.spatial import distance as dist


# create depthai pipeline
def createPipeline():
    print('Creating Pipeline')

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define color camera
    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(640, 480)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Define a source - two mono (grayscale) cameras
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

    # MonoCamera
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    outputDepth = True
    outputRectified = False
    lrcheck = False
    subpixel = False

    # Create outputs
    xoutPreview = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()
    xoutSpatialData = pipeline.createXLinkOut()
    xinSpatialCalcConfig = pipeline.createXLinkIn()

    xoutPreview.setStreamName("preview")
    xoutDepth.setStreamName("depth")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # StereoDepth
    stereo.setOutputDepth(outputDepth)
    stereo.setOutputRectified(outputRectified)
    stereo.setConfidenceThreshold(255)

    stereo.setLeftRightCheck(lrcheck)
    stereo.setSubpixel(subpixel)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
    stereo.depth.link(spatialLocationCalculator.inputDepth)

    topLeft = dai.Point2f(0.4, 0.4)
    bottomRight = dai.Point2f(0.8, 0.8)

    spatialLocationCalculator.setWaitForConfigInput(False)
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    config.roi = dai.Rect(topLeft, bottomRight)
    spatialLocationCalculator.initialConfig.addROI(config)
    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
    camRgb.preview.link(xoutPreview.input)

    return pipeline, topLeft, bottomRight, config


# call base depth estimation to set base depth for height calculations
def baseDepthEstimation(pipeline, topLeft, bottomRight, config):
    st.warning('Please make sure the bounding box area is clear of objects.')
    col1, col2 = st.beta_columns(2)
    with col1:
        st.text('Preview window.')
    with col2:
        st.text('Click start to calibrate depth for avg. 25 frames')
        start = st.button('Start Calibration')

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        device.startPipeline()

        # Output queue will be used to get the depth frames from the outputs defined above
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

        color = (255, 255, 255)
        noFrames = 0
        frameST = col1.empty()
        frameST2 = col2.empty()
        baseDepth = 0.0
        DepthValue = 0.0
        count = False
        fontType = cv2.FONT_HERSHEY_TRIPLEX

        while True:
            inDepth = depthQueue.get()  # Blocking call, will wait until a new data has arrived
            inDepthAvg = spatialCalcQueue.get()  # Blocking call, will wait until a new data has arrived
            preview = device.getOutputQueue('preview').get()

            img = preview.getFrame()

            depthFrame = inDepth.getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

            spatialData = inDepthAvg.getSpatialLocations()
            for depthData in spatialData:
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[ 1 ], height=depthFrameColor.shape[ 0 ])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                # preview window info
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                cv2.putText(img, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20),
                            fontType, 0.5, color)
                cv2.putText(img, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35),
                            fontType, 0.5, color)
                cv2.putText(img, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50),
                            fontType, 0.5, color)

                # depth window info
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20),
                            fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35),
                            fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50),
                            fontType, 0.5, color)

                if start:
                    start = False
                    count = True
                    noFrames = 1

                if count:
                    DepthValue = (depthData.spatialCoordinates.z) / 10
                    baseDepth += DepthValue
                    print(baseDepth)
                    DepthValue = 0.0

                if noFrames == 25 and count == True:
                    # Base depth value calculation, dividing by no. of frames for the average
                    count = False
                    print('Frames count {:d}'.format(noFrames))
                    baseDepth = baseDepth / 25
                    print('base depth {:>2f}'.format(baseDepth))
                    st.text("Base Depth = {:>2f}".format(baseDepth))
                    cv2.destroyAllWindows()
                    return baseDepth

                with col1:
                    frameST.image(img, channels='BGR')
                with col2:
                    frameST2.image(depthFrameColor)

                noFrames += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def midpoint(ptA, ptB):
    return ((ptA[ 0 ] + ptB[ 0 ]) * 0.5, (ptA[ 1 ] + ptB[ 1 ]) * 0.5)


# start the main codes
def main(baseDepth):
    col1, col2 = st.beta_columns([ 2, 1 ])
    with col1:
        calc = st.button('Calculate Dimensions')
    with col2:
        st.text('Depth Map')

    with dai.Device(pipeline) as device:
        device.startPipeline()
        frameST = col1.empty()
        frameST2 = col2.empty()
        frameST3 = col2.empty()

        # Output queue will be used to get the depth frames from the outputs defined above
        prevQueue = device.getOutputQueue(name="preview", maxSize=8, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

        height = 0.0
        pTime = 0
        cTime = 0
        fps = 0
        DepthValue = 0.0
        color = (255, 255, 255)

        while True:
            inDepth = depthQueue.get()  # Blocking call, will wait until a new data has arrived
            inDepthAvg = spatialCalcQueue.get()
            # get the color camera image out of queue
            preview = prevQueue.get()
            img = preview.getFrame()
            org = img.copy()

            ################################################################################################
            depthFrame = inDepth.getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            spatialData = inDepthAvg.getSpatialLocations()
            ################################################################################################

            # perform blurring, edge detection, dilation and erode to find contours
            imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            imgCanny = cv2.Canny(imgGray, 50, 100)
            kernel = np.ones((5, 5))
            edged = cv2.dilate(imgCanny, kernel, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)

            # call find contours to get all contours in image
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts) > 0:
                cnts = contours.sort_contours(cnts)[ 0 ]
            # color for each edge in case of rectangular bounding box
            colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

            # loop over the contours individually
            for (i, c) in enumerate(cnts):

                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < 1000:
                    continue

                # we are looking for only one package, detect object #1 contours
                if (i + 1) == 1:
                    # compute the rotated bounding box of the contour, then draw the contours
                    box = cv2.minAreaRect(c)
                    box = cv2.boxPoints(box)
                    box = np.array(box, dtype="int")
                    cv2.drawContours(org, [ box ], -1, (0, 255, 0), 2)
                    # show the coordinates
                    print("Object #{}:".format(i + 1))

                    # order the points in the contour such that they appear
                    # in top-left, top-right, bottom-right, and bottom-left
                    # order, then draw the outline of the rotated bounding
                    rect = perspective.order_points(box)
                    # compute the center of the bounding box
                    cX = int(np.average(box[ :, 0 ]))
                    cY = int(np.average(box[ :, 1 ]))

                    print(rect.astype("int"))
                    print("")

                    # extract all the edges as tuple
                    (tl, tr, br, bl) = rect

                    # compute width
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)
                    # multiply by a constant we used while converting from pixel to actual breadth
                    breadth = (dist.euclidean((tlblX, tlblY), (trbrX, trbrY))) * 0.046
                    print(breadth)

                    # compute length
                    (tltrX, tltrY) = midpoint(tl, tr)
                    (blbrX, blbrY) = midpoint(bl, br)
                    # multiply by a constant we used while converting from pixel to actual length
                    length = (dist.euclidean((tltrX, tltrY), (blbrX, blbrY))) * 0.042
                    print(length)

                    cv2.line(org, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 0, 255), 2)
                    cv2.line(org, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 0), 2)

                    # loop over the original points and draw them
                    for ((x, y), color) in zip(rect, colors):
                        cv2.circle(org, (int(x), int(y)), 5, color, -1)
                        cv2.circle(org, (cX, cY), 5, color, 2, cv2.FILLED)

                    # draw the object num at the top-left corner
                    cv2.putText(org, "Object #{}".format(i + 1),
                                (int(rect[ 0 ][ 0 ] - 15), int(rect[ 0 ][ 1 ] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    # calculate depth for detected contours 'rect[][]'
                    for depthData in spatialData:
                        xmin = int(rect[ 0 ][ 0 ])
                        ymin = int(rect[ 0 ][ 1 ])
                        xmax = int(rect[ 2 ][ 0 ])
                        ymax = int(rect[ 2 ][ 1 ])

                        fontType = cv2.FONT_HERSHEY_TRIPLEX
                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                        cv2.putText(depthFrameColor, "Center", (cX, cY), fontType, 0.5, color)
                        cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm",
                                    (xmin + 10, ymin + 20),
                                    fontType, 0.5, color)

                        DepthValue = (depthData.spatialCoordinates.z) / 10
                        height = baseDepth - DepthValue

                cv2.putText(org, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # calculate FPS
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime

                with col1:
                    frameST.image(org, channels='BGR')
                    if calc and (i + 1) == 1:
                        st.text('Length[cm] = {:>20f}'.format(length))
                        st.text('Breadth[cm] = {:>20f}'.format(breadth))
                        st.text('Height[cm] = {:>20f}'.format(height))
                        volume = length * breadth * height
                        st.text('Volume[cubic-cm] = {:>20f}'.format(volume))
                        length = 0.0
                        breadth = 0.0
                        height = 0.0
                        volume = 0.0
                        calc = False

                with col2:
                    frameST2.image(edged)
                with col2:
                    frameST3.image(depthFrameColor)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Destroy all the windows
        cv2.destroyAllWindows()


pipeline, topLeft, bottomRight, config = createPipeline()

st.sidebar.title("OAK-D Warehouse Management")
option = st.sidebar.selectbox('Select', [ 'None', 'Base Depth Load', 'Measure Dimensions' ])

if option == 'None':
    st.sidebar.image('/home/sumit/PycharmProjects/Dimension_Measurement/Images/Measure.gif')

if option == 'Base Depth Load':
    st.sidebar.image('/home/sumit/PycharmProjects/Dimension_Measurement/Images/Calibrate.gif')
    baseDepth = baseDepthEstimation(pipeline, topLeft, bottomRight, config)
    print(baseDepth)

if option == 'Measure Dimensions':
    st.sidebar.image('/home/sumit/PycharmProjects/Dimension_Measurement/Images/MeasureDark.gif')
    # Enter depth value calculated in Base Depth Load option page
    baseDepth = st.sidebar.number_input('Base Depth Value:')
    main(baseDepth)
