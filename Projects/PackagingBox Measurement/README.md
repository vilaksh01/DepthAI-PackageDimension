# PackagingBox Measurement

## Dimensioning and Weighing Solutions using Depth AI

In the world of warehousing, `space = money` and `high labour cost = lesser profits.` Warehouse problems can affect the speed, efficiency, and productivity of either one particular warehouse operation or the entire chain of processes that are linked with it. Packaging operations have always been a real time-thief in warehouses and often almost completely manual. Often it is a real bottleneck in the warehouse flow. A half-hour before the trucks leave the warehouse, there is usually chaos in the packaging area with staff running into each other, looking for the right boxes and packing material to finish in time. This also affects the quality with the mixing of the goods from different orders as a result. Choosing the right package dimension size hep you to prevent costs of unncessary filling materials. 
<pre>
<img src="https://fwlogistics.com/wp-content/uploads/2018/11/20500.jpg" height="280" width="350"> | <img src="https://st4.depositphotos.com/3922387/19990/i/450/depositphotos_199907606-stock-photo-young-asian-man-using-tape.jpg" height="280" width="350">
</pre>

## Operation packaging efficiency is direct result of:
- Correct product packaging dimension measurement, combined with
- Right packaging materials and
- Operational procedures,

These factors, when implemented diligently, can help you manage packages by reducing damages and satisfy your customers through high operational standards. 
From small boxes to large pallets, our solution enables logistics companies to better manage their inventory and shipments.

## Scope of the project:
- Using OAK-D 3D Stereo Vision camera for capturing measurements of package, box or parcel. 
- Measure how much storage space the object would need or determine what size of the carton to use for shipment, estimate shipping costs.
- Use AI-enabled add-ons such as open, leaking, and broken box detection and packaging optimization to improve workforce efficiency and safety.
<pre> 
<img src="https://cdn.shopify.com/s/files/1/0106/8325/2802/products/OAK-D_1024x1024@2x.jpg?v=1612573255" height="220" width="220"> | <img src="https://cdn.shopify.com/s/files/1/0106/8325/2802/products/IMG_2984_1024x1024@2x.jpg?v=1612573255" height="220" width="220"> | <img src="https://cdn.shopify.com/s/files/1/0106/8325/2802/products/ASDFAWEFAGDVASVqdawfaff2353_1024x1024@2x.jpg?v=1612573255" height="220" width="220">
</pre>

## Technology stacks used <img src="https://luxonis.com/img/brand/depthai-logo-with-text.png" width="10%" height="5%" /> | <img src="https://blog.streamlit.io/content/images/size/w1000/2021/03/logomark-color.png" width="10%" height="5%" /> | <img src="https://opencv.org/wp-content/uploads/2020/07/cropped-OpenCV_logo_white_600x.png" width="5%" />

- DepthAI SDK for OAK-D vision camera inferencing and Spatial Data.
- Streamlit for web based dashboard for one button click dimension measuremnt and all metrics calculations.
- OpenCV for all image processing and feature extaction.

## DepthAI brings realtime Spatial AI to your product
DepthAI is a platform built around the Myriad X to combine depth perception, object detection (neural inference), and object tracking that gives you this power in a simple, easy-to-use Python API and a plug/play System on Module (SoM) with Open-Source hardware.

## Project Instructions:
The project has mainly four parts:
1. DepthAI Pipeline configuration for depth and RGB image data
2. Base Depth Estimation
3. Package Contour Detection
4. Package dimension and volume estimation

# 1. DepthAI Pipeline configuration for depth and RGB image data

```Python
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
```


