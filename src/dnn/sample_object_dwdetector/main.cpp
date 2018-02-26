/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2015-2016 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////
#define _CRT_SECURE_NO_WARNINGS

#include <memory>
#include <thread>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

// Sample
#include <framework/DataPath.hpp>
#include <framework/SampleFramework.hpp>
#include <framework/Checks.hpp>

// CORE
#include <dw/core/Context.h>
#include <dw/core/Logger.h>

// Renderer
#include <dw/renderer/Renderer.h>

// SAL
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>

// IMAGE
#include <dw/image/FormatConverter.h>
#include <dw/image/ImageStreamer.h>

// DNN
#include <dw/object/Detector.h>
#include <dw/object/Tracker.h>
#include <dw/object/Clustering.h>

// Input/Output
#include <dnn_common/ISensorIO.hpp>
#include <dnn_common/SensorIOCuda.hpp>

#ifdef VIBRANTE
#include <dnn_common/SensorIONvmedia.hpp>
#endif

#define MAX_CAMERAS     2


//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------

dwContextHandle_t gSdk                              = DW_NULL_HANDLE;
dwSALHandle_t gSal                                  = DW_NULL_HANDLE;
dwRendererHandle_t gRenderer[MAX_CAMERAS]           = {DW_NULL_HANDLE};
dwRenderBufferHandle_t gLineBuffer[MAX_CAMERAS]     = {DW_NULL_HANDLE};
dwSensorHandle_t gCameraSensor[MAX_CAMERAS]         = {DW_NULL_HANDLE};
dwDNNHandle_t gDNN                                  = DW_NULL_HANDLE;
dwObjectDetectorHandle_t gDetector                  = DW_NULL_HANDLE;
dwObjectTrackerHandle_t gObjectTracker[MAX_CAMERAS] = {DW_NULL_HANDLE};
dwObjectClusteringHandle_t gClustering              = DW_NULL_HANDLE;

dwRect gScreenRectangle[MAX_CAMERAS];

dwImageCUDA *gYuvImage[MAX_CAMERAS];
dwImageCUDA *gRgbaImage[MAX_CAMERAS];
std::unique_ptr<ISensorIO> gSensorIO[MAX_CAMERAS];

uint32_t gCameraWidth    = 0U;
uint32_t gCameraHeight   = 0U;
uint32_t gNumCameras;

static const size_t gMaxNumProposals = 100U;
static const size_t gMaxNumObjects = 100U;
size_t gNumDetections[MAX_CAMERAS];
std::unique_ptr<dwObject[]> gDetectionList[MAX_CAMERAS];
size_t gNumTrackedObjects[MAX_CAMERAS];
std::unique_ptr<dwObject[]> gTrackedDetectionList[MAX_CAMERAS];
size_t gNumClusters[MAX_CAMERAS];
std::unique_ptr<dwObject[]> gClusters[MAX_CAMERAS];
size_t gNumMergedObjects[MAX_CAMERAS];
std::unique_ptr<dwObject[]> gMergedObjects[MAX_CAMERAS];

std::vector<std::pair<dwBox2D, std::string>> gDnnBoxList[MAX_CAMERAS];

cudaStream_t gCudaStream[MAX_CAMERAS] = {0};

//------------------------------------------------------------------------------

//#######################################################################################
void resizeWindowCallback(int width, int height) {
   gWindow->setWindowSize(width, height);
   for(uint32_t idx = 0; idx < gNumCameras; ++idx)
   {
    gScreenRectangle[idx].width = width / gNumCameras;
    gScreenRectangle[idx].height = height;
    gScreenRectangle[idx].x = idx * gScreenRectangle[idx].width;
    gScreenRectangle[idx].y = 0;
    dwRenderer_setRect(gScreenRectangle[idx], gRenderer[idx]);
   }
}

//#######################################################################################
void setupRenderer(dwRendererHandle_t &renderer, const dwRect &screenRectangle,
                   dwContextHandle_t dwSdk)
{
    CHECK_DW_ERROR( dwRenderer_initialize(&renderer, dwSdk) );

    float32_t rasterTransform[9];
    rasterTransform[0] = 1.0f;
    rasterTransform[3] = 0.0f;
    rasterTransform[6] = 0.0f;
    rasterTransform[1] = 0.0f;
    rasterTransform[4] = 1.0f;
    rasterTransform[7] = 0.0f;
    rasterTransform[2] = 0.0f;
    rasterTransform[5] = 0.0f;
    rasterTransform[8] = 1.0f;

    dwRenderer_set2DTransform(rasterTransform, renderer);
    float32_t boxColor[4] = {0.0f,1.0f,0.0f,1.0f};
    dwRenderer_setColor(boxColor, renderer);
    dwRenderer_setLineWidth(2.0f, renderer);
    dwRenderer_setRect(screenRectangle, renderer);
}

//#######################################################################################
void setupLineBuffer(dwRenderBufferHandle_t &lineBuffer, uint32_t maxLines, dwContextHandle_t dwSdk)
{
    dwRenderBufferVertexLayout layout;
    layout.posFormat   = DW_RENDER_FORMAT_R32G32_FLOAT;
    layout.posSemantic = DW_RENDER_SEMANTIC_POS_XY;
    layout.colFormat   = DW_RENDER_FORMAT_NULL;
    layout.colSemantic = DW_RENDER_SEMANTIC_COL_NULL;
    layout.texFormat   = DW_RENDER_FORMAT_NULL;
    layout.texSemantic = DW_RENDER_SEMANTIC_TEX_NULL;
    dwRenderBuffer_initialize(&lineBuffer, layout, DW_RENDER_PRIM_LINELIST, maxLines, dwSdk);
    dwRenderBuffer_set2DCoordNormalizationFactors((float32_t)gCameraWidth,
                                                  (float32_t)gCameraHeight, lineBuffer);
}


//#######################################################################################
void createVideoReplay(dwSensorHandle_t &salSensor,
                       uint32_t &cameraWidth,
                       uint32_t &cameraHeight,
                       uint32_t &cameraSiblings,
                       float32_t &cameraFrameRate,
                       dwImageType &imageType,
                       dwSALHandle_t sal,
                       const std::string &videoFName)
{
#ifdef VIBRANTE
    auto yuv2rgb = gArguments.get("yuv2rgb");
    std::string arguments = "video=" + videoFName + ",yuv2rgb=" + yuv2rgb;
#else
    std::string arguments = "video=" + videoFName;
#endif

    dwSensorParams params;
    params.parameters = arguments.c_str();
    params.protocol   = "camera.virtual";
    dwSAL_createSensor(&salSensor, params, sal);

    dwImageProperties cameraImageProperties;
    dwSensorCamera_getImageProperties(&cameraImageProperties,
                                      DW_CAMERA_PROCESSED_IMAGE,
                                      salSensor);
    dwCameraProperties cameraProperties;
    dwSensorCamera_getSensorProperties(&cameraProperties, salSensor);
    cameraSiblings = cameraProperties.siblings;
    cameraFrameRate = cameraProperties.framerate;

    cameraWidth = cameraImageProperties.width;
    cameraHeight = cameraImageProperties.height;
    imageType = cameraImageProperties.type;
}

//#######################################################################################
void initDriveworks()
{
    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams = {};

#ifdef VIBRANTE
    sdkParams.eglDisplay = gWindow->getEGLDisplay();
#endif

    dwInitialize(&gSdk, DW_VERSION, &sdkParams);
}

//#######################################################################################
void initRenderer(uint32_t cameraIdx, int32_t offsetX)
{
    gScreenRectangle[cameraIdx].height  = gWindow->height();
    gScreenRectangle[cameraIdx].width   = gWindow->width() / gNumCameras;
    gScreenRectangle[cameraIdx].x = offsetX;
    gScreenRectangle[cameraIdx].y = 0;

    // init renderer
    uint32_t maxLines = 20000;
    setupRenderer(gRenderer[cameraIdx], gScreenRectangle[cameraIdx], gSdk);
    setupLineBuffer(gLineBuffer[cameraIdx], maxLines, gSdk);
}

//#######################################################################################
void initCameraNumbers()
{
    gNumCameras = std::stoi(gArguments.get("video-count"));

    if(gNumCameras <= 0 || gNumCameras > 2){
        std::cerr << "ERROR: more than 2 cameras are not allowed\n";
        exit(EXIT_FAILURE);
    }
}

//#######################################################################################
void initCameras(int32_t cameraIdx)
{
    // create sensor abstraction layer
    if (gSal == DW_NULL_HANDLE)
        dwSAL_initialize(&gSal, gSdk);

    // create GMSL Camera interface
    uint32_t cameraSiblings   = 0U;
    float32_t cameraFramerate = 0.0f;
    dwImageType imageType;

    std::string video_arg = "video" + std::to_string(cameraIdx + 1);
    createVideoReplay(gCameraSensor[cameraIdx], gCameraWidth, gCameraHeight, cameraSiblings,
                      cameraFramerate, imageType, gSal, gArguments.get(video_arg.c_str()));

    std::cout << "Camera image with " << gCameraWidth << "x" << gCameraHeight << " at "
              << cameraFramerate << " FPS" << std::endl;

    // Update window size
    const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    float32_t aspectRatio = static_cast<float32_t>(gCameraHeight) /
            static_cast<float32_t>(gCameraWidth);

    uint32_t windowWidth = std::min(static_cast<int32_t>(gCameraWidth * gNumCameras), mode->width);
    uint32_t windowHeight = static_cast<uint32_t>(std::round(static_cast<float32_t>(windowWidth / gNumCameras) * aspectRatio));
    gWindow->setWindowSize(windowWidth, windowHeight);

#ifdef VIBRANTE
    gSensorIO[cameraIdx].reset(new SensorIONvmedia(gSdk, 0, gCameraSensor[cameraIdx], gCameraWidth, gCameraHeight));
#else
    gSensorIO[cameraIdx].reset(new SensorIOCuda(gSdk, 0, gCameraSensor[cameraIdx], gCameraWidth, gCameraHeight));
#endif

    gRun = gRun && dwSensor_start(gCameraSensor[cameraIdx]) == DW_SUCCESS;
}

//#####################################################################################
// read each line from the file
bool readFilelist(std::vector<std::string> &vec, const std::string &fn)
{
    std::ifstream file(fn);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty())
                vec.push_back(line);
        }
        file.close();
        return true;
    } else
        return false;
}

//#####################################################################################
// parse the variable in a line
template<class T>
bool getValFromString(T &val, std::string str, std::string key)
{
    size_t pos = str.find(key);
    if (pos != std::string::npos) {
        size_t posSpace = str.find(" ");
        std::stringstream(str.substr(posSpace + 1)) >> val;
        return true;
    }
    return false;
}

//#####################################################################################
// parse tracker parameters
bool readTrackerParam(dwObjectFeatureTrackerParams *featureTrackerParams,
                      dwObjectTrackerParams *trackerParams, std::string filename)
{
    std::vector<std::string> params;
    if (!readFilelist(params, filename))
        return false;
    else {
        for (size_t i = 0; i < params.size(); i++) {
            bool result = false;
            if (!result)
                // max feature count
                result = getValFromString<uint32_t>(featureTrackerParams->maxFeatureCount,
                                                    params[i], "maxFeatureCount");
            if (!result)
                // Lucas Kanade feature iterations
                result = getValFromString<uint32_t>(featureTrackerParams->iterationsLK,
                                                    params[i], "iterationsLK");
            if (!result)
                // Lucas Kanade feature window size
                result = getValFromString<uint32_t>(featureTrackerParams->windowSizeLK,
                                                    params[i], "windowSizeLK");
            if (!result)
                // Max feature count per box
                result = getValFromString<uint32_t>(trackerParams->maxFeatureCountPerBox,
                                                    params[i], "maxFeatureCountPerObject");
            if (!result)
                // Max box image scale
                result = getValFromString<float32_t>(trackerParams->maxBoxImageScale,
                                                     params[i], "maxObjectImageScale");
            if (!result)
                // Min box image scale
                result = getValFromString<float32_t>(trackerParams->minBoxImageScale,
                                                     params[i], "minObjectImageScale");
        }
        return true;
    }
}

//#######################################################################################
void initTracker()
{
    dwObjectFeatureTrackerParams featureTrackerParams;
    dwObjectTrackerParams objectTrackerParams;

    dwObjectTracker_initDefaultParams(&featureTrackerParams, &objectTrackerParams, 1U);
    if (!readTrackerParam(&featureTrackerParams, &objectTrackerParams, gArguments.get("tracker"))) {
        std::cout << "fail to open tracker config file, use default values\n";
    }

    objectTrackerParams.maxNumObjects = gMaxNumObjects;

    // Get image properties from the camera
    dwImageProperties imageProperties;
    dwSensorCamera_getImageProperties(&imageProperties, DW_CAMERA_PROCESSED_IMAGE, gCameraSensor[0]);

    // Initialize object tracker with the parameters
    for (uint32_t camIdx = 0U; camIdx < gNumCameras; ++camIdx) {
        dwObjectTracker_initialize(&gObjectTracker[camIdx], gSdk, &imageProperties, &featureTrackerParams,
                                   &objectTrackerParams, 1U);
    }

    // Initialize clustering
    dwObjectClusteringParams clusteringParams;
    clusteringParams.algorithm = DW_CLUSTERING_DBSCAN;
    clusteringParams.enableATHRFilter = DW_TRUE;
    clusteringParams.thresholdATHR = 60.0f;
    clusteringParams.dbScanClusteringParams.epsilon = 0.8f;
    clusteringParams.dbScanClusteringParams.minBoxes = 4U;
    clusteringParams.dbScanClusteringParams.minSumOfConfidences = 0.0f;
    clusteringParams.maxClusters = gMaxNumObjects;
    clusteringParams.maxProposals = gMaxNumProposals;
    dwObjectClustering_initialize(&gClustering, gSdk, &clusteringParams);
}

//#######################################################################################
void initDetector()
{
    dwStatus status = DW_SUCCESS;
    std::string tensorRTModel = gArguments.get("tensorRT_model");
    std::cout << "Initializing TensorRT Network: " << tensorRTModel.c_str() << std::endl;
    status = dwDNN_initializeTensorRTFromFile(&gDNN, gSdk, tensorRTModel.c_str());

    if (DW_SUCCESS != status) {
        gRun = false;
        return;
    }

    dwObjectDetectorParams detectorParams;
    dwObjectDetectorDNNParams dnnParams;
    dwObjectDetector_initDefaultParams(&dnnParams, &detectorParams);

    dnnParams.maxProposalsPerClass = gMaxNumProposals;
    // DNN loads metadata automatically from json file stored next to the dnn model,
    // with the same name but additional .json extension if present.
    // Otherwise, the metadata will be filled with default values and the dataconditioner paramaters
    // should be filled manually.
    dwDNNMetaData metadata;
    dwDNN_getMetaData(&metadata, gDNN);
    dnnParams.dataConditionerParams = metadata.dataConditionerParams;

    // Set output blob names for the network.
    dnnParams.coverageBlobName = "coverage";
    dnnParams.boundingBoxBlobName = "bboxes";

    // Set maximum number of images to two as we will run detector on two images:
    // Mode 1: run detector on two regions on a single image.
    // Mode 2: run detector on two separate images.
    detectorParams.maxNumImages = 2U;

    if(gNumCameras == 1) {
        // Run detection on two region of interests on the same image: zoomed and not-zoomed.
        // - Therefore, the results of detections coming from these images should be merged.
        detectorParams.enableFuseObjects = DW_TRUE;
    }
    else{
        // Run detection on two independent images.
        detectorParams.enableFuseObjects = DW_FALSE;
    }

    // Create two region of interests
    dwBlobSize networkInputBlobSize;
    dwDNN_getInputSize(&networkInputBlobSize, 0U, gDNN);

    // Get the region of interest as big as the network input but centralized
    dwRect roiFull = {static_cast<int32_t>(gCameraWidth - networkInputBlobSize.width)/2,
                      static_cast<int32_t>(gCameraHeight - networkInputBlobSize.height)/2,
                      networkInputBlobSize.width, networkInputBlobSize.height};

    // Get a smaller region of interest in the center
    dwRect roiZoomed = {static_cast<int32_t>(gCameraWidth - networkInputBlobSize.width/2)/2,
                        static_cast<int32_t>(gCameraHeight - networkInputBlobSize.height/2)/2,
                        networkInputBlobSize.width / 2, networkInputBlobSize.height / 2};

    detectorParams.ROIs[0] = roiFull;
    if(gNumCameras == 1){
        detectorParams.ROIs[1] = roiZoomed;
    }
    else{
        // Run detector on same regions (full resolution) on both images.
        detectorParams.ROIs[1] = roiFull;
    }

    // Since both images are the same, there is no extra transformation needed for the outputs.
    dwTransformation2D transformationIdentity = {{1.0f, 0.0f, 0.0f,
                                                  0.0f, 1.0f, 0.0f,
                                                  0.0f, 0.0f, 1.0f}};

    detectorParams.transformations[0] = transformationIdentity;
    detectorParams.transformations[1] = transformationIdentity;

    dwObjectDetector_initialize(&gDetector, gSdk, gDNN, &dnnParams, &detectorParams);

    // Initialize all object lists
    for (uint32_t camIdx = 0U; camIdx < gNumCameras; ++camIdx) {
        gDetectionList[camIdx].reset(new dwObject[gMaxNumProposals]);
        gMergedObjects[camIdx].reset(new dwObject[gMaxNumObjects]);
        gClusters[camIdx].reset(new dwObject[gMaxNumObjects]);
        gTrackedDetectionList[camIdx].reset(new dwObject[gMaxNumObjects]);
    }
}


//#######################################################################################
void init()
{
    initDriveworks();

    initCameraNumbers();

    for(uint32_t idx = 0; idx < gNumCameras; ++idx)
        initCameras(idx);

    for(uint32_t idx = 0; idx < gNumCameras; ++idx)
        initRenderer(idx, idx * gWindow->width() / 2);

    initDetector();
    initTracker();
}

//#######################################################################################
void release()
{
    dwObjectDetector_release(&gDetector);

    for (uint32_t camIdx = 0; camIdx < gNumCameras; ++camIdx) {
        if (gCudaStream[camIdx]) {
            cudaStreamDestroy(gCudaStream[camIdx]);
        }
    }

    // Release detector
    dwObjectDetector_release(&gDetector);
    dwDNN_release(&gDNN);

    // release sensors
    for (uint32_t camIdx = 0; camIdx < gNumCameras; ++camIdx) {
        dwSensor_stop(gCameraSensor[camIdx]);
        dwSAL_releaseSensor(&gCameraSensor[camIdx]);

        dwRenderBuffer_release(&gLineBuffer[camIdx]);
        dwRenderer_release(&gRenderer[camIdx]);

        // release used objects in correct order
        gSensorIO[camIdx].reset();
    }

    for (uint32_t camIdx = 0U; camIdx < gNumCameras; ++camIdx)
        dwObjectTracker_release(&gObjectTracker[camIdx]);

    dwObjectClustering_release(&gClustering);
    dwSAL_release(&gSal);
    dwRelease(&gSdk);
    dwLogger_release();
}



//#######################################################################################
int main(int argc, const char **argv)
{

    const ProgramArguments arguments = ProgramArguments(
        {
            ProgramArguments::Option_t("tracker",
                                       (DataPath::get() +
                                        std::string{"/samples/tracker/config.txt"}).c_str()),
            ProgramArguments::Option_t("tensorRT_model",
                                        (DataPath::get() +
                                         std::string{"/samples/detector/tensorRT_model.bin"})
                                            .c_str()),
            ProgramArguments::Option_t("video1",
                                       (DataPath::get() +
                                        std::string{"/samples/sfm/triangulation/video_0.h264"})
                                           .c_str()),

            ProgramArguments::Option_t("video2",
                                       (DataPath::get() +
                                        std::string{"/samples/sfm/triangulation/video_1.h264"})
                                            .c_str()),

            ProgramArguments::Option_t("video-count", "1"),

#ifdef VIBRANTE
            ProgramArguments::Option_t("yuv2rgb", "cuda"),
#endif
        });

    // init framework
    initSampleApp(argc, argv, &arguments, NULL, 1280, 800);

    // set window resize callback
    gWindow->setOnResizeWindowCallback(resizeWindowCallback);

    // init driveworks
    init();

    typedef std::chrono::high_resolution_clock myclock_t;
    typedef std::chrono::time_point<myclock_t> timepoint_t;
    timepoint_t lastUpdateTime = myclock_t::now();
    uint32_t frameCount = 0;

    dwImageGL *frameGL[MAX_CAMERAS] = {nullptr, nullptr};
    dwStatus result;
    bool endStream = false;

    // main loop
    while (gRun && !gWindow->shouldClose()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        std::this_thread::yield();

        // slow down application to run with 30FPS
        auto timeSinceUpdate = std::chrono::duration_cast<std::chrono::milliseconds>
                               (myclock_t::now() - lastUpdateTime);
        if (timeSinceUpdate < std::chrono::milliseconds(33))
        {
            std::this_thread::sleep_for(std::chrono::microseconds(std::chrono::milliseconds(33) -
                                                                  timeSinceUpdate));
        }

        lastUpdateTime = myclock_t::now();
        for (uint32_t camIdx = 0U; camIdx < gNumCameras; ++camIdx) {
            if (frameGL[camIdx]) {
                gSensorIO[camIdx]->releaseGLRgbaFrame();
                gSensorIO[camIdx]->releaseFrame();
                frameGL[camIdx] = nullptr;
            }

            result = gSensorIO[camIdx]->getFrame();

            if (result == DW_END_OF_STREAM) {
                std::cout << "Camera reached end of stream" << std::endl;
                dwSensor_reset(gCameraSensor[camIdx]);
                dwObjectTracker_reset(gObjectTracker[camIdx]);
                dwObjectClustering_reset(gClustering);
                endStream = true;
                continue;
            } else if (result != DW_SUCCESS) {
                std::cerr << "Cannot read frame: " << dwGetStatusName(result) << std::endl;
                gRun = false;
                endStream = true;
                break;
            }

            gRgbaImage[camIdx] = gSensorIO[camIdx]->getCudaRgba();
            gYuvImage[camIdx] = gSensorIO[camIdx]->getCudaYuv();
            frameGL[camIdx] = gSensorIO[camIdx]->getGlRgbaFrame();
            if (frameGL[camIdx]) {
                dwRenderer_renderTexture(frameGL[camIdx]->tex, frameGL[camIdx]->target, gRenderer[camIdx]);
            }
        }

        if (endStream) {
            endStream = false;
            continue;
        }

        // Detect, track and render objects
        {
            for (uint32_t camIdx = 0U; camIdx < gNumCameras; ++camIdx) {
                gDnnBoxList[camIdx].clear();
            }

            dwImageCUDA *images[2] = {gRgbaImage[0], gRgbaImage[0]};
            if(gNumCameras == 2){
                images[1] = gRgbaImage[1];
            }

            // Run detection
            dwObjectDetector_inferDeviceAsync(images, 2U, gDetector);
            dwObjectDetector_interpretHost(2U, gDetector);

            for (uint32_t camIdx = 0U; camIdx < gNumCameras; ++camIdx) {
                // Run tracking
                dwObjectTracker_featureTrackDeviceAsync(gYuvImage[camIdx], gObjectTracker[camIdx]);
                dwObjectTracker_boxTrackHost(gTrackedDetectionList[camIdx].get(), &gNumTrackedObjects[camIdx],
                                             gMergedObjects[camIdx].get(), gNumMergedObjects[camIdx], 0U,
                                             gObjectTracker[camIdx]);

                memcpy(gMergedObjects[camIdx].get(), gTrackedDetectionList[camIdx].get(),
                    gNumTrackedObjects[camIdx] * sizeof(dwObject));

                gNumMergedObjects[camIdx] = gNumTrackedObjects[camIdx];

                // Get detected object at index camIdx (if gNumCameras == 1 then objects
                // from two images are fused)
                dwObjectDetector_getDetectedObjects(gDetectionList[camIdx].get(), &gNumDetections[camIdx],
                                                    camIdx, 0U, gDetector);
                dwObjectClustering_cluster(gClusters[camIdx].get(), &gNumDetections[camIdx],
                                           gDetectionList[camIdx].get(),
                                           gNumDetections[camIdx], gClustering);

                const dwObject *toBeMerged[2] = {gTrackedDetectionList[camIdx].get(),
                                                 gClusters[camIdx].get()};
                const size_t sizes[2] = {gNumTrackedObjects[camIdx], gNumDetections[camIdx]};
                dwObject_merge(gMergedObjects[camIdx].get(), &gNumMergedObjects[camIdx], gMaxNumObjects,
                               toBeMerged, sizes, 2U, 0.2f, 0.2f, gSdk);


                for (uint32_t objIdx = 0U; objIdx < gNumTrackedObjects[camIdx]; ++objIdx) {
                    const dwObject &obj = gTrackedDetectionList[camIdx][objIdx];
                    dwBox2D box;
                    box.x = static_cast<int32_t>(std::round(obj.box.x));
                    box.y = static_cast<int32_t>(std::round(obj.box.y));
                    box.width = static_cast<int32_t>(std::round(obj.box.width));
                    box.height = static_cast<int32_t>(std::round(obj.box.height));

                    gDnnBoxList[camIdx].push_back(std::make_pair(box, std::to_string(obj.objectId)));
                }

                drawBoxesWithLabels(gDnnBoxList[camIdx], static_cast<float32_t>(gCameraWidth),
                          static_cast<float32_t>(gCameraHeight), gLineBuffer[camIdx], gRenderer[camIdx]);
            }
        }

        for (uint32_t camIdx = 0U; camIdx < gNumCameras; ++camIdx) {
            gSensorIO[camIdx]->releaseCudaYuv();
            gSensorIO[camIdx]->releaseCudaRgba();
        }

        gWindow->swapBuffers();
        frameCount++;
        CHECK_GL_ERROR();
    }

    for (uint32_t camIdx = 0U; camIdx < gNumCameras; ++camIdx) {
        if (frameGL[camIdx]) {
            gSensorIO[camIdx]->releaseGLRgbaFrame();
            gSensorIO[camIdx]->releaseFrame();
        }
    }

    release();

    // release framework
    releaseSampleApp();

    return 0;
}
