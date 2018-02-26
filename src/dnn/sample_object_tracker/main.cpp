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
#include <dnn_common/DNNInference.hpp>

// TRACKER
#include <dw/features/BoxTracker2D.h>
#include "FeatureTracker2D.hpp"

// Input/Output
#include <dnn_common/ISensorIO.hpp>
#include <dnn_common/SensorIOCuda.hpp>

#ifdef VIBRANTE
#include <dnn_common/SensorIONvmedia.hpp>
#endif

//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------

dwContextHandle_t gSdk                   = DW_NULL_HANDLE;
dwSALHandle_t gSal                       = DW_NULL_HANDLE;
dwRendererHandle_t gRenderer             = DW_NULL_HANDLE;
dwRenderBufferHandle_t gLineBuffer       = DW_NULL_HANDLE;
dwSensorHandle_t gCameraSensor           = DW_NULL_HANDLE;

dwRect gScreenRectangle;

dwImageCUDA *gYuvImage;
dwImageCUDA *gRgbaImage;
std::unique_ptr<ISensorIO> gSensorIO;

uint32_t gCameraWidth    = 0U;
uint32_t gCameraHeight   = 0U;
uint32_t gDetectInterval = 1;

std::unique_ptr<DNNInference> gDnnInference;
std::vector<dwBox2D> gDnnBoxList;

struct featureTrackerParams
{
    uint32_t maxFeatureCount;
    uint32_t iterationsLK;
    uint32_t windowSizeLK;
} gFeatureTrackerParams;

uint32_t gHistoryCapacity = 10U;

dwBoxTracker2DParams gBoxTrackerParams;

cudaStream_t gCudaStream = 0;
dwBoxTracker2DHandle_t gBoxTracker;
std::unique_ptr<FeatureTracker2D> gFeatureTracker;
std::vector<float32_t> gCurFeatureLocations;
std::vector<float32_t> gPreFeatureLocations;
std::vector<dwFeatureStatus> gFeatureStatuses;
//------------------------------------------------------------------------------

//#######################################################################################
void setupRenderer(dwRendererHandle_t &renderer, dwContextHandle_t dwSdk)
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
    float32_t boxColor[4] = {1.0f,0.0f,0.0f,1.0f};
    dwRenderer_setColor(boxColor, renderer);
    dwRenderer_setLineWidth(2.0f, renderer);
    dwRenderer_setRect(gScreenRectangle, renderer);
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

    cameraWidth = cameraImageProperties.width;
    cameraHeight = cameraImageProperties.height;
    imageType = cameraImageProperties.type;
    cameraFrameRate = cameraProperties.framerate;
    cameraSiblings = cameraProperties.siblings;
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
void initRenderer()
{
    gScreenRectangle.height  = gWindow->height();
    gScreenRectangle.width   = gWindow->width();
    gScreenRectangle.x = 0;
    gScreenRectangle.y = 0;

    // init renderer
    uint32_t maxLines = 20000;
    setupRenderer(gRenderer, gSdk);
    setupLineBuffer(gLineBuffer, maxLines, gSdk);
}

//#######################################################################################
void initCameras()
{
    // create sensor abstraction layer
    dwSAL_initialize(&gSal, gSdk);

    // create GMSL Camera interface
    uint32_t cameraSiblings   = 0U;
    float32_t cameraFramerate = 0.0f;
    dwImageType imageType;

    std::string yuv2rgb_arg;
    if (gArguments.has("yuv2rgb")) {
        yuv2rgb_arg = gArguments.get("yuv2rgb");
    }
    createVideoReplay(gCameraSensor, gCameraWidth, gCameraHeight, cameraSiblings,
                      cameraFramerate, imageType, gSal, gArguments.get("video"));

    std::cout << "Camera image with " << gCameraWidth << "x" << gCameraHeight << " at "
              << cameraFramerate << " FPS" << std::endl;

#ifdef VIBRANTE
    gSensorIO.reset(new SensorIONvmedia(gSdk, 0, gCameraSensor, gCameraWidth, gCameraHeight));
#else
    gSensorIO.reset(new SensorIOCuda(gSdk, 0, gCameraSensor, gCameraWidth, gCameraHeight));
#endif

    gRun = gRun && dwSensor_start(gCameraSensor) == DW_SUCCESS;
}

//#######################################################################################
void initDNN()
{
    gDnnInference.reset(new DNNInference(gSdk));

    std::string tensorRTModel = gArguments.get("tensorRT_model");
    std::cout << "Initializing TensorRT Network: " << tensorRTModel.c_str() << std::endl;
    gDnnInference->buildFromTensorRT(tensorRTModel.c_str());
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
bool readTrackerParam(std::string filename)
{
    std::vector<std::string> params;
    if (!readFilelist(params, filename))
        return false;
    else {
        for (size_t i = 0; i < params.size(); i++) {
            bool result = false;
            if (!result)
                // max feature count
                result = getValFromString<uint32_t>(gFeatureTrackerParams.maxFeatureCount,
                                                    params[i], "maxFeatureCount");
            if (!result)
                // Lucas Kanade feature iterations
                result = getValFromString<uint32_t>(gFeatureTrackerParams.iterationsLK,
                                                    params[i], "iterationsLK");
            if (!result)
                // Lucas Kanade feature window size
                result = getValFromString<uint32_t>(gFeatureTrackerParams.windowSizeLK,
                                                    params[i], "windowSizeLK");
            if (!result)
                // Detection interval
                result = getValFromString<uint32_t>(gDetectInterval,
                                                    params[i], "detectInterval");
            if (!result)
                // Max box count
                result = getValFromString<uint32_t>(gBoxTrackerParams.maxBoxCount,
                                                    params[i], "maxObjectCount");
            if (!result)
                // Max feature count per box
                result = getValFromString<uint32_t>(gBoxTrackerParams.maxFeatureCountPerBox,
                                                    params[i], "maxFeatureCountPerObject");
            if (!result)
                // Max box image scale
                result = getValFromString<float64_t>(gBoxTrackerParams.maxBoxImageScale,
                                                     params[i], "maxObjectImageScale");
            if (!result)
                // Min box image scale
                result = getValFromString<float64_t>(gBoxTrackerParams.minBoxImageScale,
                                                     params[i], "minObjectImageScale");
            if (!result)
                // Similarity threshold
                result = getValFromString<float64_t>(gBoxTrackerParams.similarityThreshold,
                                                     params[i], "similarityThreshold");
            if (!result)
                // Group Threshold
                result = getValFromString<uint32_t>(gBoxTrackerParams.groupThreshold,
                                                    params[i], "groupThreshold");
            if (!result)
                // Non-existing field
                std::cout << "invalid entry " << params[i] << " in the tracker config file\n";
        }
        return true;
    }
}

//#######################################################################################
void splitTrackedBoxInfo(std::vector<dwBox2D> &dnnBoxList, std::vector<uint32_t> &boxIds,
                         const dwTrackedBox2D* boxList, size_t boxListSize)
{
    dnnBoxList.clear();
    dnnBoxList.resize(boxListSize, dwBox2D{});
    boxIds.resize(boxListSize);
    for (uint32_t i = 0; i < boxListSize; i++) {
        dnnBoxList[i].x = boxList[i].box.x;
        dnnBoxList[i].y = boxList[i].box.y;
        dnnBoxList[i].width   = boxList[i].box.width;
        dnnBoxList[i].height  = boxList[i].box.height;
        boxIds[i] = boxList[i].id;
    }
}

//#########################################################################################
void getFeatures(std::vector<dwBox2D> &featureBoxes,
                 const size_t nFeatures, const float32_t *featureLocations)
{
    featureBoxes.resize(nFeatures);
    for (size_t i = 0; i < nFeatures; i++) {
        featureBoxes[i].x = static_cast<uint32_t>(featureLocations[2 * i]);
        featureBoxes[i].y = static_cast<uint32_t>(featureLocations[2 * i + 1]);
        featureBoxes[i].width   = 1;
        featureBoxes[i].height  = 1;
    }
}

//#######################################################################################
void initTracker()
{
    // init parameters with default values
    gFeatureTrackerParams.maxFeatureCount = 2000;
    gFeatureTrackerParams.iterationsLK    = 10;
    gFeatureTrackerParams.windowSizeLK    = 8;
    dwBoxTracker2D_initParams(&gBoxTrackerParams);

    // parse box tracker parameters
    if (!readTrackerParam(gArguments.get("tracker"))) {
        std::cout << "fail to open tracker config file, use default values\n";
    }

    // init box tracker
    dwBoxTracker2D_initialize(&gBoxTracker, &gBoxTrackerParams, static_cast<int32_t>(gCameraWidth),
                              static_cast<int32_t>(gCameraHeight), gSdk);
    // init feature tracker
    gFeatureTracker.reset(new FeatureTracker2D(gSdk, gCudaStream,
                                               gFeatureTrackerParams.maxFeatureCount, gHistoryCapacity,
                                               gFeatureTrackerParams.iterationsLK,
                                               gFeatureTrackerParams.windowSizeLK,
                                               gCameraWidth, gCameraHeight, gCameraHeight/2));
    // init feature storage
    gCurFeatureLocations.resize(2 * gFeatureTrackerParams.maxFeatureCount);
    gPreFeatureLocations.resize(2 * gFeatureTrackerParams.maxFeatureCount);

    gFeatureStatuses.resize(gFeatureTrackerParams.maxFeatureCount);
}

//#######################################################################################
void init()
{
    initDriveworks();

    initCameras();

    initRenderer();

    initDNN();

    initTracker();
}

//#######################################################################################
void release()
{
    if (gCudaStream) {
        cudaStreamDestroy(gCudaStream);
    }
    // release sensors
    dwSensor_stop(gCameraSensor);
    dwSAL_releaseSensor(&gCameraSensor);

    dwRenderBuffer_release(&gLineBuffer);
    dwRenderer_release(&gRenderer);

    // release used objects in correct order
    dwBoxTracker2D_release(&gBoxTracker);
    gSensorIO.reset();
    dwSAL_release(&gSal);
    dwRelease(&gSdk);
    dwLogger_release();
    gDnnInference.reset();
    gFeatureTracker.reset();
}

//#######################################################################################
void runDetector()
{
    // Run inference if the model is valid
    if (gDnnInference->isLoaded()) {
        gDnnBoxList.clear();
        gDnnInference->inferSingleFrame(&gDnnBoxList, gRgbaImage, false);
    }
}

//#######################################################################################
void runTracker(bool isFirstFrame)
{
    // add candidates to box tracker
    dwBoxTracker2D_add(gDnnBoxList.data(), gDnnBoxList.size(), gBoxTracker);

    // feature tracker
    uint32_t nTrackedFeatures = 0;
    gFeatureTracker->track(&nTrackedFeatures,
                           &gPreFeatureLocations, &gCurFeatureLocations,
                           &gFeatureStatuses,
                           *gYuvImage);

    if (!isFirstFrame) {
        // update box features
        dwBoxTracker2D_updateFeatures(&gPreFeatureLocations[0], &gFeatureStatuses[0],
                                      nTrackedFeatures, gBoxTracker);
    }

    // box tracker
    dwBoxTracker2D_track(&gCurFeatureLocations[0], &gFeatureStatuses[0],
                         &gPreFeatureLocations[0], gBoxTracker);

    // get tracked box
    const dwTrackedBox2D * trackedBoxes = nullptr;
    size_t numTrackedBoxes  = 0;
    dwBoxTracker2D_get(&trackedBoxes, &numTrackedBoxes, gBoxTracker);

    // draw detected features in the box
    for (uint32_t i = 0; i < numTrackedBoxes; i++) {
        std::vector<dwBox2D> features;
        getFeatures(features, trackedBoxes[i].nFeatures, trackedBoxes[i].featureLocations);
        drawBoxes(features, nullptr, static_cast<float32_t>(gCameraWidth),
                  static_cast<float32_t>(gCameraHeight), gLineBuffer, gRenderer);
    }

    // convert from dwBoxList2D
    std::vector<uint32_t> bboxIds;
    std::vector<dwBox2D> bboxes;
    splitTrackedBoxInfo(bboxes, bboxIds, trackedBoxes, numTrackedBoxes);

    // draw tracked boxes
    drawBoxes(bboxes, &bboxIds, static_cast<float32_t>(gCameraWidth),
              static_cast<float32_t>(gCameraHeight), gLineBuffer, gRenderer);
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
            ProgramArguments::Option_t("video",
                                       (DataPath::get() +
                                        std::string{"/samples/sfm/triangulation/video_0.h264"})
                                           .c_str()),

#ifdef VIBRANTE
            ProgramArguments::Option_t("yuv2rgb", "cuda"),
#endif
        });

    // init framework
    initSampleApp(argc, argv, &arguments, NULL, 1280, 800);

    // init driveworks
    init();

    typedef std::chrono::high_resolution_clock myclock_t;
    typedef std::chrono::time_point<myclock_t> timepoint_t;
    timepoint_t lastUpdateTime = myclock_t::now();
    uint32_t frameCount = 0;

    dwImageGL *frameGL = nullptr;
    dwStatus result;
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

        if (frameGL) {
            gSensorIO->releaseGLRgbaFrame();
            gSensorIO->releaseFrame();
            frameGL = nullptr;
        }

        result = gSensorIO->getFrame();

        if (result == DW_END_OF_STREAM) {
            std::cout << "Camera reached end of stream" << std::endl;
            dwSensor_reset(gCameraSensor);
            gFeatureTracker->reset();
            gDnnInference->reset();
            dwBoxTracker2D_reset(gBoxTracker);
            continue;
        } else if (result != DW_SUCCESS) {
            std::cerr << "Cannot read frame: " << dwGetStatusName(result) << std::endl;
            gRun = false;
            continue;
        }

        gRgbaImage = gSensorIO->getCudaRgba();
        gYuvImage = gSensorIO->getCudaYuv();
        frameGL = gSensorIO->getGlRgbaFrame();
        if (frameGL) {
            dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);
        }

        if (frameCount % gDetectInterval == 0)
            runDetector();
        else
            gDnnBoxList.clear();
        runTracker(frameCount == 0);

        gSensorIO->releaseCudaYuv();
        gSensorIO->releaseCudaRgba();

        gWindow->swapBuffers();
        frameCount++;
        CHECK_GL_ERROR();
    }

    if (frameGL) {
        gSensorIO->releaseGLRgbaFrame();
        gSensorIO->releaseFrame();
    }

    release();

    // release framework
    releaseSampleApp();

    return 0;
}
