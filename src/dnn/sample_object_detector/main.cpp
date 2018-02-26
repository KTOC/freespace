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
#include <unordered_map>

// Sample
#include <framework/DataPath.hpp>
#include <framework/ProgramArguments.hpp>
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
dwImageStreamerHandle_t gCuda2gl         = DW_NULL_HANDLE;
dwImageStreamerHandle_t gNvMedia2Cuda    = DW_NULL_HANDLE;
dwImageFormatConverterHandle_t gYuv2rgba = DW_NULL_HANDLE;
dwRendererHandle_t gRenderer             = DW_NULL_HANDLE;
dwRenderBufferHandle_t gLineBuffer       = DW_NULL_HANDLE;
dwSensorHandle_t gCameraSensor           = DW_NULL_HANDLE;

dwRect gScreenRectangle;

dwImageCUDA *gRgbaImage;
dwImageCUDA *gYuvImage;
std::unique_ptr<ISensorIO> gSensorIO;

uint32_t gCameraWidth  = 0U;
uint32_t gCameraHeight = 0U;

std::unique_ptr<DNNInference> gDnnInference;
std::vector<dwBox2D> gDnnBoxList;
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
    float32_t boxColor[4] = {0.0f,1.0f,0.0f,1.0f};
    dwRenderer_setColor(boxColor, renderer);
    dwRenderer_setLineWidth(2.0f, renderer);
    dwRenderer_setRect(gScreenRectangle, renderer);
}

//#######################################################################################
void setupLineBuffer(dwRenderBufferHandle_t &lineBuffer, unsigned int maxLines, dwContextHandle_t dwSdk)
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

    dwImageProperties cameraImageProperties{};
    dwSensorCamera_getImageProperties(&cameraImageProperties,
                                      DW_CAMERA_PROCESSED_IMAGE,
                                      salSensor);
    dwCameraProperties cameraProperties{};
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
    // init renderer
    gScreenRectangle.height = gWindow->height();
    gScreenRectangle.width = gWindow->width();
    gScreenRectangle.x = 0;
    gScreenRectangle.y = 0;

    unsigned int maxLines = 20000;
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

    createVideoReplay(gCameraSensor, gCameraWidth, gCameraHeight, cameraSiblings,
                      cameraFramerate, imageType, gSal,
                      gArguments.has("video") ? gArguments.get("video") : "");

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

//#######################################################################################
void init()
{
    initDriveworks();

    initCameras();

    initRenderer();

    initDNN();
}

//#######################################################################################
void release()
{
    dwSensor_stop(gCameraSensor);
    dwSAL_releaseSensor(&gCameraSensor);

    dwRenderBuffer_release(&gLineBuffer);
    dwRenderer_release(&gRenderer);

    // release used objects in correct order
    gSensorIO.reset();
    dwSAL_release(&gSal);
    dwRelease(&gSdk);
    dwLogger_release();
    gDnnInference.reset();
}

//#######################################################################################
void runDetector()
{
    // Run inference if the model is valid
    if (gDnnInference->isLoaded()) {
        gDnnBoxList.clear();
        gDnnInference->inferSingleFrame(&gDnnBoxList, gRgbaImage, true);

        drawBoxes(gDnnBoxList, NULL, static_cast<float32_t>(gCameraWidth),
                  static_cast<float32_t>(gCameraHeight), gLineBuffer, gRenderer);
    }
}

//#######################################################################################
int main(int argc, const char **argv)
{
    const ProgramArguments arguments = ProgramArguments(
        {
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

    dwImageGL *frameGL = nullptr;
    dwStatus result;
    // main loop
    while (gRun && !gWindow->shouldClose()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        std::this_thread::yield();

        // run with at most 30FPS
        std::chrono::milliseconds timeSinceUpdate =
                std::chrono::duration_cast<std::chrono::milliseconds>(myclock_t::now() - lastUpdateTime);
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

        runDetector();

        gSensorIO->releaseCudaYuv();
        gSensorIO->releaseCudaRgba();

        gWindow->swapBuffers();
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
