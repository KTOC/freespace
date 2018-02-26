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
// Copyright (c) 2017 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////
#define _CRT_SECURE_NO_WARNINGS

#include <thread>

// Sample
#include <framework/Checks.hpp>
#include <framework/DataPath.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SampleFramework.hpp>

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

// FreeSpaceDetector
#include <dw/freespace/FreeSpaceDetector.h>

// RCCB
#include <dw/isp/SoftISP.h>

// Calibrated camera
#include <dw/rigconfiguration/RigConfiguration.h>
#include <dw/rigconfiguration/Camera.h>

//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------
dwContextHandle_t gSdk                         = DW_NULL_HANDLE;
dwSALHandle_t gSal                             = DW_NULL_HANDLE;
dwImageFormatConverterHandle_t gInput2Rgba     = DW_NULL_HANDLE;
dwImageStreamerHandle_t gCamera2gl             = DW_NULL_HANDLE; // Could be using CUDA or NVMedia
dwRendererHandle_t gRenderer                   = DW_NULL_HANDLE;
dwRenderBufferHandle_t gLineBuffer             = DW_NULL_HANDLE;
dwSensorHandle_t gCameraSensor                 = DW_NULL_HANDLE;
dwFreeSpaceDetectorHandle_t gFreeSpaceDetector = DW_NULL_HANDLE;
dwImageCUDA gFrameCUDArgba{};
dwSoftISPHandle_t gSoftISP                    = DW_NULL_HANDLE;
dwImageStreamerHandle_t gInput2cuda            = DW_NULL_HANDLE;
dwImageCUDA gRcbImage{};
dwImageProperties gCameraImageProperties;
dwRigConfigurationHandle_t gRigConfig          = DW_NULL_HANDLE;
dwCameraRigHandle_t gRigHandle                 = DW_NULL_HANDLE;
dwCalibratedCameraHandle_t gCalibratedCam      = DW_NULL_HANDLE;


#ifdef VIBRANTE
dwImageFormatConverterHandle_t gNvMYuv2rgba  = DW_NULL_HANDLE;
dwImageStreamerHandle_t gNvMedia2Cuda        = DW_NULL_HANDLE;
dwImageNvMedia gFrameNVMrgba{};
#endif

dwRect gScreenRectangle{};

uint32_t gWindowWidth = 960;
uint32_t gWindowHeight = 604;

uint32_t gCameraWidth  = 0U;
uint32_t gCameraHeight = 0U;

float32_t gTemporalSmoothFactor = 0.5f;
uint32_t gSpatialSmoothFilterWidth = 5;

float32_t gDrawScaleX;
float32_t gDrawScaleY;

std::string gInputType;
bool gRaw = false;
bool gRig = false;
cudaStream_t gCudaStream  = 0;

//#######################################################################################
void drawFreeSpaceDetectionROI(dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer)
{
    dwRect roi{};
    dwFreeSpaceDetector_getDetectionROI(&roi,gFreeSpaceDetector);
    float32_t x_start = static_cast<float32_t>(roi.x)*gDrawScaleX;
    float32_t x_end   = static_cast<float32_t>(roi.x + roi.width)*gDrawScaleX;
    float32_t y_start = static_cast<float32_t>(roi.y)*gDrawScaleY;
    float32_t y_end   = static_cast<float32_t>(roi.y + roi.height)*gDrawScaleY;
    float32_t *coords     = nullptr;
    uint32_t maxVertices  = 0;
    uint32_t vertexStride = 0;
    dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);
    coords[0]  = x_start;
    coords[1]  = y_start;
    coords    += vertexStride;
    coords[0]  = x_start;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_start;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_end;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_end;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0] = x_end;
    coords[1] = y_start;
    coords    += vertexStride;
    coords[0] = x_end;
    coords[1] = y_start;
    coords    += vertexStride;
    coords[0] = x_start;
    coords[1] = y_start;
    dwRenderBuffer_unmap(8, renderBuffer);
    dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, renderer);
    dwRenderer_setLineWidth(2, renderer);
    dwRenderer_renderBuffer(renderBuffer, renderer);
}

//#######################################################################################
void drawFreeSpaceBoundary(dwFreeSpaceDetection* boundary, dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer)
{
    drawFreeSpaceDetectionROI(renderBuffer, renderer);

    uint32_t n_verts = 0;
    float32_t* coords= nullptr;
    uint32_t maxVertices = 0;
    uint32_t vertexStride = 0;
    dwFreeSpaceBoundaryType category = boundary->boundaryType[0];
    float32_t maxWidth = 8.0; //10 meters as a step, [0, 10) will have max line width
    float32_t witdhRatio = 0.8;
    float32_t dist2Width[20];
    dist2Width[0] = maxWidth;
    for(uint32_t i = 1; i < 20; i++)
        dist2Width[i] = dist2Width[i-1]*witdhRatio;

    float32_t prevWidth, curWidth = maxWidth/2;
    if(gRig) {
        prevWidth = dist2Width[static_cast<uint32_t>(boundary->boundaryWorldPoint[0].x/10)];
    } else {
        prevWidth = curWidth;
    }

    dwRenderer_setLineWidth(prevWidth, renderer);

    if(category==DW_BOUNDARY_TYPE_OTHER)
        dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, renderer);
    else if(category==DW_BOUNDARY_TYPE_CURB)
        dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, renderer);
    else if(category==DW_BOUNDARY_TYPE_VEHICLE)
        dwRenderer_setColor(DW_RENDERER_COLOR_RED, renderer);
    else if(category==DW_BOUNDARY_TYPE_PERSON)
        dwRenderer_setColor(DW_RENDERER_COLOR_BLUE, renderer);

    dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);

    for (uint32_t i = 1; i < boundary->numberOfBoundaryPoints; ++i) {
        if(gRig) {
            curWidth = dist2Width[static_cast<uint32_t>(boundary->boundaryWorldPoint[i].x/10)];
        }

        if(boundary->boundaryType[i] != boundary->boundaryType[i-1] || curWidth != prevWidth) {
            dwRenderBuffer_unmap(n_verts, renderBuffer);
            dwRenderer_renderBuffer(renderBuffer, renderer);

            coords = nullptr;
            maxVertices = 0;
            vertexStride = 0;
            n_verts = 0;
            dwRenderer_setLineWidth(curWidth, renderer);

            category = boundary->boundaryType[i];
            if(category==DW_BOUNDARY_TYPE_OTHER)
                dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, renderer);
            else if(category==DW_BOUNDARY_TYPE_CURB)
                  dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, renderer);
            else if(category==DW_BOUNDARY_TYPE_VEHICLE)
                  dwRenderer_setColor(DW_RENDERER_COLOR_RED, renderer);
            else if(category==DW_BOUNDARY_TYPE_PERSON)
                  dwRenderer_setColor(DW_RENDERER_COLOR_BLUE, renderer);

            dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);
        }
        n_verts += 2;
        if(n_verts > maxVertices)
            break;

        coords[0] = static_cast<float32_t>(boundary->boundaryImagePoint[i-1].x*gDrawScaleX);
        coords[1] = static_cast<float32_t>(boundary->boundaryImagePoint[i-1].y*gDrawScaleY);
        coords += vertexStride;

        coords[0] = static_cast<float32_t>(boundary->boundaryImagePoint[i].x*gDrawScaleX);
        coords[1] = static_cast<float32_t>(boundary->boundaryImagePoint[i].y*gDrawScaleY);
        coords += vertexStride;
        prevWidth = curWidth;
    }

    dwRenderBuffer_unmap(n_verts, renderBuffer);
    dwRenderer_renderBuffer(renderBuffer, renderer);
}

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
    dwRenderBuffer_set2DCoordNormalizationFactors((float32_t)gWindow->width(),
                                                  (float32_t)gWindow->height(), lineBuffer);
}

//#######################################################################################
void renderCameraTexture(dwImageStreamerHandle_t streamer, dwRendererHandle_t renderer)
{
    dwImageGL *frameGL = nullptr;

    if (dwImageStreamer_receiveGL(&frameGL, 30000, streamer) != DW_SUCCESS) {
        std::cerr << "did not received GL frame within 30ms" << std::endl;
    } else {
        // render received texture
        dwRenderer_renderTexture(frameGL->tex, frameGL->target, renderer);
        dwImageStreamer_returnReceivedGL(frameGL, streamer);
    }
}

//#######################################################################################
void runDetector(dwImageCUDA* frame)
{
    // Run inference if the model is valid
    if (gFreeSpaceDetector)
    {
        dwFreeSpaceDetection boundary{};

        dwStatus res = dwFreeSpaceDetector_processDeviceAsync(frame, gFreeSpaceDetector);
        res = res == DW_SUCCESS ? dwFreeSpaceDetector_interpretHost(gFreeSpaceDetector) : res;

        if (res != DW_SUCCESS)
        {
            std::cerr << "runDetector failed with: " << dwGetStatusName(res) << std::endl;
        }
        else
            dwFreeSpaceDetector_getBoundaryDetection(&boundary, gFreeSpaceDetector);

        drawFreeSpaceBoundary(&boundary, gLineBuffer, gRenderer);
    }
}

//#######################################################################################
dwStatus runSingleCameraPipelineRaw()
{
    dwStatus result                 = DW_FAILURE;
    dwCameraFrameHandle_t frame     = nullptr;
    dwImageCUDA* frameCUDARaw       = nullptr;
    dwImageCPU *frameCPURaw         = nullptr;
    dwImageCUDA* retimg             = nullptr;
    const dwImageDataLines* dataLines;
#ifdef VIBRANTE
    dwImageNvMedia *frameNvMediaRaw = nullptr;
#endif

    result = dwSensorCamera_readFrame(&frame, 0, 1000000, gCameraSensor);
    if (result == DW_END_OF_STREAM)
        return result;
    if (result != DW_SUCCESS && result != DW_END_OF_STREAM) {
        std::cerr << "readFrameNvMedia: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    if (gInputType.compare("camera") == 0) {
#ifdef VIBRANTE
        result = dwSensorCamera_getImageNvMedia(&frameNvMediaRaw, DW_CAMERA_RAW_IMAGE, frame);
#endif
    }
    else{
        result = dwSensorCamera_getImageCPU(&frameCPURaw, DW_CAMERA_RAW_IMAGE, frame);
    }
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot get raw image: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    result = dwSensorCamera_getDataLines(&dataLines, frame);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot get data lines: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    if (gInputType.compare("camera") == 0) {
#ifdef VIBRANTE
        result = dwImageStreamer_postNvMedia(frameNvMediaRaw, gInput2cuda);
#endif
    }
    else{
        result = dwImageStreamer_postCPU(frameCPURaw, gInput2cuda);
    }
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot post image: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    result = dwImageStreamer_receiveCUDA(&frameCUDARaw, 10000, gInput2cuda);

    // Raw -> RCB & RGBA
    dwSoftISP_bindRawInput(frameCUDARaw, gSoftISP);
    CHECK_DW_ERROR(dwSoftISP_processDeviceAsync(DW_SOFT_ISP_PROCESS_TYPE_DEMOSAIC | DW_SOFT_ISP_PROCESS_TYPE_TONEMAP,
                                                gSoftISP));

    // frame -> GL (rgba) - for rendering
    {
        result = dwImageStreamer_postCUDA(&gFrameCUDArgba, gCamera2gl);
        if (result != DW_SUCCESS) {
            std::cerr << "cannot post RGBA image" << dwGetStatusName(result) << std::endl;
            return result;
        }

        renderCameraTexture(gCamera2gl, gRenderer);

        result = dwImageStreamer_waitPostedCUDA(&retimg, 60000, gCamera2gl);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot wait post RGBA image" << dwGetStatusName(result) << std::endl;
            return result;
        }
    }

    // Run FreeSpaceNet
    runDetector(&gRcbImage);

    dwImageStreamer_returnReceivedCUDA(frameCUDARaw, gInput2cuda);

    if (gInputType.compare("camera") == 0) {
#ifdef VIBRANTE
        dwImageStreamer_waitPostedNvMedia(&frameNvMediaRaw, 10000, gInput2cuda);
#endif
    }
    else{
        dwImageStreamer_waitPostedCPU(&frameCPURaw, 10000, gInput2cuda);
    }

    dwSensorCamera_returnFrame(&frame);

    return DW_SUCCESS;
}

//#######################################################################################
dwStatus runSingleCameraPipelineH264()
{
    dwStatus result             = DW_FAILURE;
    dwCameraFrameHandle_t frame     = nullptr;
#ifdef VIBRANTE
    dwImageNvMedia *frameNvMediaYuv = nullptr;
    dwImageCUDA *imgCUDA            = nullptr;
    dwImageNvMedia *retimg          = nullptr;
#else
    dwImageCUDA *frameCUDAyuv       = nullptr;
    dwImageCUDA *retimg             = nullptr;
#endif

    result = dwSensorCamera_readFrame(&frame, 0, 50000, gCameraSensor);
    if (result == DW_END_OF_STREAM)
        return result;
    if (result != DW_SUCCESS) {
        std::cout << "readFrameCUDA: " << dwGetStatusName(result) << std::endl;
        return result;
    }

#ifdef VIBRANTE
    result = dwSensorCamera_getImageNvMedia(&frameNvMediaYuv, DW_CAMERA_PROCESSED_IMAGE, frame);
#else
    result = dwSensorCamera_getImageCUDA(&frameCUDAyuv, DW_CAMERA_PROCESSED_IMAGE, frame);
#endif
    if (result != DW_SUCCESS) {
        std::cout << "getImage: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    // YUV->RGBA
#ifdef VIBRANTE
    result = dwImageFormatConverter_copyConvertNvMedia(&gFrameNVMrgba, frameNvMediaYuv, gNvMYuv2rgba);
#else
    result = dwImageFormatConverter_copyConvertCUDA(&gFrameCUDArgba, frameCUDAyuv, gInput2Rgba, 0);
#endif
    if (result != DW_SUCCESS) {
        std::cout << "Cannot convert to RGBA: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    // we can return the frame already now, we are working with a copy from now on
    dwSensorCamera_returnFrame(&frame);

    // frame -> GL (rgba) - for rendering
    {
#if VIBRANTE
        result = dwImageStreamer_postNvMedia(&gFrameNVMrgba, gCamera2gl);
#else
        result = dwImageStreamer_postCUDA(&gFrameCUDArgba, gCamera2gl);
#endif
        if (result != DW_SUCCESS) {
            std::cerr << "cannot post RGBA image" << dwGetStatusName(result) << std::endl;
            return result;
        }

        renderCameraTexture(gCamera2gl, gRenderer);

#if VIBRANTE
        result = dwImageStreamer_waitPostedNvMedia(&retimg, 60000, gCamera2gl);
#else
        result = dwImageStreamer_waitPostedCUDA(&retimg, 60000, gCamera2gl);
#endif
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot wait post RGBA image" << dwGetStatusName(result) << std::endl;
            return result;
        }
    }

#if VIBRANTE
    // (nvmedia) NVMEDIA -> CUDA (rgba) - for processing
    // since DNN expects pitch linear cuda memory we cannot just post gFrameNVMrgba through the streamer
    // cause the outcome of the streamer would have block layout, but we need pitch
    // hence we perform one more extra YUV2RGBA conversion using CUDA
    {
        result = dwImageStreamer_postNvMedia(frameNvMediaYuv, gNvMedia2Cuda);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot post NvMedia frame " << dwGetStatusName(result) << std::endl;
            return result;
        }

        result = dwImageStreamer_receiveCUDA(&imgCUDA, 60000, gNvMedia2Cuda);
        if (result != DW_SUCCESS || imgCUDA == 0) {
            std::cerr << "did not received CUDA frame within 60ms" << std::endl;
            return result;
        }

        // copy convert into RGBA
        result = dwImageFormatConverter_copyConvertCUDA(&gFrameCUDArgba, imgCUDA, gInput2Rgba, 0);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot convert to RGBA" << std::endl;
            return result;
        }

    }
#endif

    // Run FreeSpaceNet
    runDetector(&gFrameCUDArgba);

#if VIBRANTE
    dwImageStreamer_returnReceivedCUDA(imgCUDA, gNvMedia2Cuda);
    dwImageStreamer_waitPostedNvMedia(&retimg, 60000, gNvMedia2Cuda);
#endif

    return DW_SUCCESS;
}


//#######################################################################################
bool createVideoReplay(dwSensorHandle_t &salSensor,
                       float32_t &cameraFrameRate,
                       dwSALHandle_t sal)
{
    dwSensorParams params;
    dwStatus result;

    if (gInputType.compare("camera") == 0) {
        std::string cameraType = gArguments.get("camera-type");
        std::string parameterString = "camera-type=" + cameraType;
        parameterString += ",csi-port=" + gArguments.get("csi-port");
        parameterString += ",slave=" + gArguments.get("slave");
        parameterString += ",serialize=false,camera-count=4";
        if(cameraType.compare("c-ov10640-b1") == 0 || cameraType.compare("ov10635") == 0){
            parameterString += ",output-format=yuv";
            gRaw = false;
        }
        else{
            parameterString += ",output-format=raw";
            gRaw = true;
        }
        std::string cameraMask[4] = {"0001", "0010", "0100", "1000"};
        uint32_t cameraIdx = std::stoi(gArguments.get("camera-index"));
        if(cameraIdx < 0 || cameraIdx > 3){
            std::cerr << "Error: camera index must be 0, 1, 2 or 3" << std::endl;
            return false;
        }
        parameterString += ",camera-mask=" + cameraMask[cameraIdx];

        params.parameters           = parameterString.c_str();
        params.protocol             = "camera.gmsl";

        result                      = dwSAL_createSensor(&salSensor, params, sal);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot create driver: camera.gmsl with params: " << params.parameters << std::endl
                      << "Error: " << dwGetStatusName(result) << std::endl;
            return false;
        }
    }
    else{
        std::string parameterString = gArguments.parameterString();
        params.parameters           = parameterString.c_str();
        params.protocol             = "camera.virtual";
        result                      = dwSAL_createSensor(&salSensor, params, sal);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot create driver: camera.virtual with params: " << params.parameters << std::endl
                      << "Error: " << dwGetStatusName(result) << std::endl;
            return false;
        }
        std::string videoFormat = gArguments.get("video");
        std::size_t found = videoFormat.find_last_of(".");
        gRaw = videoFormat.substr(found+1).compare("raw") == 0 ? true : false;
    }

    dwCameraProperties cameraProperties;
    dwSensorCamera_getSensorProperties(&cameraProperties, salSensor);
    cameraFrameRate = cameraProperties.framerate;

    if(gRaw)
        dwSensorCamera_getImageProperties(&gCameraImageProperties, DW_CAMERA_RAW_IMAGE, salSensor);
    else
        dwSensorCamera_getImageProperties(&gCameraImageProperties, DW_CAMERA_PROCESSED_IMAGE, salSensor);

    if(gRaw && gInputType.compare("camera") == 0)
        gCameraImageProperties.height = cameraProperties.resolution.y;
    gCameraHeight = gCameraImageProperties.height;
    gCameraWidth = gCameraImageProperties.width;

    return true;
}


//#######################################################################################
bool initDriveworks()
{
    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_DEBUG);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams{};

    std::string path = DataPath::get();
    sdkParams.dataPath = path.c_str();

#ifdef VIBRANTE
    sdkParams.eglDisplay = gWindow->getEGLDisplay();
#endif

    return dwInitialize(&gSdk, DW_VERSION, &sdkParams) == DW_SUCCESS;
}

//#######################################################################################
bool initRenderer()
{
    // init renderer
    gScreenRectangle.height = gWindow->height();
    gScreenRectangle.width = gWindow->width();
    gScreenRectangle.x = 0;
    gScreenRectangle.y = 0;

    unsigned int maxLines = 20000;
    setupRenderer(gRenderer, gSdk);
    setupLineBuffer(gLineBuffer, maxLines, gSdk);

    return true;
}

//#######################################################################################
bool initRigConfiguration()
{
    dwStatus result = DW_SUCCESS;
    //Load vehicle configuration
    result = dwRigConfiguration_initializeFromFile(&gRigConfig, gSdk, gArguments.get("rig").c_str());
    if (result != DW_SUCCESS) {
        std::cerr << "Error dwRigConfiguration_initialize: " << dwGetStatusName(result) << std::endl;
        return false;
    }
    uint32_t cameraCount;
    result = dwCameraRig_initializeFromConfig(&gRigHandle, &cameraCount,
                                              &gCalibratedCam,
                                              1U, gSdk, gRigConfig);
    if (result != DW_SUCCESS) {
        std::cerr << "Error dwCameraRig_initializeFromConfig: " << dwGetStatusName(result) << std::endl;
        return false;
    }
    return true;
}

//#######################################################################################
bool initCameras()
{
    dwStatus result = DW_FAILURE;

    // create sensor abstraction layer
    result = dwSAL_initialize(&gSal, gSdk);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot init sal: " << dwGetStatusName(result) << std::endl;
        return false;
    }
    // create GMSL Camera interface
    float32_t cameraFramerate = 0.0f;

    if(!createVideoReplay(gCameraSensor, cameraFramerate, gSal))
        return false;

    std::cout << "Camera image with " << gCameraWidth << "x" << gCameraHeight << " at "
              << cameraFramerate << " FPS" << std::endl;

    gRun = gRun && dwSensor_start(gCameraSensor) == DW_SUCCESS;

    return gRun;
}

//#######################################################################################
bool initPipeline()
{
    dwStatus status = DW_FAILURE;

#ifdef VIBRANTE
    // NvMedia yuv -> rgba format converter
    dwImageProperties cameraImageProperties = gCameraImageProperties;
    cameraImageProperties.type = DW_IMAGE_NVMEDIA;

    dwImageFormatConverter_initialize(&gNvMYuv2rgba, cameraImageProperties.type, gSdk);

    // NvMedia -> CUDA image streamer
    status = dwImageStreamer_initialize(&gNvMedia2Cuda, &cameraImageProperties, DW_IMAGE_CUDA, gSdk);
    if (status != DW_SUCCESS) {
        std::cerr << "Cannot init image streamer: " << dwGetStatusName(status) << std::endl;
        return false;
    }
#endif

    if(gRaw){
        dwImageProperties cameraImageProperties;
        status = dwSensorCamera_getImageProperties(&cameraImageProperties, DW_CAMERA_RAW_IMAGE, gCameraSensor);
        if( gInputType.compare("camera") == 0)
            cameraImageProperties.height = gCameraHeight;
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot get image properties: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        dwCameraProperties cameraProperties;
        status = dwSensorCamera_getSensorProperties(&cameraProperties, gCameraSensor);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot get camera properties: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Raw pipeline
        dwImageProperties rccbImageProperties;
        dwSoftISPParams softISPParams;
        dwSoftISP_initParamsFromCamera(&softISPParams, cameraProperties);
        status = dwSoftISP_initialize(&gSoftISP, softISPParams, gSdk);
        status = status != DW_SUCCESS ? status : dwSoftISP_setCUDAStream(gCudaStream, gSoftISP);
        status = status != DW_SUCCESS ? status : dwSoftISP_getDemosaicImageProperties(&rccbImageProperties, gSoftISP);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot initialize raw pipeline: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Input -> CUDA streamer
        dwImageStreamer_initialize(&gInput2cuda, &cameraImageProperties, DW_IMAGE_CUDA, gSdk);

        // Input -> RGBA format converter
        dwImageProperties displayImageProperties = rccbImageProperties;
        displayImageProperties.pxlFormat = DW_IMAGE_RGBA;
        displayImageProperties.pxlType = DW_TYPE_UINT8;
        displayImageProperties.planeCount = 1;
        status = dwImageFormatConverter_initialize(&gInput2Rgba, DW_IMAGE_CUDA, gSdk);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot initialize input -> rgba format converter: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Setup RCB image
        gRcbImage.prop = rccbImageProperties;
        gRcbImage.layout = DW_IMAGE_CUDA_PITCH;
        cudaMallocPitch(&gRcbImage.dptr[0], &gRcbImage.pitch[0], rccbImageProperties.width * dwSizeOf(rccbImageProperties.pxlType),
            rccbImageProperties.height * rccbImageProperties.planeCount);
        gRcbImage.pitch[1] = gRcbImage.pitch[2] = gRcbImage.pitch[0];
        gRcbImage.dptr[1] = reinterpret_cast<uint8_t*>(gRcbImage.dptr[0]) + rccbImageProperties.height * gRcbImage.pitch[0];
        gRcbImage.dptr[2] = reinterpret_cast<uint8_t*>(gRcbImage.dptr[1]) + rccbImageProperties.height * gRcbImage.pitch[1];

        dwSoftISP_bindDemosaicOutput(&gRcbImage, gSoftISP);

        // Camera -> GL image streamer
        status = dwImageStreamer_initialize(&gCamera2gl, &displayImageProperties, DW_IMAGE_GL, gSdk);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot init GL streamer: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Setup RGBA CUDA image
        {
            void *dptr = nullptr;
            size_t pitch = 0;
            cudaMallocPitch(&dptr, &pitch, rccbImageProperties.width * 4, rccbImageProperties.height);
            dwImageCUDA_setFromPitch(&gFrameCUDArgba, dptr, rccbImageProperties.width, rccbImageProperties.height,
                                     pitch, DW_IMAGE_RGBA);

            dwSoftISP_bindTonemapOutput(&gFrameCUDArgba, gSoftISP);
        }
#ifdef VIBRANTE
        // Setup RGBA NvMedia image
        {
            dwImageProperties properties = rccbImageProperties;
            properties.type = DW_IMAGE_NVMEDIA;
            properties.pxlFormat = DW_IMAGE_RGBA;
            properties.pxlType = DW_TYPE_UINT8;
            properties.planeCount = 1;
            dwImageNvMedia_create(&gFrameNVMrgba, &properties, gSdk);
        }
#endif
        // Set camera width and height for DNN
        gCameraWidth = rccbImageProperties.width;
        gCameraHeight = rccbImageProperties.height;
    }
    else{
        dwImageProperties cameraImageProperties;
        status = dwSensorCamera_getImageProperties(&cameraImageProperties, DW_CAMERA_PROCESSED_IMAGE, gCameraSensor);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot get image properties: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Input -> RGBA format converter
        dwImageProperties displayImageProperties = cameraImageProperties;
        displayImageProperties.pxlFormat = DW_IMAGE_RGBA;
        displayImageProperties.pxlType = DW_TYPE_UINT8;
        displayImageProperties.planeCount = 1;
        displayImageProperties.type = DW_IMAGE_CUDA;
        cameraImageProperties.type = DW_IMAGE_CUDA;
        status = dwImageFormatConverter_initialize(&gInput2Rgba, DW_IMAGE_CUDA, gSdk);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot initialize input -> rgba format converter: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Camera -> GL image streamer
#ifdef VIBRANTE
        displayImageProperties.type = DW_IMAGE_NVMEDIA;
        status = dwImageStreamer_initialize(&gCamera2gl, &displayImageProperties, DW_IMAGE_GL, gSdk);
#else
        status = dwImageStreamer_initialize(&gCamera2gl, &displayImageProperties, DW_IMAGE_GL, gSdk);
#endif
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot init GL streamer: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Setup RGBA CUDA image
        {
            void *dptr = nullptr;
            size_t pitch = 0;
            cudaMallocPitch(&dptr, &pitch, gCameraWidth * 4, gCameraHeight);
            dwImageCUDA_setFromPitch(&gFrameCUDArgba, dptr, gCameraWidth, gCameraHeight,
                                     pitch, DW_IMAGE_RGBA);
        }

#ifdef VIBRANTE
        // Setup RGBA NvMedia image
        {
            dwImageProperties properties = displayImageProperties;
            properties.type = DW_IMAGE_NVMEDIA;
            properties.pxlFormat = DW_IMAGE_RGBA;
            properties.pxlType = DW_TYPE_UINT8;
            properties.planeCount = 1;
            dwImageNvMedia_create(&gFrameNVMrgba, &properties, gSdk);
        }
#endif
        // Set camera width and height for DNN
        gCameraWidth = cameraImageProperties.width;
        gCameraHeight = cameraImageProperties.height;
    }

    return true;
}


//#######################################################################################
bool initDNN()
{
    dwStatus res = DW_FAILURE;

    dwTransformation transformation{};
    res = dwRigConfiguration_getSensorToRigTransformation(&transformation, 0, gRigConfig);
    if (res != DW_SUCCESS) //only compute free space bounday in image space
    {
        std::cerr << "Cannot parse rig configuration: " << dwGetStatusName(res) <<std::endl;
        std::cerr << "Compute free space boundary in image space only." <<std::endl;
        res = dwFreeSpaceDetector_initializeFreeSpaceNet(&gFreeSpaceDetector,
                                                         gCameraWidth, gCameraHeight,
                                                         gCudaStream,
                                                         gSdk);
    }
    else //compute free space bounday in image and vehicle coordinates
    {
        float32_t maxDistance = 50.0f;
        std::string maxDistanceStr = gArguments.get("maxDistance");
        if(maxDistanceStr!="50.0") {
            try{
                maxDistance = std::stof(maxDistanceStr);
                if (maxDistance < 0.0f) {
                    std::cerr << "maxDistance cannot be negative." << std::endl;
                    return false;
                }
            } catch(...) {
                std::cerr << "Given maxDistance can't be parsed" << std::endl;
                return false;
            }
        }
        res = dwFreeSpaceDetector_initializeCalibratedFreeSpaceNet(&gFreeSpaceDetector,
                                                                   gCameraWidth, gCameraHeight,
                                                                   gCudaStream, transformation, maxDistance,
                                                                   gCalibratedCam ,gSdk);
    }

    if (gRaw){
        // this is in case the network requires a different tonemapper than the default one
        dwDNNMetaData metaData;
        dwFreeSpaceDetector_getDNNMetaData(&metaData, gFreeSpaceDetector);
        dwSoftISP_setTonemapType(metaData.tonemapType, gSoftISP);
    }

    if (res != DW_SUCCESS)
    {
        std::cerr << "Cannot initialize FreeSpaceNet: " << dwGetStatusName(res) << std::endl;
        return false;
    }

    //Default to no spatial filtering, uncomment this block to customize free space boundary spatial smoothing filter width
    res = dwFreeSpaceDetector_setSpatialSmoothFilterWidth(gSpatialSmoothFilterWidth, gFreeSpaceDetector);
    if (res != DW_SUCCESS)
    {
        std::cerr << "Cannot set free space boundary spatial smooth filter: " << dwGetStatusName(res) << std::endl;
        return false;
    }

    // customize ROI to a subset in the center of the frame
    dwRect roi = {int32_t(gCameraWidth)/10,   int32_t(gCameraHeight)/10,
                  int32_t(gCameraWidth)*8/10, int32_t(gCameraHeight)*8/10};
    res = dwFreeSpaceDetector_setDetectionROI(&roi, gFreeSpaceDetector);
    if (res != DW_SUCCESS)
    {
        std::cerr << "Cannot set FreeSpaceNet detection ROI: " << dwGetStatusName(res) << std::endl;
        return false;
    }

    // customize free space boundary temporal smoothing factor
    res = dwFreeSpaceDetector_setTemporalSmoothFactor(gTemporalSmoothFactor, gFreeSpaceDetector);
    if (res != DW_SUCCESS)
    {
        std::cerr << "Cannot set free space boundary temporal smooth factor: " << dwGetStatusName(res) << std::endl;
        return false;
    }

    // boundary points are in camera space
    gDrawScaleX = static_cast<float32_t>(gWindowWidth)/static_cast<float32_t>(gCameraWidth);
    gDrawScaleY = static_cast<float32_t>(gWindowHeight)/static_cast<float32_t>(gCameraHeight);

    return true;
}


//#######################################################################################
bool init()
{
    if (!initDriveworks()) return false;
    if (!initCameras()) return false;
    gRig = initRigConfiguration(); //allow sample runing with or without calibration file
    if (!initRenderer()) return false;
    if (!initPipeline()) return false;
    if (!initDNN()) return false;

    return true;
}

//#######################################################################################
void release()
{
    if (gInput2Rgba != DW_NULL_HANDLE)
        dwImageFormatConverter_release(&gInput2Rgba);

    if (gFrameCUDArgba.dptr[0])
        cudaFree(gFrameCUDArgba.dptr[0]);

    if (gCamera2gl != DW_NULL_HANDLE)
        dwImageStreamer_release(&gCamera2gl);

    if(gRaw){
        if (gInput2cuda != DW_NULL_HANDLE)
            dwImageStreamer_release(&gInput2cuda);
        if (gSoftISP != DW_NULL_HANDLE)
            dwSoftISP_release(&gSoftISP);
        if (gRcbImage.dptr[0])
            cudaFree(gRcbImage.dptr[0]);
    }

#ifdef VIBRANTE
    if (gNvMYuv2rgba != DW_NULL_HANDLE)
        dwImageFormatConverter_release(&gNvMYuv2rgba);

    if (gNvMedia2Cuda != DW_NULL_HANDLE)
        dwImageStreamer_release(&gNvMedia2Cuda);

    if (gFrameNVMrgba.img != nullptr)
        NvMediaImageDestroy(gFrameNVMrgba.img);
#endif

    dwCalibratedCamera_release(&gCalibratedCam);
    dwCameraRig_reset(gRigHandle);
    dwRigConfiguration_reset(gRigConfig);

    dwSensor_stop(gCameraSensor);
    dwSAL_releaseSensor(&gCameraSensor);

    dwRenderBuffer_release(&gLineBuffer);
    dwRenderer_release(&gRenderer);

    // release used objects in correct order
    dwSAL_release(&gSal);
    dwFreeSpaceDetector_release(&gFreeSpaceDetector);
    dwRelease(&gSdk);
    dwLogger_release();
}

//#######################################################################################
int main(int argc, const char **argv)
{
    const ProgramArguments arguments = ProgramArguments({
#if VIBRANTE
            ProgramArguments::Option_t("camera-type", "ar0231-rccb-ssc"),
            ProgramArguments::Option_t("csi-port", "ab"),
            ProgramArguments::Option_t("camera-index", "0"),
            ProgramArguments::Option_t("slave", "0"),
            ProgramArguments::Option_t("input-type", "video"),
#endif
            ProgramArguments::Option_t("video", \
                                       (DataPath::get() + \
                                       std::string{"/samples/freespace/video_freespace.h264"}).c_str()),
            ProgramArguments::Option_t("rig", \
                                       (DataPath::get() + \
                                       std::string{"/samples/freespace/rig_freespace.xml"}).c_str()),
            ProgramArguments::Option_t("width", "960"),
            ProgramArguments::Option_t("height", "576"),
            ProgramArguments::Option_t("maxDistance", "50.0"),
        });

    // init framework
    initSampleApp(argc, argv, &arguments, NULL, gWindowWidth, gWindowHeight);

#ifdef VIBRANTE
    gInputType = gArguments.get("input-type");
#else
    gInputType = "video";
#endif

    // init driveworks
    if (!init())
    {
        std::cerr << "Cannot initialize DW subsystems" << std::endl;
        gRun = false;
    }

    typedef std::chrono::high_resolution_clock myclock_t;
    typedef std::chrono::time_point<myclock_t> timepoint_t;
    timepoint_t lastUpdateTime = myclock_t::now();

    // main loop
    while (gRun && !gWindow->shouldClose()) {
        std::this_thread::yield();

        bool processImage = true;

        // run with at most 30FPS
        std::chrono::milliseconds timeSinceUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(myclock_t::now() - lastUpdateTime);
        if (timeSinceUpdate < std::chrono::milliseconds(33)) //33
            processImage = false;

        dwStatus status = DW_FAILURE;
        if (processImage) {

            lastUpdateTime = myclock_t::now();

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            if(gRaw){
                status = runSingleCameraPipelineRaw();
            }
            else{
                status = runSingleCameraPipelineH264();
            }

            if (status == DW_END_OF_STREAM) {
                std::cout << "Camera reached end of stream" << std::endl;
                dwSensor_reset(gCameraSensor);
            }
            else if (status != DW_SUCCESS) {
                gRun = false;
            }

            gWindow->swapBuffers();
        }
    }

    release();

    // release framework
    releaseSampleApp();

    return 0;
}
