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
#include "FeatureTracker2D.hpp"
#include <iostream>
#include <algorithm>

FeatureTracker2D::FeatureTracker2D(dwContextHandle_t &context, cudaStream_t &stream,
                                   uint32_t maxFeatureCount, uint32_t historyCapacity,
                                   uint32_t iterationsLK, uint32_t windowSizeLK,
                                   uint32_t imageWidth, uint32_t imageHeight, uint32_t maskHeight)
    : m_context(context)
    , m_cudaStream(stream)
    , m_imageWidth(imageWidth)
    , m_imageHeight(imageHeight)
    , m_maxFeatureCount(maxFeatureCount)
    , m_historyCapacity(historyCapacity)
    , m_iterationsLK(iterationsLK)
    , m_windowSizeLK(windowSizeLK)
    , m_featureMask(nullptr)
{
    cudaMalloc((void**)&m_d_validFeatureCount, sizeof(uint32_t));
    cudaMalloc((void**)&m_d_validFeatureIndexes, m_maxFeatureCount * sizeof(uint32_t));
    cudaMalloc((void**)&m_d_invalidFeatureCount, sizeof(uint32_t));
    cudaMalloc((void**)&m_d_invalidFeatureIndexes, m_maxFeatureCount * sizeof(uint32_t));

    if (!init(maskHeight)) {
        std::cerr << "fail to initialize feature tracker\n";
    }
}

FeatureTracker2D::~FeatureTracker2D()
{
    dwPyramid_release(&m_pyramidCurrent);
    dwPyramid_release(&m_pyramidPrevious);
    dwFeatureList_release(&m_featureList);
    dwFeatureTracker_release(&m_tracker);
    cudaFree(m_d_validFeatureCount);
    cudaFree(m_d_validFeatureIndexes);
    cudaFree(m_d_invalidFeatureCount);
    cudaFree(m_d_invalidFeatureIndexes);
    cudaFree(m_featureMask);
}

void FeatureTracker2D::reset()
{
    dwPyramid_reset(m_pyramidCurrent);
    dwPyramid_reset(m_pyramidPrevious);
    dwFeatureList_reset(m_featureList);

    //Note: do not reset the tracker to keep the mask
    //dwFeatureTracker_reset(m_tracker);
}

bool FeatureTracker2D::init(uint32_t maskHeight)
{
    //Tracker
    dwStatus result;
    dwFeatureTrackerConfig trackerConfig;
    trackerConfig.imageWidth             = m_imageWidth;
    trackerConfig.imageHeight            = m_imageHeight;
    trackerConfig.maxFeatureCount        = m_maxFeatureCount;
    trackerConfig.interationsLK          = m_iterationsLK;
    trackerConfig.windowSizeLK           = m_windowSizeLK;
    trackerConfig.detectorScoreThreshold = 0.1f;

    //Print out settings
    std::cout << "Feature Tracker settings: "
              << ", width=" << trackerConfig.imageWidth
              << ", imageHeight=" << trackerConfig.imageHeight
              << ", maxFeatureCount=" << trackerConfig.maxFeatureCount
              << ", iterationsLK=" << trackerConfig.interationsLK
              << ", windowSizeLK=" << trackerConfig.windowSizeLK
              << ", detectorScoreThreshold=" << trackerConfig.detectorScoreThreshold
              << std::endl;

    result = dwFeatureTracker_initialize(&m_tracker, m_context, m_cudaStream, trackerConfig);
    if (result != DW_SUCCESS) {
        std::cout << "failed in dwFeatureTracker_initialize: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    if (maskHeight > m_imageHeight) {
        std::cout << "wrong mask size" << std::endl;
        return false;
    }

    //Mask ROI for feature tracker
    size_t pitch = 0;
    cudaMallocPitch(&m_featureMask, &pitch, m_imageWidth, m_imageHeight);
    cudaMemset((uint8_t *)m_featureMask, 255, pitch * m_imageHeight);
    cudaMemset((uint8_t *)m_featureMask, 0, pitch * maskHeight);

    result = dwFeatureTracker_setMask((uint8_t *)m_featureMask, (uint32_t)pitch,
                                      m_imageWidth, m_imageHeight, m_tracker);
    if (result != DW_SUCCESS) {
        std::cout << "fail to set mask: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    result = dwFeatureList_initialize(&m_featureList, m_context, m_cudaStream, trackerConfig.maxFeatureCount, m_historyCapacity,
                                      trackerConfig.imageWidth, trackerConfig.imageHeight);
    if (result != DW_SUCCESS) {
        std::cout << "fail to init features list: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    result = dwFeatureList_getDataBasePointer(&m_d_featureDataBase, &m_featureDataSize, m_featureList);
    if (result != DW_SUCCESS) {
        std::cout << "fail to get feature data: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    result = dwFeatureList_getDataPointers(&m_d_featureData, m_d_featureDataBase, m_featureList);
    if (result != DW_SUCCESS) {
        std::cout << "fail to get feature locations: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    m_featureDataBase.reset(new uint8_t[m_featureDataSize]);

    result = dwFeatureList_getDataPointers(&m_featureData, m_featureDataBase.get(), m_featureList);
    if (result != DW_SUCCESS) {
        std::cout << "fail to get feature locations: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    dwPyramidConfig pyramidConfig;
    pyramidConfig.width      = trackerConfig.imageWidth;
    pyramidConfig.height     = trackerConfig.imageHeight;
    pyramidConfig.levelCount = 6;
    pyramidConfig.dataType   = DW_TYPE_UINT8;
    result = dwPyramid_initialize(&m_pyramidPrevious, m_context, m_cudaStream, pyramidConfig);
    if (result != DW_SUCCESS) {
        std::cout << "fail to init previous pyramid: " << dwGetStatusName(result) << std::endl;
        return false;
    }
    result = dwPyramid_initialize(&m_pyramidCurrent, m_context, m_cudaStream, pyramidConfig);
    if (result != DW_SUCCESS) {
        std::cout << "fail to init current pyramid: " << dwGetStatusName(result) << std::endl;
        return false;
    }
    return true;
}

bool FeatureTracker2D::track(uint32_t *nFeatures,
                             std::vector<float32_t> *previousLocations, std::vector<float32_t> *currentLocations,
                             std::vector<dwFeatureStatus> *statuses,
                             const dwImageCUDA &image)
{
    if (!nFeatures) {
        std::cout << "FeatureTracker2D::track, null nFeatures\n";
        return false;
    }
    if (previousLocations->size() / 2 != statuses->size() || currentLocations->size() / 2 != statuses->size()) {
        std::cout << "FeatureTracker2D::track, feature dimension mis-matches with statuses\n";
        return false;
    }

    std::swap(m_pyramidCurrent, m_pyramidPrevious);

    //Build pyramid
    dwImageCUDA planeY;
    dwImageCUDA_getPlaneAsImage(&planeY, &image, 0);

    dwStatus result;
    result = dwPyramid_build(&planeY, m_pyramidCurrent);
    if (result != DW_SUCCESS) {
        std::cout << "fail to build pyramid: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    //track
    dwFeatureTracker_trackFeatures(m_featureList, m_pyramidPrevious, m_pyramidCurrent, 0, m_tracker);

    //Get feature info to CPU
    cudaMemcpy(m_featureDataBase.get(), m_d_featureDataBase, m_featureDataSize, cudaMemcpyDeviceToHost);
    m_featureCount = *m_featureData.featureCount;
    *nFeatures = getFeatures(previousLocations, currentLocations, statuses);

    //filter too close ones
    dwFeatureList_proximityFilter(m_featureList);

    //Determine which features to throw away
    dwFeatureList_selectValid(m_d_validFeatureCount, m_d_validFeatureIndexes,
                              m_d_invalidFeatureCount, m_d_invalidFeatureIndexes,
                              m_featureList);

    //Compact list
    dwFeatureList_compact(m_featureList,
                          m_d_validFeatureCount, m_d_validFeatureIndexes,
                          m_d_invalidFeatureCount, m_d_invalidFeatureIndexes);

    dwFeatureTracker_detectNewFeatures(m_featureList, m_pyramidCurrent, m_tracker);

    return true;
}

uint32_t FeatureTracker2D::getFeatures(std::vector<float32_t> *previousLocations, std::vector<float32_t> *currentLocations,
                                       std::vector<dwFeatureStatus> *statuses)
{
    uint32_t maxFeatureCount;
    uint32_t historyCapacity;
    dwFeatureList_getFeatureListSize(&maxFeatureCount, &historyCapacity, m_featureList);

    uint32_t currentTimeIdx;
    dwFeatureList_getCurrentTimeIdx(&currentTimeIdx, m_featureList);

    uint32_t previousTimeIdx = (currentTimeIdx + 1) % historyCapacity;

    dwVector2f *preLocations = &m_featureData.locationHistory[previousTimeIdx * m_maxFeatureCount];
    dwVector2f *curLocations = &m_featureData.locationHistory[currentTimeIdx * m_maxFeatureCount];

    uint32_t newSize = std::min(static_cast<uint32_t>(statuses->size()), m_featureCount);
    for (uint32_t i = 0; i < newSize; i++) {
        previousLocations->at(2 * i)     = preLocations[i].x;
        previousLocations->at(2 * i + 1) = preLocations[i].y;
        currentLocations->at(2 * i)      = curLocations[i].x;
        currentLocations->at(2 * i + 1)  = curLocations[i].y;

        statuses->at(i) = m_featureData.statuses[i];
    }
    return newSize;
}
