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
#ifndef FEATURE_TRACKER_2D_HPP__
#define FEATURE_TRACKER_2D_HPP__

// C API
#include <dw/core/Types.h>
#include <dw/features/Features.h>

#include <memory> 
#include <vector>
class FeatureTracker2D
{
  public:
    /* constructor
    *  maxFeatureCount: Maximum features to track, 0 < maxFeatureCount < 8000
    *  historyCapacity: Capacity of feature history circular buffer
    *  iterationsLK: Lucas-Kanade feature tracker iterations, set it between 6 and 14
    *  windowSizeLK: Spatial window size for the feature tracker, typical size 5-9
    *  imageWidth: Width of the input image frame
    *  imageHeight: Height of the input image frame
    *  maskHeight: Height of the region of interest (ROI), set it between 0 and 1,
    *              maskHeight = 0.5, the feature tracker will process
    *              half image counting from the bottom
    *              maskHeight = 1/3, the feature tracker will process
    *              imageWidth x 2/3 imageHeight counting from the bottom
    */
    FeatureTracker2D(dwContextHandle_t &context, cudaStream_t &stream,
                     uint32_t maxFeatureCount, uint32_t historyCapacity,
                     uint32_t iterationsLK, uint32_t windowSizeLK,
                     uint32_t imageWidth, uint32_t imageHeight, uint32_t maskHeight);

    // destructor
    ~FeatureTracker2D();

    /*
    * Resets the object to the same status as after construction.
    */
    void reset();

    /* track and detect the features
    *  nFeatures: total number of tracked features
    *  previousLocations: previous feature locations with 2 * nFeatures items
    *  currentLocations: current feature locations with 2 * nFeatures items
    *  statuses: The feature statues
    */
    bool track(uint32_t *nFeatures,
               std::vector<float32_t> *previousLocations, std::vector<float32_t> *currentLocations,
               std::vector<dwFeatureStatus> *statuses,
               const dwImageCUDA &image);

  private:
    // functions
    bool init(uint32_t maskHeight);

    uint32_t getFeatures(std::vector<float32_t> *previousLocations, std::vector<float32_t> *currentLocations,
                         std::vector<dwFeatureStatus> *statuses);

    // variables for stream and context
    dwContextHandle_t m_context;
    cudaStream_t m_cudaStream;

    // variables for image size
    uint32_t m_imageWidth;
    uint32_t m_imageHeight;

    // variables for image features and pyramids
    uint32_t m_maxFeatureCount;
    uint32_t m_historyCapacity;
    uint32_t m_iterationsLK;
    uint32_t m_windowSizeLK;
    dwFeatureTrackerHandle_t m_tracker;
    dwPyramidHandle_t m_pyramidPrevious;
    dwPyramidHandle_t m_pyramidCurrent;

    dwFeatureListHandle_t m_featureList;

    void *m_featureMask;
    void *m_d_featureDataBase;
    dwFeatureListPointers m_d_featureData;

    size_t m_featureDataSize;
    std::unique_ptr<uint8_t[]> m_featureDataBase;
    dwFeatureListPointers m_featureData;
    uint32_t m_featureCount;

    uint32_t *m_d_validFeatureCount;
    uint32_t *m_d_validFeatureIndexes;
    uint32_t *m_d_invalidFeatureCount;
    uint32_t *m_d_invalidFeatureIndexes;
};

#endif
