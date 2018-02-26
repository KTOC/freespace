# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

@page dwx_freespace_detection_sample Free-Space Detection Sample (FreeSpaceNet)

Drive-able collision-free space, i.e. the space that can be immediately reached without
collision, provides critical information for the navigation in autonomous driving.
This free-space example demonstrates the NVIDIA end-to-end technique of detecting the
collision-free space in the road scenario. The problem is modeled within a deep
neural network (FreeSpaceNet), with the input being a three-channel RCB image and
the output being a boundary across the image from the left to the right. The boundary
separates the obstacle from open road space. In parallel, each pixel on the
boundary is associated with one of the four semantic labels:
- vehicle
- pedestraian
- curb
- other

This FreeSpaceNet sample has been trained using RCB images
with moderate augmentation.

This sample streams a H.264 or RAW video and computes the free
space boundary on each frame. The sample can also be operated with cameras.

![Free-Space Detection Sample](sample_freespace_detection.png)

### Sensor details ####

The image datasets used to train FreeSpaceNet have been captured using a View Sekonix
Camera Module (SS3323) with AR0231 RCCB sensor. The camera is mounted high up at the
rear-view mirror position. Demo videos are captured at 2.3 MP.

To achieve the best free-space detection performance, NVIDIA recommends to adopt
a similar camera setup and align the video center vertically with the horizon before
recording new videos.

### Sample ###

The sample H264 video and camera calibration files are located at:

    sdk/data/samples/freespace/

The latency of the sample FreeSpaceNet model:
- NVIDIA DRIVE PX<sup>&trade;</sup> 2 with GP106: 3.98 milliseconds
- NVIDIA DRIVE PX 2 with iGPU: 13.82 milliseconds

#### Running the Sample

The command lines for running the sample on Linux:

    ./sample_freespace_detection --video=<video file.h264> --rig=<calibration file.xml>
or

    ./sample_freespace_detection --video=<video file.raw> --rig=<calibration file.xml>

The command line for running the sample on NVIDIA DRIVE PX 2 with cameras:

    ./sample_freespace_detection --input-type=camera --camera-type=<camera_type> --csi-port=<csi_port> --rig=<calibration file.xml>

@note The free-space detection sample directly resizes video frames to the network input
resolution. Therefore, to get the best performance, it is suggested to use videos with
similar aspect ratio to the demo video. Or you can set the Region of Interest (ROI) to perform
inference on a sub-window of the full frame.

#### Output

The free-space detection sample:
- Creates a window.
- Displays a video.
- Overlays ploylines for the detected free-space boundary points.
- Computes boundary points in car coorindate system, if a valid camera calibration file is provided.

The colors of the ploylines represent the types of obstacle the boundary interface with:

- Red: Vehicle
- Green: Curb
- Blue: Pedestrian
- Yellow: Other
