# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

@page dwx_object_tracker_sample Basic Object Tracker Sample

The basic object tracker sample demonstrates the 2D object tracking capabilities
of the @ref boxtracker_group module.
It reads video streams sequentially. For each frame, it detects the
object locations and tracks the objects between video frames. Currently, the
object tracker resorts to image feature detection and tracking. The tracker uses
features motion to predict the object location.

![Object tracker on a single H.264 stream](sample_object_tracker.png)

## Running the Sample

The default usage is:

     ./sample_object_tracker

The sample can process up to 4 video streams. The video file must be an H.264 stream.
Video containers such as MP4, AVI, MKV, etc. are not supported.

The sample usage with NVIDIA<sup>&reg;</sup> TensorRT<sup>&tm;</sup> is:

    ./sample_object_tracker --video=video_file.h264
                            --tensorRT_model=TensorRT_model_file --tracker=config_file.txt

Note that this sample loads its DataConditioner parameters from DNN metadata. This metadata
can be provided to DNN module by placing the json file in the same directory as the model file
with json extension; i.e. TensorRT_model_file.json.
Please see data/samples/detector/tensorRT_model.bin.json as an example of DNN metadata file.

The tracker configuration file includes the following parameters:

- `maxFeatureCount`: Max number of features to track. Set this value between 1000 and 8000.
- `iterationLK`: Optimization iteration to locate features. By default, it is set to 40.
- `windowSizeLK`: Search window for Lucas Kanade features. By default, it is set to 14.
- `detectInterval`: The frequency of calling detection routine. Set this value between 1 and n.
- `trackInterval`: The life span of object tracker. Default is 10. The tracker resets after this interval.
- `maxObjectCount`: Max number of 2D objects to track.
- `maxFeatureCountPerObject`: Max number of features to track for each object. By default, it is set to 500.
- `maxObjectImageScale`: Max image scale of the object to track. Set this value between 0 and 1. The multiplication
                         of this parameter with the image size gives the maximum object size, in pixels.
- `minObjectImageScale`: Min image scale of the object to track. Set this value between 0 and 1. The multiplication
                         of this parameter with the image size gives the maximum object size, in pixels.
- `similarityThreshold`: Similarity threshold to group close-by object bounding boxes. By default, it is set to 0.2.
- `groupThreshold`: Min possible number of bounding boxes minus 1. By default, it is set to 0.

## Output

The sample creates a window, displays the video streams, and overlays the list
of features and detected/tracked bounding boxes of the objects with IDs.

The color coding of the overlay is:

- Red squares: Successfully tracked bounding boxes.
- Red cross: Successfully tracked 2D features.
