freespace输出
========
# 1，free space单独编译
将free space例程从所有例程中单独列出进行编译。

首先，将samples路径下的文件复制到新的newsamples文件夹中，清空新的src文件夹
```
cd /usr/local/driveworks/
sudo cp -r samples newsamples && cd newsamples/src && sudo rm -fr *
```
复制需要的文件到新src:
```
cd /samples/SOURCESsudo cp -r framework dnn freespace ../newsamples/src
cd ../newsamples/src
```
新建newfreespace文件目录，并且将相关文件复制到newfreespace：
```
sudo makdir newfreespce
sudo cp -r dnn/dnn_common freespace newfreespce/
sudo rm -fr dnn freespace
```
此时，src文件夹下有两个文件夹framework和newfreespace，newfreespace下有dnn_common和newfreespace两个文件夹。

编辑newsamples下CMakeLists.txt
```
cd /usr/local/driveworks/newsamples
sudo vim CMakeLists.txt
```
用下面内容代替，set(SAMPLES framework;egomotion;sensors;features;mapping;rigconfiguration;renderer;sfm;dnn;laneDetection;colorcorrection;isp;rectifier;ipc;hello_world;image;stereo;freespace;drivenet;maps;template;icp;lidar_accumulator;calibration;vehicleio)
```
set(SAMPLES framework;newfreespce)
```
为src/newfreespace创建CMakeLists.txt，并添加如下内容：
```
add_subdirectory(dnn_common)
add_subdirectory(newfreespce)
```
修改src/newfreespce/newfreespce下的CMakeLists.txt文件，将project(sample_freespace_detection C CXX)替换为project(sample_newfreespace_detection C CXX)。

然后即可编译运行：
```
cd /usr/local/driveworks/newsamples
sudo mkdir build-host
cd build-host
sudo cmake ..
sudo make -j
sudo make install
```
然后即可运行程序：
```
cd install/bin
./sample_newfreespace_detection
```
