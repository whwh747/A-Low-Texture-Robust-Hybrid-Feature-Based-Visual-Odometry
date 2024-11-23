//
// Created by wanghe on 23-7-10.
//

#ifndef ORB_SLAM2_SURFACENORMAL_H
#define ORB_SLAM2_SURFACENORMAL_H

#include <stdio.h>
#include <fstream>
#include<iostream>
#include <numeric>
//#include "../include/Image_ScrollBar.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cxcore.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <eigen3/Eigen/Core>
#include "auxiliar.h"

class SurfaceNormal {
public:
    cv::Point3f normal;
    cv::Point3f cameraPosition;
    cv::Point2i FramePosition;

    SurfaceNormal() {}
};

#endif //ORB_SLAM2_SURFACENORMAL_H
