#pragma once

#include "facedetection.h"
#include <opencv2/opencv.hpp>

class FaceRecognition
{
public:
    FaceRecognition();

    void setImage(const FaceBox &box, const cv::Mat &image);
    void analyze();
    cv::Mat getAlignResult();

private:
    FaceBox boxAlign;
    // QImage imageAlign;
    // QImage result;

    cv::Mat m_result;
    cv::Mat m_imageAlign;
};
