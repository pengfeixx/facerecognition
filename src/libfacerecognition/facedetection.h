// SPDX-FileCopyrightText: 2020 - 2022 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// #include <QImage>
#include <vector>
#include <opencv2/opencv.hpp>

namespace ncnn {
class Net;
}

struct FaceBox {
    cv::Rect rect;
    std::vector<cv::Point> keyPoints;
    double prob;
};

class FaceDetection
{
public:
    FaceDetection();

    void setImage(const cv::Mat &image);
    void analyze();
    std::vector<FaceBox> getDetResult();

private:
    ncnn::Net *detNet;
    // QImage imageDet;
    cv::Mat m_imageDet;
    std::vector<FaceBox> result;   //人脸检测的识别结果
};
