#pragma once

// #include <QImage>
#include <vector>
#include <opencv2/opencv.hpp>

namespace ncnn {
class Net;
}

class FaceFeature
{
public:
    FaceFeature();

    void setImage(const cv::Mat &image);
    void analyze();
    std::vector<double> getRecResult();

private:
    ncnn::Net *recNet;
    // QImage imageRec;
    cv::Mat m_image;
    std::vector<double> result;
};
