#pragma once

// #include <QImage>
#include <vector>
#include <opencv2/opencv.hpp>

namespace ncnn {
class Net;
}

class FaceLiveness {
    struct FaceBox {
        float confidence;
        float x1;
        float y1;
        float x2;
        float y2;
    };

public:
    FaceLiveness();

    // void setImage(const QImage &image);
    void setImage(const cv::Mat &image);
    void analyze();
    std::vector<float> getRecResult();

private:
    void loadModels();
    cv::Rect getNewBox(FaceBox &box, int w, int h, int index);

private:
    std::vector<ncnn::Net *> nets;
    ncnn::Net *detection;
    // QImage imageRec;
    cv::Mat m_imageRec;
    std::vector<float> result;
};
