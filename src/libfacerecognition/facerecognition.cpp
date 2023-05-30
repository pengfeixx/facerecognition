#include "facerecognition.h"

#include <ncnn/mat.h>

FaceRecognition::FaceRecognition()
{
    // result = QImage(112, 112, QImage::Format_RGB888);
    m_result = cv::Mat(112, 112, CV_8UC3);
}

void FaceRecognition::setImage(const FaceBox &box, const cv::Mat &image)
{
    boxAlign = box;
    // imageAlign = image.convertToFormat(QImage::Format_RGB888);
    m_imageAlign = image;
}

void FaceRecognition::analyze()
{
    ncnn::Mat src = ncnn::Mat::from_pixels(m_imageAlign.data, ncnn::Mat::PIXEL_RGB, m_imageAlign.rows, m_imageAlign.cols);

    //人脸区域在原图上的坐标
    float point_src[10] = {
        static_cast<float>(boxAlign.keyPoints[0].x), static_cast<float>(boxAlign.keyPoints[0].y),
        static_cast<float>(boxAlign.keyPoints[1].x), static_cast<float>(boxAlign.keyPoints[1].y),
        static_cast<float>(boxAlign.keyPoints[2].x), static_cast<float>(boxAlign.keyPoints[2].y),
        static_cast<float>(boxAlign.keyPoints[3].x), static_cast<float>(boxAlign.keyPoints[3].y),
        static_cast<float>(boxAlign.keyPoints[4].x), static_cast<float>(boxAlign.keyPoints[4].y)
    };

    float point_dst[10] = { // +8 是因为我们处理112*112的图，不加则是 112*96
        30.2946f + 8.0f, 51.6963f,
        65.5318f + 8.0f, 51.5014f,
        48.0252f + 8.0f, 71.7366f,
        33.5493f + 8.0f, 92.3655f,
        62.7299f + 8.0f, 92.2041f,
    };

    //计算变换矩阵
    float tm_inv[6];
    ncnn::get_affine_transform(point_dst, point_src, 5, tm_inv);

    //执行变换，结果会直接刷入result
    ncnn::warpaffine_bilinear_c3(m_imageAlign.data, m_imageAlign.rows, m_imageAlign.cols, m_imageAlign.step[0],
                                 m_result.data, m_result.rows, m_result.cols, m_result.step[0], tm_inv);
}

cv::Mat FaceRecognition::getAlignResult()
{
    return m_result;
}
