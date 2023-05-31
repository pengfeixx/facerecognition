#include "face.h"

#include "../libfacerecognition/facedetection.h"
#include "../libfacerecognition/facefeature.h"
#include "../libfacerecognition/facerecognition.h"
#include "../libfacerecognition/liveness.h"


#include <numeric>

Face::Face()
{
    m_threshold = 0.85;

    m_faceLiveness = new FaceLiveness;
    m_faceDetection = new FaceDetection;
    m_faceFeature = new FaceFeature;
    m_faceRecognition = new FaceRecognition;
}

Face::~Face()
{

}

bool Face::isliving(const cv::Mat &image)
{
    if (!image.data)
        return false;

    m_faceLiveness->setImage(image);
    m_faceLiveness->analyze();

    std::vector<float> resultSet = m_faceLiveness->getRecResult();
    float mean = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0) / resultSet.size();

    return mean >= m_threshold;
}

inline void Face::setThreshold(const float thre)
{
    m_threshold = thre;
}

double Face::calculateSimilarity(const std::vector<double> &eigenvalue1, const std::vector<double> &eigenvalue2)
{
    if (eigenvalue1.empty() || eigenvalue2.empty())
        return -1;

        //计算相似度
    double sum_top = 0;
    double sum_l = 0;
    double sum_r = 0;
    int k=0;
    for (size_t i = 0; i != eigenvalue1.size(); ++i) {
        sum_top += eigenvalue1[i] * eigenvalue2[i];   //特征相乘
        sum_l += eigenvalue1[i] * eigenvalue1[i];     //平方
        sum_r += eigenvalue2[i] * eigenvalue2[i];     //平方
        k++;
    }
    printf("特征大小%d\n",k);

    double feat_result = sum_top / (std::sqrt(sum_l) * std::sqrt(sum_r));
    //printf("相似度计算：%lf\n", feat_result);
    feat_result=feat_result*0.5+0.5;
    printf("相似度计算：%lf\n", feat_result);
    return feat_result;
}

std::vector<cv::Rect> Face::getFacialRange(const cv::Mat &image)
{
    if (!image.data)
        return std::vector<cv::Rect> ();

    m_faceDetection->setImage(image);
    m_faceDetection->analyze();

    std::vector<cv::Rect> faceRect;

    for(auto var : m_faceDetection->getDetResult())
    {
        faceRect.push_back(var.rect);
    }

    return faceRect;
}

std::vector<FaceBox> Face::getAllFaces(const cv::Mat &image)
{
    if (!image.data)
        return std::vector<FaceBox>();

    m_faceDetection->setImage(image);
    m_faceDetection->analyze();

    std::vector<FaceBox> faceRect = m_faceDetection->getDetResult();
    return faceRect;
}

std::vector<double> Face::getEigenvalue(const cv::Mat &image)
{
    if (!image.data)
        return std::vector<double>();

    std::vector<FaceBox> facelist = getAllFaces(image);

    m_faceRecognition->setImage(facelist[0], image);
    m_faceRecognition->analyze();

    std::vector<double> eigenvalue = m_faceRecognition->getAlignResult();
    return eigenvalue;
}

bool Face::isSamePerson(const cv::Mat &image1, const cv::Mat &image2)
{
    if (!image1.data || !image2.data)
        return false;

    std::vector<double> eigenvalue1 = getEigenvalue(image1);
    std::vector<double> eigenvalue2 = getEigenvalue(image2);

    double similarity = calculateSimilarity(eigenvalue1, eigenvalue2);

    return similarity >= m_threshold;
}
