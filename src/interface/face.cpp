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

cv::Rect Face::getFacialRange(const cv::Mat &image)
{

}
