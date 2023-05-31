#include <opencv2/opencv.hpp>

class FaceLiveness;
class FaceDetection;
class FaceFeature;
class FaceRecognition;

class Face 
{
public:
    Face();
    ~Face();

    std::vector<cv::Rect> getFacialRange(const cv::Mat &image);
    std::vector<FaceBox> getAllFaces(const cv::Mat &image);

    std::vector<double> getEigenvalue(const cv::Mat &image);

    bool isSamePerson(const cv::Mat &image1, const cv::Mat &image2);

    bool isliving(const cv::Mat &image);

    inline void setThreshold(const float thre);

private:
    double calculateSimilarity(const std::vector<double> &eigenvalue1, const std::vector<double> &eigenvalue2);

private:
    FaceLiveness* m_faceLiveness;        //人脸活体检测
    FaceDetection* m_faceDetection;      //人脸检测（包含五点）
    FaceFeature* m_faceFeature;          //特征识别图像转换
    FaceRecognition* m_faceRecognition;  //面部特征识别

    float m_threshold;
};