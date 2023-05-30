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

    /**
     * @brief Get the Facial Range object
     * 
     * @return cv::Rect 
     */
    cv::Rect getFacialRange(const cv::Mat &image);

    bool isliving(const cv::Mat &image);

    inline void setThreshold(const float thre);

private:
    FaceLiveness* m_faceLiveness;        //人脸活体检测
    FaceDetection* m_faceDetection;      //人脸检测（包含五点）
    FaceFeature* m_faceFeature;          //特征识别图像转换
    FaceRecognition* m_faceRecognition;  //面部特征识别

    float m_threshold;
};