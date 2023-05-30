#include "liveness.h"

#include <ncnn/net.h>
#include <ncnn/layer.h>

#include <cstdio>

FaceLiveness::FaceLiveness()
{
    //基础参数设置
    ncnn::Option option;
    option.lightmode = true;
    option.num_threads = 2;

    detection = new ncnn::Net;
    detection->opt = option;
    int ret = detection->load_param("/usr/share/facerecognition/models/detection.param");
    ret = detection->load_model("/usr/share/facerecognition/models/detection.bin");

    loadModels();
}

// void FaceAnit::setImage(const QImage &image)
// {
//     imageRec = image.convertToFormat(QImage::Format_RGB888);
// }

void FaceLiveness::setImage(const cv::Mat &image)
{
    m_imageRec = image;
}

void FaceLiveness::analyze()
{
    //清理已有数据
    result.clear();
    int input_size = 192;
    float threshold = 0.6f;
    float confidence = 0.f;

    int w = m_imageRec.cols;
    int h = m_imageRec.rows;
    float aspect_ratio = w / (float)h;

    int input_width = static_cast<int>(input_size * sqrt(aspect_ratio));
    int input_height = static_cast<int>(input_size / sqrt(aspect_ratio));

    //执行分析

    ncnn::Mat in_pad = ncnn::Mat::from_pixels_resize(m_imageRec.data, ncnn::Mat::PIXEL_BGR,
                                                     w, h, input_width, input_height);

    const float mean_val[3] = {104.f, 117.f, 123.f};
    in_pad.substract_mean_normalize(mean_val, nullptr);

    auto extractor = detection->create_extractor();
    extractor.input("data", in_pad);

    ncnn::Mat out;
    extractor.extract("detection_out", out);

    std::vector<FaceBox> boxes;
    for (int i = 0; i < out.h; ++i) {
        const float* values = out.row(i);
        float confidence = values[1];

        if(confidence < threshold) continue;

        FaceBox box;
        box.confidence = confidence;
        box.x1 = values[2] * w;
        box.y1 = values[3] * h;
        box.x2 = values[4] * w;
        box.y2 = values[5] * h;

        // square
        float box_width = box.x2 - box.x1 + 1;
        float box_height = box.y2 - box.y1 + 1;

        float size = (box_width + box_height) * 0.5f;

        if(size < 64) continue;

        float cx = box.x1 + box_width * 0.5f;
        float cy = box.y1 + box_height * 0.5f;

        box.x1 = cx - size * 0.5f;
        box.y1 = cy - size * 0.5f;
        box.x2 = cx + size * 0.5f - 1;
        box.y2 = cy + size * 0.5f - 1;

        boxes.emplace_back(box);
    }

    for (int i = 0;i < nets.size(); i++) {
        cv::Rect newBox = getNewBox(boxes[0], w, h, i);
        // QImage img = imageRec.copy(newBox).scaled(QSize(80, 80));
        cv::Mat img;
        cv::resize(m_imageRec(newBox), img, cv::Size(80, 80));
        ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_RGB, img.rows, img.cols);

        // inference
        ncnn::Extractor extractor = nets[i]->create_extractor();
        extractor.set_light_mode(true);
        extractor.set_num_threads(2);

        extractor.input("data", in);
        ncnn::Mat out;
        extractor.extract("softmax", out);

        confidence += out.row(0)[1];
        result.push_back(out.row(0)[1]);
    }

    confidence /= 2;
}

std::vector<float> FaceLiveness::getRecResult()
{
    return result;
}

void FaceLiveness::loadModels()
{
    ncnn::Net *net1 = new ncnn::Net();
    ncnn::Net *net2 = new ncnn::Net();

    //基础参数设置
    ncnn::Option option;
    option.lightmode = true;
    option.num_threads = 2;
    net1->opt = option;
    net2->opt = option;

    //所有的设置都必须在加载模型前完成
    net1->load_param("/usr/share/facerecognition/models/liveness_1.param");
    net1->load_model("/usr/share/facerecognition/models/liveness_1.bin");

    net2->load_param("/usr/share/facerecognition/models/liveness_2.param");
    net2->load_model("/usr/share/facerecognition/models/liveness_2.bin");

    nets.push_back(net1);
    nets.push_back(net2);
}

cv::Rect FaceLiveness::getNewBox(FaceBox &box, int w, int h, int index)
{
    float scale = 2.7;
    if (index == 1)
        scale = 4.0;
    int x = static_cast<int>(box.x1);
    int y = static_cast<int>(box.y1);
    int box_width = static_cast<int>(box.x2 - box.x1 + 1);
    int box_height = static_cast<int>(box.y2 - box.y1 + 1);

    scale = std::min(
            scale, std::min((w - 1) / (float) box_width, (h - 1) / (float) box_height)
    );

    int box_center_x = box_width / 2 + x;
    int box_center_y = box_height / 2 + y;

    int new_width = static_cast<int>(box_width * scale);
    int new_height = static_cast<int>(box_height * scale);

    int left_top_x = box_center_x - new_width / 2;
    int left_top_y = box_center_y - new_height / 2;
    int right_bottom_x = box_center_x + new_width / 2;
    int right_bottom_y = box_center_y + new_height / 2;

    if (left_top_x < 0) {
        right_bottom_x -= left_top_x;
        left_top_x = 0;
    }

    if (left_top_y < 0) {
        right_bottom_y -= left_top_y;
        left_top_y = 0;
    }

    if (right_bottom_x >= w) {
        int s = right_bottom_x - w + 1;
        left_top_x -= s;
        right_bottom_x -= s;
    }

    if (right_bottom_y >= h) {
        int s = right_bottom_y - h + 1;
        left_top_y -= s;
        right_bottom_y -= s;
    }

    return cv::Rect(left_top_x, left_top_y, new_width, new_height);
}
