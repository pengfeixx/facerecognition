#include "facefeature.h"

#include <ncnn/net.h>
#include <ncnn/layer.h>

#include <cstdio>

static void pretty_print(const ncnn::Mat &m)
{
    for (int q = 0; q < m.c; q++) {
        const float *ptr = m.channel(q);
        for (int z = 0; z < m.d; z++) {
            for (int y = 0; y < m.h; y++) {
                for (int x = 0; x < m.w; x++) {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

FaceFeature::FaceFeature()
    : recNet(new ncnn::Net)
{
    //基础参数设置
    ncnn::Option option;
    option.lightmode = true;
    option.num_threads = 2;
    recNet->opt = option;

    //所有的设置都必须在加载模型前完成
    recNet->load_param("/usr/share/facerecognition/models/facefrature.param");
    recNet->load_model("/usr/share/facerecognition/models/facefrature.bin");
}

void FaceFeature::setImage(const cv::Mat &image)
{
    // imageRec = image.convertToFormat(QImage::Format_RGB888);
    m_image = image;
}

std::vector<double> FaceFeature::getRecResult()
{
    return result;
}

void FaceFeature::analyze()
{
    //清理已有数据
    result.clear();

    //执行分析

    ncnn::Mat in_pad = ncnn::Mat::from_pixels(m_image.data, ncnn::Mat::PIXEL_BGR, m_image.rows, m_image.cols);
    const float meanValues[3] = { 255.0f / 2.0f ,255.0f / 2.0f,255.0f / 2.0f};
    const float normValues[3] = { 2.0f / 255.0f ,2.0f / 255.0f,2.0f / 255.0f};

    in_pad.substract_mean_normalize(meanValues, normValues);

    auto extractor = recNet->create_extractor();
    extractor.input("in0", in_pad);

    ncnn::Mat out;
    extractor.extract("out0", out);

    for (int i = 0; i != out.w; ++i) {
        result.emplace_back(out[static_cast<size_t>(i)]);
    }
}
