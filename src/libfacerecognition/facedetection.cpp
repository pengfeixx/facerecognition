// SPDX-FileCopyrightText: 2020 - 2022 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#include "facedetection.h"

#include <ncnn/net.h>
#include <ncnn/layer.h>

#include <cstdio>

#include <algorithm>    // std::min()
#include <math.h>       // ceil()
#include <algorithm>    // sort()

static void pretty_print(const ncnn::Mat &m)
{
    for (int q = 0; q < m.c; q++) {
        const float *ptr = m.channel(q);
        for (int z = 0; z < m.d; z++) {
            for (int y = 0; y < m.h; y++) {
                for (int x = 0; x < m.w; x++) {
                    fprintf(stderr, "%f ", ptr[x]);
                }
                ptr += m.w;
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "------------------------\n");
    }
}

FaceDetection::FaceDetection()
    : detNet(new ncnn::Net)
{
    //基础参数设置
    ncnn::Option option;
    option.lightmode = false;
    option.num_threads = 2;
    detNet->opt = option;

    //所有的设置都必须在加载模型前完成
    detNet->load_param("/usr/share/facerecognition/models/livedetection.param");
    detNet->load_model("/usr/share/facerecognition/models/livedetection.bin");
}

void FaceDetection::setImage(const cv::Mat &image)
{
    m_imageDet = image;
}

std::vector<FaceBox> FaceDetection::getDetResult()
{
    return result;
}

// cv::Mat qim2mat(QImage & qim)
// {
//     cv::Mat mat = cv::Mat(qim.height(), qim.width(),
//         CV_8UC3,(void*)qim.constBits(),qim.bytesPerLine());
//     return mat;
// }

//----------------------------------------
//  resize图片的大小
//  默认输入尺寸    [1280, 1280, 3]
//----------------------------------------
cv::Mat resize_image(cv::Mat &image,int dst_width=1280,int dst_height=1280)
{
    //Qimage转为cv::mat
    // cv::Mat img=qim2mat(image);
    //cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    //cv::imwrite("/home/hello/Desktop/face2/face2/save_img/1.png",img);
    //cv::imwrite("/home/hello/Desktop/face2/face2/save_img/2.png",img);
    //  qimage转为opencv
    //  获得照片的维度
    double iw = image.cols;  //宽度
    double ih = image.rows;   //高度

    //  获得scale参数，小的那一边的比例
    double scale = std::min(dst_width/iw,dst_height/ih);

    int nw = int(iw*scale);
    int nh = int(ih*scale);

    //  调整图片大小,不失真缩放方法
    cv::Mat resized_image;
    //  Size参数是cols,rows
    cv::resize(image, resized_image, cv::Size(nw, nh), cv::INTER_CUBIC);
    //  打印图片到本地
    //-------------------------------------------
    //cv::imwrite("/home/hello/Desktop/face2/face2/save_img/resize.png",resized_image);

    //  填充,在图片下部或者右部填充图片
    if(dst_width>resized_image.rows){
        cv::copyMakeBorder(resized_image,resized_image,0,dst_width-resized_image.rows,0,0,cv::BORDER_CONSTANT,(0,0,0));
    }
    if(dst_height>resized_image.cols){
        cv::copyMakeBorder(resized_image,resized_image,0,0,0,dst_height-resized_image.cols,cv::BORDER_CONSTANT,(0,0,0));
    }

    //cv::cvtColor(resized_image,resized_image,cv::COLOR_BGR2RGB);
    //-------------------------------------------
    // cv::imwrite("/home/uos/work/face/face2/save_img/resize2.png",resized_image);
    //cv::imshow("/home/hello/Desktop/face2/face2/save_img/resize3.png",resized_image);

    return resized_image;
}
//----------------------------------------
//  生成先验框
//----------------------------------------
std::vector<std::vector<double> >  get_anchors(int width=1280,int height=1280){
     // config区域
    std::vector<int> steps={8,16,32};
    std::vector<std::vector<int> > min_sizes={{16,32},{64,128},{256,512}};
    //  每个steps储存一次
    std::vector<std::vector<double> > feature_maps(steps.size());
    for(int i=0;i<steps.size();i++){
        int step=steps[i];
        //feature_maps[i]={ceil(width/step),ceil(height/step)};
        feature_maps[i].push_back(ceil(width/step));//取整问题
        feature_maps[i].push_back(ceil(height/step));
    }


    //  获取anchors ，遍历宽高除以步长
    std::vector<std::vector<double> > anchors;
    for(int k=0;k<feature_maps.size();k++)
    {
        std::vector<double> f=feature_maps[k];
        //  其中一种尺寸先验图
        std::vector<int> min_size=min_sizes[k];
        // feature_maps[i]的两个数字生产笛卡尔积   ，宽高的range,
        for(int i=0;i<f[0];i++){
            for(int j=0;j<f[1];j++){
                //  每个点有两个先验框
                for(int m=0;m<min_size.size();m++){
                    //  尺寸/高宽
                    double s_kx=min_size[m]/(double)height;
                    double s_ky=min_size[m]/(double)width;

                    double dense_cx=(j+0.5)*steps[k]/height;
                    double dense_cy=(i+0.5)*steps[k]/width;

                    //  添加到anchors中
                    std::vector<double> temp;
                    temp.push_back(dense_cx);
                    temp.push_back(dense_cy);
                    temp.push_back(s_kx);
                    temp.push_back(s_ky);

                    anchors.push_back(temp);
                }

            }
        }
    }

    return anchors;

}
//中心解码，宽高解码   ,结果，anchors, variance[0.1,0.2]
std::vector<std::vector<double> >  decode(cv::Mat &loc,std::vector<std::vector<double> > &priors, double variance_0=0.1,double variance_1=0.2){
      std::vector<std::vector<double> > res(priors.size(),std::vector<double>(4,0));
     // 遍历loc和priors应该一样
     for(int i=0;i<priors.size();i++){
         // 前两个计算 ,前两个+前两个*var1*后两个
         double loc_0=loc.at<float>(i,0);
         double loc_1=loc.at<float>(i,1);
         double loc_2=loc.at<float>(i,2);
         double loc_3=loc.at<float>(i,3);
         res[i][0]=priors[i][0]+loc_0*variance_0*priors[i][2];
         res[i][1]=priors[i][1]+loc_1*variance_0*priors[i][3];
         // 后面两个计算 ,后两个+e的后两个次方   *  var2
         res[i][2]=priors[i][2]*exp(loc_2*variance_1);
         res[i][3]=priors[i][3]*exp(loc_3*variance_1);

         // 前两个=前两个-（后两个/2）
         res[i][0]-=(res[i][2]/2);
         res[i][1]-=(res[i][3]/2);
         // 后两个=后两个+前两个
         res[i][2]+=res[i][0];
         res[i][3]+=res[i][1];
     }
     return res;
 }
//获得置信度
std::vector<double> get_confidence(cv::Mat &out_2){
    //  结果就是out_2的后一列
    std::vector<double> res;
    int j=out_2.rows;
    int k=0;
    for(int i=0;i<out_2.rows;i++){
        double c=out_2.at<float>(i,0);
        res.push_back(out_2.at<float>(i,1));
        k++;
    }

    return res;
}
//关键点解码 , 10列，对应5个坐标点
std::vector<std::vector<double> >  decode_landm(cv::Mat &pre,std::vector<std::vector<double> > priors, double variance_1=0.1,double variance_2=0.2){
     std::vector<std::vector<double> >  res(priors.size());
     // 遍历loc和priors应该一样
     for(int i=0;i<priors.size();i++){
         for(int j=0;j<10;j+=2){
             double pre_ij0=pre.at<float>(i,j);
             double pre_ij1=pre.at<float>(i,j+1);
             double num_1=priors[i][0]+pre_ij0*variance_1*priors[i][2];
             double num_2=priors[i][1]+pre_ij1*variance_1*priors[i][3];

             res[i].push_back(num_1);
             res[i].push_back(num_2);
         }
     }
     return res;
 }

double iou(std::vector<double> box1,std::vector<double> box2)
{
    double b1_x1=box1[0]; double b1_y1=box1[1]; double b1_x2=box1[2] ; double  b1_y2=box1[3];
    double b2_x1=box2[0]; double b2_y1=box2[1]; double b2_x2=box2[2] ; double  b2_y2=box2[3];

    double inter_rect_x1 = std::max(b1_x1, b2_x1);
    double inter_rect_y1 = std::max(b1_y1, b2_y1);
    double inter_rect_x2 = std::min(b1_x2, b2_x2);
    double inter_rect_y2 = std::min(b1_y2, b2_y2);

    double inter_area = std::max(inter_rect_x2 - inter_rect_x1, 0.0) * std::max(inter_rect_y2 - inter_rect_y1, 0.0);
    double area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1);
    double area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1);
    double iou=inter_area/std::max((area_b1 + area_b2 - inter_area), 1e-6);
    return iou;
}
//排序回调函数
bool confidence_comp(std::vector<double> i,std::vector<double> j)
{
    return (i[4]>j[4]);
}
//非极大抑制 置信度0.5, 重合度阈值0.3
std::vector<std::vector<double> > non_max_suppression(std::vector<std::vector<double> > boxes,std::vector<double> confidence,std::vector<std::vector<double> > landms ,double confidence_threshold=0.5 ,double overlap_threshold=0.3){
    double conf=confidence[0];
    //  输入的box,confidence,landms应该是1：1对应的，先组合起来，再排序,先不初始化维度
    std::vector<std::vector<double> > boxes_conf_landms;
    for(int i=0;i<boxes.size();i++)
    {
        //置信度小于阈值的就不加入了
        if(confidence[i]<confidence_threshold)
            continue;
        std::vector<double> temp;

        //box
        for(int j=0;j<4;j++){
            temp.push_back(boxes[i][j]);
        }
        //conf
        temp.push_back(confidence[i]);

        //landms
        for(int j=0;j<10;j++){
            temp.push_back(landms[i][j]);
        }
        boxes_conf_landms.push_back(temp);
    }
    int num=0;
    int iter=0;
    double con=confidence[0];
    //  先去掉threshold以下的数据,在遍历删除防止迭代器失效
    /*
    for(std::vector<std::vector<double> >::iterator it =boxes_conf_landms.begin();   it!=boxes_conf_landms.end();)
    {

        iter++;
        double c=(*it)[iter++];
        //如果置信度小就删除
        if((*it)[4]<confidence_threshold)
        {
            //  删除
            it=boxes_conf_landms.erase(it);
            num++;

        }else{
            ++it;
        }
    }*/

    //  根据得分对框进行从大到小排序
    std::sort(boxes_conf_landms.begin(),boxes_conf_landms.end(),confidence_comp);
    //  每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。在遍历删除防止迭代器失效
    std::vector<std::vector<double> > res;
     for(;  boxes_conf_landms.size()>0 ;)
     {
         std::vector<std::vector<double> >::iterator it =boxes_conf_landms.begin();
         //把最好的框保存下来
         res.push_back(*it);
         //如果boxes_conf_landms==1 循环结束
         //if(boxes_conf_landms.size()==1){
         //    break;
        // }
         //第一轮是开头，要跳过
        // if(it==boxes_conf_landms.begin()){
         //    it++;
        //     continue;
        // }
         //如果重合度大，删除 ，否则继续迭代
         //最好的最后一个框，和其他对比
         //res中最后一个和boxes中所有对比
         while(it!=boxes_conf_landms.end()){
             if(iou(res[res.size()-1],*it)>overlap_threshold)
             {
                 //  重合度比较大就删除
                 it=boxes_conf_landms.erase(it);
             }else{
                 ++it;
             }
         }

     }
    //  返回结果
    return res;
}

cv::Mat ncnn2cv(ncnn::Mat in)
{
    cv::Mat a(in.h, in.w, CV_32FC1);
    memcpy((uchar*)a.data, in.data, in.w * in.h * sizeof(float));
    return a.clone();
}

void retinaface_correct_boxes(std::vector<std::vector<double> > &result, double origin_high,double origin_width,
                              double width=1280,double high=1280){
    //  比例
    double new_high=origin_high*(std::min(width/origin_width,high/origin_high));
    double new_width=origin_width*(std::min(width/origin_width,high/origin_high));

    for(std::vector<std::vector<double> >::iterator it =result.begin();   it!=result.end();++it){
        //前4个，人脸框  ,1,0,1,0  ，宽，高，宽，高
        (*it)[0]=(*it)[0]*width/new_width*origin_width;
        (*it)[1]=(*it)[1]*high/new_high*origin_high;
        (*it)[2]=(*it)[2]*width/new_width*origin_width;
        (*it)[3]=(*it)[3]*high/new_high*origin_high;
        //后10个，人脸关键点

        for(int i=0;i<10;i++)
        {
            //奇数乘1,偶数乘0
            if((i+5)%2==1){
                (*it)[i+5]=(*it)[i+5]*width/new_width*origin_width;
            }else{
                (*it)[i+5]=(*it)[i+5]*high/new_high*origin_high;
            }
        }
    }
     /*
    double offset_0=(high-new_high)/2/high;
   double offset_1=(width-new_width)/2/width;

    double scale_0=high/new_high;
    double scale_1=width/new_width;


    for(std::vector<std::vector<double> >::iterator it =result.begin();   it!=result.end();++it){
        //前4个，人脸框  ,1,0,1,0
        (*it)[0]=(*it)[0]+offset_1*scale_1;
        (*it)[1]=(*it)[1]+offset_0*scale_0;
        (*it)[2]=(*it)[2]+offset_1*scale_1;
        (*it)[3]=(*it)[3]+offset_0*scale_0;
        //后10个，人脸关键点

        for(int i=0;i<10;i++)
        {
            //奇数乘1,偶数乘0
            if((i+5)%2==1){
                (*it)[i+5]=(*it)[i+5]+offset_1*scale_1;
            }else{
                (*it)[i+5]=(*it)[i+5]+offset_0*scale_0;
            }
        }
    }
    */

    return ;


}
void FaceDetection::analyze()
{
    //清理已有数据
    result.clear();

    //执行分析
    //----------------------------------------------
    //  获得原始照片的维度
    //----------------------------------------------
    double src_width = m_imageDet.rows;  //宽度
    double src_height = m_imageDet.cols;   //高度


    //----------------------------------------------
    //  对图片resize
    //----------------------------------------------
    cv::Mat img=resize_image(m_imageDet);  //输出RGB ,type16
    //  保存原图片，用于画图
    // cv::Mat old_img=qim2mat(imageDet);

    //----------------------------------------------
    //  把图片从cv::Mat格式，转成推理框架ncnn的，ncnn::Mat格式
    //  ncnn::Mat in_pad = ncnn::Mat::from_pixels(imageDet.bits(), ncnn::Mat::PIXEL_RGB, imageDet.width(), imageDet.height());
    //----------------------------------------------
    //cv::imwrite("/home/hello/Desktop/face2/face2/save_img/img1.png",img);
    ncnn::Mat in_pad = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_RGB, img.cols, img.rows);

    //pretty_print(in_pad);
    //----------------------------------------------
    //归一化处理
    //----------------------------------------------
    //  BGR
    //const float meanValues[3] = { 104, 117, 123 };
    const float meanValues[3] = { 104, 123, 117 };
    //const float meanValues[3] = { 123, 104, 117 };
    //const float meanValues[3] = { 123, 117, 104 };
    //const float meanValues[3] = { 117, 123, 104 };
    //const float meanValues[3] = { 117, 123, 104 };

    in_pad.substract_mean_normalize(meanValues, nullptr);
    //pretty_print(in_pad);
    //pretty_print(in_pad);
    //----------------------------------------------
    //  执行推理
    //----------------------------------------------
    auto extractor = detNet->create_extractor();
    extractor.input("in0", in_pad);

    ncnn::Mat out[3];
    extractor.extract("out0", out[0]);    //预测框
    extractor.extract("out1", out[1]);    //置信度
    extractor.extract("out2", out[2]);   //五点坐标

    //查看ncnn：：Mat维度


    //pretty_print(out[0]);
    //pretty_print(out[1]);

    //pretty_print(out[2]);
    //ncnn::mat 转化为  cv::mat
    cv::Mat out_box,out_confidence,out_landms;
    out_box=ncnn2cv(out[0]);
    out_confidence=ncnn2cv(out[1]); //置信度
    out_landms=ncnn2cv(out[2]);
    //double c=out_confidence.at<float>(0,0);

    //----------------------------------------------
    //生成先验框
    //----------------------------------------------
    std::vector<std::vector<double> >  anchors=get_anchors();
    //----------------------------------------------
    //预测框解码 decode---------------
    //----------------------------------------------
    std::vector<std::vector<double> > boxes=decode(out_box,anchors);
    //----------------------------------------------
    //人脸置信度
    //----------------------------------------------
    std::vector<double> confidence=get_confidence(out_confidence);
    //----------------------------------------------
    //人脸关键点解码
    //----------------------------------------------
    std::vector<std::vector<double> > landms=decode_landm(out_landms,anchors);
    //----------------------------------------------
    //得到非极大抑制结果，有可能结果为空
    //----------------------------------------------
    std::vector<std::vector<double> > rec_result=non_max_suppression(boxes,confidence,landms);

    //----------------------------------------------
    //  将输出调整为相对于原图的大小
    //----------------------------------------------
    retinaface_correct_boxes(rec_result,src_height,src_width);
    //----------------------------------------------
    //  要把坐标都还原为resize之前的原图坐标，confidence不用变换，主要还原预测框坐标和人脸关键点坐标
    //----------------------------------------------
    //----------------------------------------------
    //  计算scale，用于将获得的预测框转换成原图的宽高,和上面一样的
    //----------------------------------------------
    // double scale_width=imageDet.width();  //宽度
    // double scale_height=imageDet.height();   //高度
    /*
    for(std::vector<std::vector<double> >::iterator it =rec_result.begin();   it!=rec_result.end();++it){
        //  高

        (*it)[0]*=scale_width;
        (*it)[2]*=scale_width;

        (*it)[5]*=scale_width;
        (*it)[7]*=scale_width;
        (*it)[9]*=scale_width;
        (*it)[11]*=scale_width;
        (*it)[13]*=scale_width;

        //  高
        (*it)[1]*=scale_height;
        (*it)[3]*=scale_height;

        (*it)[6]*=scale_height;
        (*it)[8]*=scale_height;
        (*it)[10]*=scale_height;
        (*it)[12]*=scale_height;
        (*it)[14]*=scale_height;

        for(int i=0;i<15;i++){
            if(i!=4){
                (*it)[i]*=1280;
            }
        }
    }*/

    //printf("%f",rec_result[0][0]);

    //测试结果
    //  先打印结果看看,划线，画点，保存到图片中查看

    // cv::Scalar color;
    // color[0]=0;color[1]=0;color[2]=255;//    红色
    // cv::rectangle(old_img,cv::Point(rec_result[0][0],rec_result[0][1]),cv::Point(rec_result[0][2],rec_result[0][3]),color);
    // cv::circle(old_img,cv::Point(rec_result[0][5],rec_result[0][6]),1,color);
    // cv::circle(old_img,cv::Point(rec_result[0][7],rec_result[0][8]),1,color);
    // cv::circle(old_img,cv::Point(rec_result[0][9],rec_result[0][10]),1,color);
    // cv::circle(old_img,cv::Point(rec_result[0][11],rec_result[0][12]),1,color);
    // cv::circle(old_img,cv::Point(rec_result[0][13],rec_result[0][14]),1,color);


    // cv::imwrite("/home/hello/Desktop/face2/face2/save_img/end.png",old_img);

    //cv::imwrite("/home/hello/Desktop/face2/face2/save_img/end_1.png",old_img);


    cv::Rect rect(rec_result[0][0],rec_result[0][1],rec_result[0][2],rec_result[0][3]);
    std::vector<cv::Point> keyPoints;
    for(int i=0;i<10;i+=2){
        cv::Point point(rec_result[0][i+5],rec_result[0][i+6]);
        keyPoints.push_back(point);
    }
    double prob=rec_result[0][4];

    FaceBox facebox;

    facebox.rect=rect;
    facebox.keyPoints=keyPoints;
    facebox.prob=prob;
    result.push_back(facebox);


    /*
    printf("box:%d---confidence:%d---landms:%d",boxes.size(),confidence.size(),landms.size());
    //把灰条去掉-----------------

    //重新scale

    //返回结果到result
    printf("-----------------------------------------------\n");
    //pretty_print(out[0]);                // 五点坐标
    printf("-----------------------------------------------\n");
    //pretty_print(out[1]);              //
    printf("-----------------------------------------------\n");
   // pretty_print(out[2]);
    printf("-----------------------------------------------\n");

    printf("hello");
    */
}
