#ifndef RCNN_H
#define RCNN_H

#include<opencv2/opencv.hpp>
#include<vector>
#include<string>
#include <opencv2/core.hpp>
#include<memory.h>
#include<onnxruntime_cxx_api.h>

class RCNN
{
public:
    explicit RCNN(const std::string& pathModel);
    cv::Mat masktop() const;
    cv::Mat maskbelow() const;
    void run(const cv::Mat& img);
private:
    std::vector<float> PreprocessImg(const cv::Mat& img);
    void process();

    Ort::Session session{nullptr};

    cv::Mat MatTop,MatBelow;

    const std::vector<const char *> name_input {"im_shape","image","scale_factor"};
    const std::vector<const char *> name_output {"tmp_42","concat_7.tmp_0","cast_18.tmp_0"};
    cv::Size target_sz{224,224};
    // std::unique_ptr<Ort::Session> session;
    

};

#endif