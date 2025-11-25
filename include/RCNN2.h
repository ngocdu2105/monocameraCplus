#ifndef RCNN2_H
#define RCNN2_H

#include<opencv2/opencv.hpp>
#include<vector>
#include<string>
#include <opencv2/core.hpp>
#include<memory.h>
#include<onnxruntime_cxx_api.h>

class RCNN2
{
public:
    explicit RCNN2(const std::string& pathModel);
    cv::Mat getMaskTop() const;
    cv::Mat getMaskBelow() const;
    void run(const cv::Mat& img) const;
private:
    std::vector<float> PreprocessImg(const cv::Mat& img) const;
    void process();

    mutable Ort::Session session{nullptr};
    mutable Ort::Env env;
    mutable cv::Mat MatTop;
    mutable cv::Mat MatBelow;

    const std::vector<const char *> name_input {"im_shape","image","scale_factor"};
    const std::vector<const char *> name_output {"tmp_42","concat_7.tmp_0","cast_18.tmp_0"};
    cv::Size target_sz{224,224};
    // std::unique_ptr<Ort::Session> session;
    

};

#endif