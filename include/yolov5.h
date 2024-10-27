#pragma once
#include <string.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdlib> 
#include <iostream>
#include <fstream>
#include <memory.h>


struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class Yolov5
{
public:
    explicit Yolov5(const std::string& PathModel, const int img_sz):Path(PathModel),img_size(img_sz)
    {
        build_model();
    }

    void build_model(); 
    cv::Mat get_frame(const cv::Mat& img, cv::Rect& results);
    void loop_frame();
    void detect (const cv::Mat& img, std::vector<Detection>& results);
    void wrap_detection(const cv::Mat& input_img, const cv::Mat& output_img);
    void mask_object();
    cv::Mat format_yolov5(const cv::Mat& img);
    std::vector<std::string> load_label();
    // ~Yolov5();
private:
    std::string Path="best.onnx";
    std::string divice = "CPU";
    int img_size = 640;
    cv::Mat _predict;
    cv::dnn::Net ModelYolov5;
    const float CONFIDENCE_THRESHOLD = 0.8;
    const float SCORE_THRESHOLD = 0.8;
    const std::string path_label_txt = "model/classes.txt";
    std::vector<std::string> className;
    const float NMS_THRESHOLD = 0.6;
};
