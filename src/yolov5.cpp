#include <string.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdlib> 
#include <iostream>
#include <fstream>
#include <algorithm>
#include "include/yolov5.h"
#include "include/LoadIMG.h"

void Yolov5::build_model()
{
    Timer t;
   
    ModelYolov5 = cv::dnn::readNetFromONNX(Path);

    if(divice == "GPU")
    {
        ModelYolov5.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        ModelYolov5.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);        
    }
    else
    {
        ModelYolov5.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        ModelYolov5.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    
    className = load_label();
    size_t t1 = cv::getTickCount();
    std::cerr<< "Load success model Yolov5 run in "<<divice<<" ";
    t.stop();

}

std::vector<std::string> Yolov5::load_label()
{
    std::vector<std::string> class_list;
    std::ifstream ifs(path_label_txt);
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}


void Yolov5::detect(const cv::Mat& img, std::vector<Detection>& results )
{
    cv::Mat img_pre;

    img_pre=cv::dnn::blobFromImage(img,1.0f/255.0,cv::Size(this->img_size,this->img_size),cv::Scalar(),true, false);
    ModelYolov5.setInput(img_pre);
    
    std::vector<cv::Mat> outputs;
 
     
    ModelYolov5.forward(outputs,"output0");




    float x_factor = img.cols / this->img_size;
    float y_factor = img.rows / this->img_size;
    
    float *data = (float *)outputs[0].data;

    const int dimensions = 6;
    const int rows = 25200;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;


    for (int i = 0; i < rows; ++i) {

        // auto data = outputs[0].ptr<float>(i); 
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float * classes_scores = data+5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += dimensions;

    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        results.push_back(result);
    }

}

cv::Mat Yolov5::get_frame(const cv::Mat& img, cv::Rect& box)
{
    Timer t;
    cv::Mat img_square = Yolov5::format_yolov5(img);
    // cv::imshow("img", img_square);
    // cv::waitKey(0);
    std::vector<Detection> results;
    // Yolov5::detect detect;
    detect(img_square,results);
    if(results.empty())
    {
        std::cout<<"Empty image";
    }
    cv::Rect box1 = results[0].box;

    box1.x -= 20;
    box1.y -= 20;
    box1.width += 40;
    box1.height += 40;
    box = box1;

    std::cerr<<"Yolov5 model predicts image ";
    t.stop();
    return img_square(box);    

}


cv::Mat Yolov5::format_yolov5(const cv::Mat& img)
{
    int width = img.cols, height = img.rows;
    int max_ = (width >= height) ? width :height;
   
    cv::Mat square = cv::Mat::zeros(max_,max_,img.type());
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = max_;
        roi.x = 0;
        roi.height = height;
        roi.y = ( max_ - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = max_;
        roi.width = width;
        roi.x = ( max_ - roi.width ) / 2;
    }
    cv::resize(img, square(roi),roi.size());
    return square;

}

