#include <string.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cstdlib> 
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <assert.h>
#include<include/RCNN2.h>
#include "onnxruntime_cxx_api.h"
#include <include/LoadIMG.h>

RCNN2::RCNN2(const std::string& pathModel)
{
    Timer t;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXExample");
    Ort::SessionOptions sOpts;
    session= Ort::Session(env, pathModel.c_str(), Ort::SessionOptions());
    std::cerr<<"Load success model RCNN2 ";
    t.stop();
}

cv::Mat RCNN2::getMaskBelow() const
{
    return MatBelow;
}
cv::Mat RCNN2::getMaskTop() const
{
    return MatTop;
}
std::vector<float> RCNN2::PreprocessImg(const cv::Mat& img) const
{
    cv::Mat img_rz,resizedImage, img_rgb,img_merge, preprocessedImage;
    cv::resize(img, img_rz, target_sz,0,0,cv::InterpolationFlags::INTER_LINEAR);
    cv::cvtColor(img_rz,img_rgb,cv::ColorConversionCodes::COLOR_BGR2RGB);
    img_rgb.convertTo(resizedImage, CV_32F, 1.0 / 255.0);
    cv::Mat channel[3];
    cv::split(resizedImage,channel);
    channel[0] = (channel[0]- 0.485)/ 0.229;
    channel[1] = (channel[1] - 0.456)/0.224;
    channel[2] = (channel[2]-0.406) / 0.225;
    cv::merge(channel, 3, img_merge);
    cv::dnn::blobFromImage(img_merge, preprocessedImage);

    // cv::Mat preprocessing = cv::dnn::blobFromImage(img_merge,1,cv::Size(224, 224), cv::Scalar(1, 1, 1), false, false);
    // return matToVector(resizedImage);
    // return matToVectorCols(channel);

    return matToVector(preprocessedImage);
    // return matToVector(preprocessing);

}
void RCNN2::run(const cv::Mat& img) const
{
    Timer t;
    assert(!img.empty() && "RCNN2 Input image is empty");
    std::vector<Ort::Value> ort_inputs;
    auto info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU); // Changed allocator type
    cv::Size input_f = img.size();

    std::vector<int64_t> input_imshapeDataShape = { 1, 2};
    std::vector<float> input_image = PreprocessImg(img);
    std::vector<int64_t> input_imageshape = {1, 3, 224, 224};
    std::vector<float> input_scale_factor = generate_scale(img);
    std::vector<float> input_imshapeData{std::ceil(input_f.height*input_scale_factor[0]),std::ceil(input_f.width*input_scale_factor[1])};
    std::vector<int64_t> input_scale_factorShape = { 1, 2};

    print_Vector(input_imshapeData);
    print_Vector(input_scale_factor);

    // Create tensors
    try {
        ort_inputs.push_back(
            Ort::Value::CreateTensor<float>(info, input_imshapeData.data(), 
                                            input_imshapeData.size(), 
                                            input_imshapeDataShape.data(), 
                                            input_imshapeDataShape.size()));

        ort_inputs.push_back(
            Ort::Value::CreateTensor<float>(info, input_image.data(), 
                                            input_image.size(), 
                                            input_imageshape.data(), 
                                            input_imageshape.size()));

        ort_inputs.push_back(
            Ort::Value::CreateTensor<float>(info, input_scale_factor.data(), 
                                            input_scale_factor.size(), 
                                            input_scale_factorShape.data(), 
                                            input_scale_factorShape.size()));



    // Run the model
    assert(ort_inputs.back().GetTensorTypeAndShapeInfo().GetElementCount() > 0 && "input size ort not true");

    std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{}, name_input.data(),
                                    ort_inputs.data(), ort_inputs.size(),
                                    name_output.data(), name_output.size());

    GetOutputOnnx(ort_outputs, MatTop,MatBelow,input_f.height,input_f.width,0.95);
    std::cerr<<"RCNN2 model predicts crop image ";

    } catch (const std::exception& e) {
        std::cerr << "Error creating tensors: " << e.what() << std::endl;
    }
    ort_inputs.clear();
    t.stop();    
}