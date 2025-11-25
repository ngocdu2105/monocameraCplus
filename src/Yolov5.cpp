#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "include/Yolov5.h"
#include "include/LoadIMG.h"

// Build and load the YOLOv5 model
void Yolov5::buildModel()
{
    Timer timer;

    modelYolov5 = cv::dnn::readNetFromONNX(pathModel);

    if (device == "GPU") {
        modelYolov5.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        modelYolov5.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        modelYolov5.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        modelYolov5.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    classNames = loadLabels();
    std::cerr << "[INFO] YOLOv5 model loaded successfully on " << device << std::endl;
    timer.stop();
}

// Load class labels from a text file
std::vector<std::string> Yolov5::loadLabels()
{
    std::vector<std::string> labels;
    std::ifstream ifs(pathLabelTxt);
    std::string line;
    while (getline(ifs, line)) {
        labels.push_back(line);
    }
    return labels;
}

// Run detection on an input image
void Yolov5::detect(const cv::Mat& image, std::vector<Detection>& detections) const
{
    cv::Mat inputBlob = cv::dnn::blobFromImage(
        image, 1.0f / 255.0f, cv::Size(imgSize, imgSize), cv::Scalar(), true, false
    );
    modelYolov5.setInput(inputBlob);

    std::vector<cv::Mat> outputs;
    modelYolov5.forward(outputs, "output0");

    float xFactor = static_cast<float>(image.cols) / imgSize;
    float yFactor = static_cast<float>(image.rows) / imgSize;

    float* data = reinterpret_cast<float*>(outputs[0].data);
    const int dimensions = 6;
    const int numRows = 25200;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < numRows; ++i) {
        float confidence = data[4];
        if (confidence >= confidenceThreshold ) {
            float* classScores = data + 5;
            cv::Mat scores(1, classNames.size(), CV_32FC1, classScores);
            cv::Point classId;
            double maxScore;
            cv::minMaxLoc(scores, nullptr, &maxScore, nullptr, &classId);

            if (maxScore > scoreThreshold) {
                confidences.push_back(confidence);
                classIds.push_back(classId.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((x - 0.5f * w) * xFactor);
                int top = static_cast<int>((y - 0.5f * h) * yFactor);
                int width = static_cast<int>(w * xFactor);
                int height = static_cast<int>(h * yFactor);

                boxes.emplace_back(left, top, width, height);
            }
        }
        data += dimensions;
    }

    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, nmsIndices);

    for (int idx : nmsIndices) {
        Detection detection;
        detection.classId = classIds[idx];
        detection.confidence = confidences[idx];
        detection.box = boxes[idx];
        detections.push_back(detection);
    }
}

// Crop and return the detected object frame
cv::Mat Yolov5::getFrame(const cv::Mat& image, cv::Rect& detectedBox) const
{
    Timer timer;
    cv::Mat squareImage = formatYolov5(image);
    std::vector<Detection> detections;
    detect(squareImage, detections);

    if (detections.empty()) {
        std::cerr << "[WARN] No objects detected." << std::endl;
        return cv::Mat();
    }

    cv::Rect box = detections[0].box;
    box.x = std::max(0, box.x - 20);
    box.y = std::max(0, box.y - 20);
    box.width += 40;
    box.height += 40;

    detectedBox = box;
    timer.stop();
    std::cerr << "[INFO] YOLOv5 model predicted object frame." << std::endl;

    return squareImage(box);
}

// Format image into square for YOLOv5 input
cv::Mat Yolov5::formatYolov5(const cv::Mat& image) const
{
    int width = image.cols;
    int height = image.rows;
    int maxDim = std::max(width, height);

    cv::Mat square = cv::Mat::zeros(maxDim, maxDim, image.type());
    cv::Rect roi;

    if (width >= height) {
        roi = cv::Rect(0, (maxDim - height) / 2, width, height);
    } else {
        roi = cv::Rect((maxDim - width) / 2, 0, width, height);
    }

    cv::resize(image, square(roi), roi.size());
    return square;
}
