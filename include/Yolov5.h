#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

/**
 * @brief Struct to hold a single detection result
 */
struct Detection
{
    int classId;       // ID of the detected class
    float confidence;  // Confidence score
    cv::Rect box;      // Bounding box of the detected object
};

/**
 * @brief YOLOv5 object detection wrapper
 */
class Yolov5
{
public:
    /**
     * @brief Construct a new Yolov5 object
     * @param pathModel Path to the ONNX model
     * @param imgSz Input image size for the model (e.g., 640)
     */
    explicit Yolov5(const std::string& pathModel, int imgSz = 640)
        : pathModel(pathModel), imgSize(imgSz)
    {
        buildModel();
    }

    /**
     * @brief Load and initialize the YOLOv5 model
     */
    void buildModel();

    /**
     * @brief Detect objects in the given image and return the cropped frame of the first detection
     * @param img Input image
     * @param box Bounding box of detected object
     * @return Cropped image corresponding to the detected object
     */
    cv::Mat getFrame(const cv::Mat& img, cv::Rect& box) const;

    /**
     * @brief Run detection on an image and populate results vector
     * @param img Input image
     * @param results Vector to store detection results
     */
    void detect(const cv::Mat& img, std::vector<Detection>& results) const;

    /**
     * @brief Format image into square dimensions for YOLOv5 input
     * @param img Input image
     * @return Formatted square image
     */
    cv::Mat formatYolov5(const cv::Mat& img ) const;

    /**
     * @brief Load class labels from a text file
     * @return Vector of class names
     */
    std::vector<std::string> loadLabels();

private:
    std::string pathModel = "best.onnx";          // Path to ONNX model
    std::string device = "CPU";                   // Device to run the model on ("CPU" or "GPU")
    int imgSize = 640;                            // Input image size for the model
    cv::Mat predictMat;                           // Temporary matrix for predictions
    mutable cv::dnn::Net modelYolov5;                     // OpenCV DNN network

    const float confidenceThreshold = 0.8f;      // Minimum confidence for detections
    const float scoreThreshold = 0.8f;           // Minimum class score for detection
    const float nmsThreshold = 0.6f;             // Non-maximum suppression threshold

    const std::string pathLabelTxt = "model/classes.txt"; // Path to class labels
    std::vector<std::string> classNames;                  // Loaded class names
};
