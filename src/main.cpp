#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/Yolov5.h"
#include "include/RCNN2.h"
#include "include/Calibration.h"
#include "include/LoadIMG.h"
#include "include/PointSet.h"
#include "include/Utils.h"
// ============================
// Function to process one image
// ============================
void ProcessImage(const cv::Mat& img,
                  const Yolov5& yolo,
                  const RCNN2& rcnn,
                  const Calibration& calibration)
{
    // 1. Detect object with YOLOv5
    cv::Rect detectedBox;
    cv::Mat cropObject = yolo.getFrame(img, detectedBox);
    cv::Size cropSize = cropObject.size();
    std::cout << "size cropSize: " << cropSize.width << " x " << cropSize.height << std::endl; 
    // 2. Predict masks with RCNN
    rcnn.run(cropObject);
    cv::Mat maskTop   = rcnn.getMaskTop();
    cv::Mat maskBelow = rcnn.getMaskBelow();


    // 3. Compute displacement with PointSet
    geom::PointSet displacementPoints;
    cv::Point topCenter, topCenterMoved;

    geom::computeTopCenterDisplacement(
        maskTop,
        maskBelow,
        cropSize.width,
        cropSize.height,
        topCenter,
        topCenterMoved,
        displacementPoints
    );

    // // 4. Convert point to original image coordinates
    cv::Point pointInOriginal = geom::convertSquareToOriginalImage(img, detectedBox, topCenterMoved);

    // // 5. Visualization / calibration
    cv::Mat imgClone = img.clone();
    calibration.convertPointToChessboardUnits(pointInOriginal, imgClone);
    cv::circle(cropObject, topCenter, 3, cv::Scalar(0, 0, 255), -1);    
    cv::circle(imgClone, pointInOriginal, 3, cv::Scalar(0, 0, 255), -1);

    cv::Point p1 = displacementPoints[0];
    cv::Point p2 = displacementPoints[1];
    cv::circle(cropObject, p1, 3, cv::Scalar(0, 255, 255), -1);
    cv::circle(cropObject, p2, 3, cv::Scalar(0, 255, 255), -1);
    cv::imshow("img", imgClone);
    cv::imshow("imgClone", maskBelow);
    cv::imshow("cropObject", cropObject);
    cv::waitKey(0);

    std::cout << "Pixel coordinates of point P in original image: " << pointInOriginal << std::endl;
    // cv::imshow("maskTop", maskTop );
    // cv::imshow("maskBelow", maskBelow);
}

// ============================
// Main function
// ============================
int main()
{
    // Load paths and models
    std::string pathImages      = "./dataset";
    std::string pathYOLOv5Model = "./model/model_yolov5.onnx";
    std::string pathRCNNModel   = "./model/mask_rcnn_dataset_fix.onnx";
    std::string pathCalibration = "./CalibrationIMG/calibration.jpg";

    LoadIMG loader(pathImages);
    Calibration calibration(pathCalibration);
    Yolov5 yolo(pathYOLOv5Model, 640);
    RCNN2 rcnn(pathRCNNModel);

    auto& images = loader.getImgs();

    // Process each image
    for (auto& img : images)
    {
        try
        {
            ProcessImage(img, yolo, rcnn, calibration);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error processing image: " << e.what() << std::endl;
            continue;
        }
        catch (...)
        {
            std::cerr << "Unknown exception occurred while processing image." << std::endl;
        }
    }

    std::cerr << "Processing completed for all images!" << std::endl;
    return 0;
}
