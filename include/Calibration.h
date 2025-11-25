#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "include/Utils.h"   // Utility functions like distancePointToLinePoint

class Calibration
{
private:
    std::vector<cv::Point2f> chessboardCorners_;
    std::string calibrationImagePath_;
    cv::Size chessboardSize_ = cv::Size(13, 9);

    // Reference points for calibration
    mutable cv::Point2f origin_;
    mutable cv::Point2f originX_;
    mutable cv::Point2f originY_;

    // Distances in chessboard unit
    mutable double avgDistanceBetweenCorners_ = 0.0;
    mutable double normalizedX_ = 0.0;
    mutable double normalizedY_ = 0.0;

public:
    explicit Calibration(const std::string& calibrationImagePath);

    // Get the average distance between adjacent chessboard corners
    const double& getAverageCornerDistance() const;

    // Convert a 2D point to normalized chessboard coordinates
    void convertPointToChessboardUnits(const cv::Point2f& point, cv::Mat& image) const;

    // Draw result text on image
    void displayCalibrationResult(cv::Mat& image) const;
};
