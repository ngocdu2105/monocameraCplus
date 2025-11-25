#include "include/Calibration.h"
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <iostream>
#include "include/LoadIMG.h"
#include "include/Utils.h"

Calibration::Calibration(const std::string& calibrationImagePath) 
    : calibrationImagePath_(calibrationImagePath)
{
    Timer timer;

    cv::Mat grayImage = cv::imread(calibrationImagePath_, cv::IMREAD_GRAYSCALE);
    bool cornersFound = cv::findChessboardCorners(
        grayImage, 
        chessboardSize_, 
        chessboardCorners_, 
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS
    );

    if (cornersFound)
    {
        cv::cornerSubPix(
            grayImage, 
            chessboardCorners_, 
            cv::Size(13, 9), 
            cv::Size(-1, -1), 
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.1)
        );
        cv::drawChessboardCorners(grayImage, chessboardSize_, chessboardCorners_, cornersFound);
    }

    // Compute average distance between adjacent corners
    double referenceDistance = Utils::pointLineDistance(
        chessboardCorners_[13], chessboardCorners_[0], chessboardCorners_[1]
    );

    std::vector<float> validDistances;
    for (size_t i = 1; i < chessboardCorners_.size(); ++i)
    {
        double distance = cv::norm(chessboardCorners_[i] - chessboardCorners_[i-1]);
        if (distance < referenceDistance + 10)
            validDistances.push_back(static_cast<float>(distance));
    }
    avgDistanceBetweenCorners_ = std::accumulate(validDistances.begin(), validDistances.end(), 0.0) 
                                / validDistances.size();

    // Reference points for calibration axes
    origin_ = chessboardCorners_[0];
    originX_ = chessboardCorners_[1];
    originY_ = chessboardCorners_[13];

    std::cerr << "Calibration image loaded successfully.\n";
    timer.stop();
}

const double& Calibration::getAverageCornerDistance() const
{
    return avgDistanceBetweenCorners_;
}

void Calibration::convertPointToChessboardUnits(const cv::Point2f& point, cv::Mat& image) const
{
    normalizedX_ = Utils::pointLineDistance(point, origin_, originX_) / avgDistanceBetweenCorners_;
    normalizedY_ = Utils::pointLineDistance(point, origin_, originY_) / avgDistanceBetweenCorners_;

    // Draw points on image for visualization
    cv::circle(image, point, 2, cv::Scalar(0, 0, 255), 2);
    cv::circle(image, origin_, 2, cv::Scalar(0, 0, 255), 2);
    cv::circle(image, originX_, 2, cv::Scalar(0, 0, 255), 2);
    cv::circle(image, originY_, 2, cv::Scalar(0, 0, 255), 2);

    displayCalibrationResult(image);

    std::cerr << "2D calibration point converted: P(" 
              << normalizedX_ << ", " << normalizedY_ << ").\n";
}

void Calibration::displayCalibrationResult(cv::Mat& image) const
{
    std::string text = "P(" + std::to_string(normalizedX_) + ", " + std::to_string(normalizedY_) + ")";
    cv::putText(
        image, 
        text, 
        cv::Point(50, 50), 
        cv::FONT_HERSHEY_SIMPLEX, 
        1.0, 
        cv::Scalar(0, 0, 255), 
        2
    );
}
