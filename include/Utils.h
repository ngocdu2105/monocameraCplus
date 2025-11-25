#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "include/PointSet.h"
#include "include/PointSetArray.h"
#include "include/NeighborPointSet.h"
namespace Utils {

// Conversions
std::vector<float> matToFloatVector(const cv::Mat& img);
std::vector<float> matToFloatVectorChannels(const cv::Mat* channels);
std::vector<std::vector<float>> reshape1Dto2D(const std::vector<float>& vec, int rows);
std::vector<cv::Mat> floatVectorToMatChannels(const std::vector<float>& vec, int w, int h, int channels);
void tensorTo2DVector(const float* ptr, size_t size, std::vector<std::vector<float>>& out, int rows);

// Geometry
std::vector<std::vector<double>> computeDistanceP1toP2(const std::vector<cv::Point>& p1, const std::vector<cv::Point>& p2);
double sumMinPerRow(const std::vector<std::vector<double>>& mat);
std::vector<double> minPerRow(const std::vector<std::vector<double>>& mat);
double pointLineDistance(const cv::Point& P, const cv::Point& A, const cv::Point& B);
std::vector<cv::Point> offsetPoints(const std::vector<cv::Point>& pts, const cv::Point& offset);
cv::Point findExtremeDistancePoint(const std::vector<std::vector<double>>& mat);
cv::Point findMinMaxDisP1P2(const std::vector<std::vector<double>> &disP1P2);

geom::NeighborPointSet findNeighborPointSetsWithThreshold(
    const geom::PointSet& setA,
    geom::PointSet& setB,
    double threshold);


// ONNX
void extractOnnxOutputMasks(std::vector<Ort::Value>& results, cv::Mat& topMask, cv::Mat& bottomMask, int w, int h, float threshold);

// Mask
void mergeMasksByIndex(const std::vector<cv::Mat>& masks, const std::vector<size_t>& indices, cv::Mat& outMask);

// Random
std::vector<int> generateRandomVector(int count, int minVal, int maxVal);

// Resize scale
std::vector<float> computeResizeScale(const cv::Mat& img, const std::vector<int>& targetSize = {224,224});

// Printing
// template<typename T> void printVector(const std::vector<T>& v);
// template<typename T> void printVector2D(const std::vector<std::vector<T>>& v);
// template<typename T> std::vector<size_t> sortIndex(const std::vector<T>& v);
template<typename T>
inline void printVector(const std::vector<T>& v) {
    std::cout << "[";
    for (const auto& x : v)
        std::cout << x << ",";
    std::cout << "]\n";
}

template<typename T>
inline void printVector2D(const std::vector<std::vector<T>>& v) {
    std::cout << "[";
    for (const auto& row : v) {
        std::cout << "[";
        for (const auto& x : row)
            std::cout << x << ",";
        std::cout << "],";
    }
    std::cout << "]\n";
}

template<typename T>
inline std::vector<size_t> sortIndex(const std::vector<T>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](size_t i, size_t j){ return v[i] < v[j]; });
    return idx;
}


} 

namespace geom {

/// @brief Compute the center point of the top mask and its displacement relative to the bottom mask
/// @param topMask Top mask image
/// @param bottomMask Bottom mask image
/// @param width Image width
/// @param height Image height
/// @param topCenter Output: center of the top mask
/// @param displacedTopCenter Output: displaced top center after edge alignment
/// @param displacementPoints Output: points representing the displacement vector
void computeTopCenterDisplacement(const cv::Mat& topMask, 
                                  const cv::Mat& bottomMask, 
                                  int width, int height,
                                  cv::Point& topCenter,
                                  cv::Point& displacedTopCenter,
                                  geom::PointSet& displacementPoints);

/// @brief Convert a point from a square/cropped image back to the original image coordinates
/// @param originalImage Original full image
/// @param boundingBox Bounding rectangle of the square/cropped image in original coordinates
/// @param squarePoint Point in the square/cropped image
/// @return Corresponding point in the original image
cv::Point convertSquareToOriginalImage(const cv::Mat& originalImage, 
                                       const cv::Rect& boundingBox, 
                                       const cv::Point& squarePoint);

} // namespace geom