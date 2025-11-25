#include "include/Utils.h"
#include "include/NeighborPointSet.h"
#include "include/PointSet.h"
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>

namespace Utils {

// ---------------- Conversion ----------------
std::vector<float> matToFloatVector(const cv::Mat& img) {
    assert(!img.empty());
    std::vector<float> vec(img.total() * img.channels());
    std::memcpy(vec.data(), img.data, vec.size() * sizeof(float));
    return vec;
}

std::vector<float> matToFloatVectorChannels(const cv::Mat* channels) {
    size_t totalPixels = channels[0].total();
    std::vector<float> vec(3 * totalPixels);

    for (size_t i = 0; i < totalPixels; ++i) {
        vec[i * 3 + 0] = channels[0].at<float>(i);
        vec[i * 3 + 1] = channels[1].at<float>(i);
        vec[i * 3 + 2] = channels[2].at<float>(i);
    }
    return vec;
}

std::vector<std::vector<float>> reshape1Dto2D(const std::vector<float>& vec, int rows) {
    assert(!vec.empty());
    size_t cols = vec.size() / rows;
    std::vector<std::vector<float>> mat(cols, std::vector<float>(rows));
    for (size_t i = 0; i < cols; ++i) {
        std::copy(vec.begin() + i * rows, vec.begin() + (i + 1) * rows, mat[i].begin());
    }
    return mat;
}

std::vector<cv::Mat> floatVectorToMatChannels(const std::vector<float>& vec, int w, int h, int channels) {
    assert(vec.size() == w * h * channels);
    std::vector<std::vector<float>> data = reshape1Dto2D(vec, w * h);
    std::vector<cv::Mat> mats;
    mats.reserve(channels);

    for (const auto& ch : data) {
        mats.push_back(cv::Mat(w, h, CV_32F, const_cast<float*>(ch.data())).clone());
    }
    return mats;
}

void tensorTo2DVector(const float* ptr, size_t size, std::vector<std::vector<float>>& out, int rows) {
    assert(ptr != nullptr);
    std::vector<float> vec(ptr, ptr + size);
    out = reshape1Dto2D(vec, rows);
}

// ---------------- Geometry ----------------
std::vector<std::vector<double>> computeDistanceP1toP2(const std::vector<cv::Point>& p1,
                                                       const std::vector<cv::Point>& p2) {
    assert(!p1.empty() && !p2.empty());
    std::vector<std::vector<double>> dist;
    dist.reserve(p1.size());

    for (const auto& pt1 : p1) {
        std::vector<double> row;
        row.reserve(p2.size());
        for (const auto& pt2 : p2)
            row.push_back(cv::norm(pt1 - pt2));
        dist.push_back(row);
    }
    return dist;
}

double sumMinPerRow(const std::vector<std::vector<double>>& mat) {
    assert(!mat.empty());

    double sum = 0.0;
    for (const auto& row : mat) {
        sum += *std::min_element(row.begin(), row.end());
    }
    return sum;
}

std::vector<double> minPerRow(const std::vector<std::vector<double>>& mat) {
    assert(!mat.empty());

    std::vector<double> rowMins;
    rowMins.reserve(mat.size());

    for (const auto& row : mat) {
        if (row.empty()) {
            rowMins.push_back(std::numeric_limits<double>::infinity());
            continue;
        }

        rowMins.push_back(*std::min_element(row.begin(), row.end()));
    }

    return rowMins;
}


double pointLineDistance(const cv::Point& P, const cv::Point& A, const cv::Point& B) {
    double numerator = std::abs((B.y - A.y) * P.x - (B.x - A.x) * P.y + B.x * A.y - B.y * A.x);
    double denominator = std::sqrt(std::pow(B.y - A.y, 2) + std::pow(B.x - A.x, 2));
    return numerator / denominator;
}

std::vector<cv::Point> offsetPoints(const std::vector<cv::Point>& pts, const cv::Point& offset) {
    std::vector<cv::Point> result;
    result.reserve(pts.size());
    std::transform(pts.begin(), pts.end(), std::back_inserter(result),
                   [&offset](const cv::Point& p){ return p + offset; });
    return result;
}

cv::Point findMinMaxDisP1P2(const std::vector<std::vector<double>> &disP1P2)
{
    assert(!disP1P2.empty() && "Input matrix disP1P2 is empty!");

    std::vector<double> minPerRow;
    minPerRow.reserve(disP1P2.size());

    // Tính min cho từng hàng
    for (const auto &row : disP1P2)
    {
        double mn = *std::min_element(row.begin(), row.end());
        if (std::isnan(mn)) mn = 0.0;
        minPerRow.push_back(mn);
    }

    // Tìm hàng có min lớn nhất
    size_t idxP1 = std::distance(
        minPerRow.begin(), 
        std::max_element(minPerRow.begin(), minPerRow.end())
    );

    // Tìm vị trí min trong hàng đó
    const auto &row = disP1P2[idxP1];
    size_t idxP2 = std::distance(
        row.begin(),
        std::min_element(row.begin(), row.end())
    );

    return cv::Point(static_cast<int>(idxP1), static_cast<int>(idxP2));
}


cv::Point findExtremeDistancePoint(const std::vector<std::vector<double>>& mat) {
    assert(!mat.empty());
    std::vector<double> minVals;
    minVals.reserve(mat.size());
    for (const auto& row : mat)
        minVals.push_back(*std::min_element(row.begin(), row.end()));

    size_t idxRow = std::distance(minVals.begin(), std::max_element(minVals.begin(), minVals.end()));
    size_t idxCol = std::distance(mat[idxRow].begin(), std::min_element(mat[idxRow].begin(), mat[idxRow].end()));

    return cv::Point(idxRow, idxCol);
}

geom::NeighborPointSet findNeighborPointSetsWithThreshold(
    const geom::PointSet& setA,
    geom::PointSet& setB,
    double threshold)
{
    assert(setA.size() > 0 && setB.size() > 0 &&
           "Input PointSets must not be empty!");

    auto distAB = setA.computeDistanceMatrix(setB);

    std::vector<size_t> nearestBIndex;
    nearestBIndex.reserve(distAB.size());

    for (const auto& row : distAB) {
        auto itMin = std::min_element(row.begin(), row.end());
        double dmin = *itMin;

        if (dmin < threshold) {
            nearestBIndex.push_back(std::distance(row.begin(), itMin));
        } else {
            nearestBIndex.push_back(std::numeric_limits<size_t>::max());
        }
    }

    geom::PointSet A_primary;   // near
    geom::PointSet B_primary;
    geom::PointSet A_secondary; // far
    geom::PointSet B_secondary; // remaining B

    std::vector<bool> bUsed(setB.size(), false);

    for (size_t i = 0; i < nearestBIndex.size(); ++i) {
        size_t idxB = nearestBIndex[i];

        if (idxB != std::numeric_limits<size_t>::max()) {
            A_primary.add(setA[i]);
            B_primary.add(setB[idxB]);
            bUsed[idxB] = true;
        } else {
            A_secondary.add(setA[i]);
        }
    }

    for (size_t j = 0; j < setB.size(); ++j) {
        if (!bUsed[j]) {
            B_secondary.add(setB[j]);
        }
    }

    return geom::makeNeighborSet(
        A_primary.data(),
        B_primary.data(),
        A_secondary.data(),
        B_secondary.data()
    );
};




// ---------------- Masks ----------------
void mergeMasksByIndex(const std::vector<cv::Mat>& masks, const std::vector<size_t>& indices, cv::Mat& outMask) {
    assert(!masks.empty());
    outMask = cv::Mat::zeros(masks[0].size(), masks[0].type());
    for (size_t idx : indices)
        cv::add(outMask, masks[idx], outMask);
    outMask = outMask > 0;
}

// ---------------- Random ----------------
std::vector<int> generateRandomVector(int count, int minVal, int maxVal) {
    std::vector<int> vec(count);
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(minVal, maxVal);
    std::generate(vec.begin(), vec.end(), [&](){ return dist(gen); });
    return vec;
}

// ---------------- Resize ----------------
std::vector<float> computeResizeScale(const cv::Mat& img, const std::vector<int>& targetSize) {
    assert(!img.empty());
    int h = img.rows, w = img.cols;
    float scaleX = static_cast<float>(targetSize[1]) / w;
    float scaleY = static_cast<float>(targetSize[0]) / h;
    float scale = std::min(scaleX, scaleY);
    return {scale * h / h, scale * w / w}; // return scale factors
}



//onnx 

void extractOnnxOutputMasks(std::vector<Ort::Value>& results,
                                   cv::Mat& topMask,
                                   cv::Mat& bottomMask,
                                   int width,
                                   int height,
                                   float threshold)
{
    assert(!results.empty() && "ONNX output results must not be empty!");

    // Get raw tensor pointers
    const float* predLabel = results[1].GetTensorData<float>();
    const float* predMask = results[2].GetTensorData<float>();

    // Get tensor info
    Ort::TensorTypeAndShapeInfo tensorInfoLabel = results[1].GetTensorTypeAndShapeInfo();
    Ort::TensorTypeAndShapeInfo tensorInfoMask  = results[2].GetTensorTypeAndShapeInfo();

    size_t numLabels = tensorInfoLabel.GetElementCount();
    size_t numMasks  = tensorInfoMask.GetElementCount();

    // Convert tensor to 2D vector [N, 6] for labels
    std::vector<std::vector<float>> labelArray;
    tensorTo2DVector(predLabel, numLabels, labelArray, 6);

    // Identify indices for top and bottom masks
    std::vector<size_t> topIndices;
    std::vector<size_t> bottomIndices;
    size_t idx = 0;
    for (const auto& row : labelArray) {
        if (static_cast<int>(row[0]) == 0 && row[1] > threshold) {
            topIndices.push_back(idx);
        }
        if (static_cast<int>(row[0]) == 1 && row[1] > threshold) {
            bottomIndices.push_back(idx);
        }
        ++idx;
    }

    // Copy mask tensor to vector
    std::vector<float> maskData(predMask, predMask + numMasks);

    // Convert vector to OpenCV Mat with multiple channels
    std::vector<cv::Mat> maskMats = floatVectorToMatChannels(maskData,
                                                      tensorInfoMask.GetShape()[1],
                                                      tensorInfoMask.GetShape()[2],
                                                      static_cast<int>(idx));

    // Sum masks for top and bottom using selected indices
    mergeMasksByIndex(maskMats, topIndices, topMask);
    mergeMasksByIndex(maskMats, bottomIndices, bottomMask);
}

}

namespace geom {

using namespace Utils;
// Compute the top center and its displacement based on top and bottom masks
void computeTopCenterDisplacement(const cv::Mat& topMask, 
                                  const cv::Mat& bottomMask, 
                                  int width, int height,
                                  cv::Point& topCenter,
                                  cv::Point& displacedTopCenter,
                                  geom::PointSet& displacementPoints)
{

    
    assert(!topMask.empty() && !bottomMask.empty());

    int centerX = 0, centerY = 0;
    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(topMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        topCenter = displacedTopCenter = cv::Point(-1, -1);
        displacementPoints = geom::PointSet();
        return;
    }

    cv::Moments m = cv::moments(contours[0], false);

    if (m.m00 <= 5000) {
        topCenter = displacedTopCenter = cv::Point(-1, -1);
        displacementPoints = geom::PointSet();
        return;
    }

    centerX = static_cast<int>(m.m10 / m.m00);
    centerY = static_cast<int>(m.m01 / m.m00);

    topCenter = cv::Point(centerX, centerY);

    cv::imshow("TopMask", topMask);
    cv::imshow("TopBelow",bottomMask );
    cv::waitKey(1);

    // Gaussian + Canny
    cv::Mat blurredTop, blurredBottom;
    cv::GaussianBlur(topMask, blurredTop, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(bottomMask, blurredBottom, cv::Size(5, 5), 1.5);

    cv::Mat edgesTop, edgesBottom;
    cv::Canny(blurredTop, edgesTop, 50, 150);
    cv::Canny(blurredBottom, edgesBottom, 50, 150);

    std::vector<cv::Point> nonZeroTop, nonZeroBottom;
    cv::findNonZero(edgesTop, nonZeroTop);
    cv::findNonZero(edgesBottom, nonZeroBottom);

    geom::PointSet topEdgePoints(nonZeroTop);
    geom::PointSet bottomEdgePoints(nonZeroBottom);

    if (bottomEdgePoints.size() == 0) {
        displacedTopCenter = topCenter;
        displacementPoints = geom::PointSet({ topCenter });
        return;
    }



    displacedTopCenter = topCenter;        
    displacementPoints = geom::PointSet(); 
    

    // Compute neighbor points
    NeighborPointSet neighbor = geom::findNeighborPointSetsWithThreshold(bottomEdgePoints,topEdgePoints, 5 );
    //calculate distance and displacement vector
    auto displacementVectors = neighbor.primaryA.computeDistanceMatrix( neighbor.secondaryA);
    cv::Point displacedVectorIndex = Utils::findMinMaxDisP1P2(displacementVectors);
    size_t SampleRecheckNumber = 20;

    cv::Point movedSecondPoint = neighbor.secondaryA[displacedVectorIndex.x]; 
    //get more sample moved point to recheck

    geom::PointSet samplePrimaryA = neighbor.primaryA(displacedVectorIndex.y - SampleRecheckNumber,displacedVectorIndex.y + SampleRecheckNumber );

    // move point Primary
    geom::PointSet movedVecters = samplePrimaryA - movedSecondPoint;
    geom::PointSetArray movedSamplePrimaryA  = neighbor.primaryA + movedVecters ;


    // calc distance movedSamplePrimaryA to secondaryB

    size_t optimalIndex  = movedSamplePrimaryA.indexOfMinDistanceTo(neighbor.secondaryB);
    displacedTopCenter = movedVecters[optimalIndex] + topCenter;
    displacementPoints = PointSet({neighbor.primaryA[optimalIndex], neighbor.secondaryA[displacedVectorIndex.y]});

    // rowMins =Utils::sumMinPerRow(displacementVectors)
    // // Compute displacement vector and move primary points
    // PointSet displacementVector = neighbor.primaryA - neighbor.secondaryA;

    // PointSet movedPrimary = neighbor.primaryA + displacementVector;

    // PointSetArray  
    // size_t optimalIndex = movedPrimary.indexOfMinDistanceTo(neighbor.secondaryA);

    // // Final results
    // displacedTopCenter = displacementVector[optimalIndex] + topCenter;
    // displacementPoints = PointSet({neighbor.primaryA[optimalIndex], neighbor.secondaryA[0]});
}

// Convert a point from a square/cropped image back to the original image coordinates
cv::Point convertSquareToOriginalImage(const cv::Mat& originalImage, 
                                       const cv::Rect& boundingBox, 
                                       const cv::Point& squarePoint)
{
    cv::Size imageSize = originalImage.size();
    int maxDim = std::max(imageSize.width, imageSize.height);
    cv::Point offset = (cv::Point(maxDim, maxDim) - cv::Point(imageSize.width, imageSize.height)) / 2;
    cv::Point origin = cv::Point(boundingBox.x, boundingBox.y);

    return origin - offset + squarePoint;
}

} // namespace geom
