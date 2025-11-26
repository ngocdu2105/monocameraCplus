#include <cassert>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <numeric>
#include "include/PointSet.h"
// #include "include/PointSetArray.h"
#include "include/Utils.h"


namespace geom {

// ---------------- Constructors ----------------
PointSet::PointSet(const std::vector<cv::Point>& pts) : points(pts) {}

// ---------------- Modifiers ----------------
void PointSet::add(const cv::Point& p) {
    points.push_back(p);
}

PointSet& PointSet::append(const cv::Point& p) {
    points.push_back(p);
    return *this;
}

PointSet PointSet::translate(const cv::Point& offset) const {
    std::vector<cv::Point> result;
    result.reserve(points.size());
    std::transform(points.begin(), points.end(), std::back_inserter(result),
                   [&offset](const cv::Point& pt){ return pt + offset; });
    return PointSet(result);
}

PointSet PointSet::operator-(const cv::Point& offset) const {
    std::vector<cv::Point> result;
    result.reserve(points.size());
    std::transform(points.begin(), points.end(), std::back_inserter(result),
                   [&offset](const cv::Point& pt){ return pt - offset; });
    return PointSet(result);
}

// ---------------- Accessors ----------------
const cv::Point& PointSet::operator[](size_t index) const {
    return points.at(index);
}

cv::Point& PointSet::operator[](size_t index) {
    return points.at(index);
}

void PointSet::print() const {
    std::cout << "[";
    for (const auto& pt : points) {
        std::cout << "(" << pt.x << "," << pt.y << "),";
    }
    std::cout << "]\n";
}

// ---------------- Distance matrix ----------------
std::vector<std::vector<double>> PointSet::computeDistanceMatrix(const PointSet& other) const {
    std::vector<std::vector<double>> distances;
    distances.reserve(points.size());

    for (const auto& pt1 : points) {
        std::vector<double> row;
        row.reserve(other.points.size());
        for (const auto& pt2 : other.points) {
            row.push_back(cv::norm(pt1 - pt2));
        }
        distances.push_back(row);
    }
    return distances;
}

PointSet PointSet::slice(size_t start, size_t end) const {
    PointSet result;
    if (start >= points.size()) return result;
    if (end > points.size()) end = points.size();

    for (size_t i = start; i < end; ++i)
        result.add(points[i]);

    return result;
}

PointSet PointSet::operator()(size_t start, size_t end) const {
    return slice(start, end);
}

PointSetArray PointSet::operator+(const PointSet& offsetVecter) const
{
    PointSetArray result;

    // Duyệt từng điểm của offsetVecter
    for (const auto& off : offsetVecter.data())
    {
        std::vector<cv::Point> movedPoints;
        movedPoints.reserve(points.size());

        // Mỗi điểm trong PointSet gốc + offset
        for (const auto& p : points)
        {
            movedPoints.push_back(p + off);
        }

        // Thêm vào mảng PointSetArray
        result.sets.emplace_back(movedPoints);
    }

    return result;
}



const PointSet& PointSetArray::operator[](size_t i) const {
    return sets.at(i);
}

PointSet& PointSetArray::operator[](size_t i) {
    return sets.at(i);
}

size_t PointSetArray::indexOfMinDistanceTo(const PointSet& target) const {
    assert(!sets.empty());
    std::vector<double> minSums;
    minSums.reserve(sets.size());

    for (const auto& set : sets) {
        auto distMat = target.computeDistanceMatrix(set);
        minSums.push_back(Utils::sumMinPerRow(distMat));
    }

    auto minIt = std::min_element(minSums.begin(), minSums.end());
    return std::distance(minSums.begin(), minIt);
}


// size_t PointSet::indexOfMinDistanceTo(const PointSet& other) const{
//     assert(other.empty() == false);
//     for (const auto& pt1: points)
//     {
//             std::vector<double> min;
//     std::transform(v.begin(),v.end(),std::back_inserter(min), [&NNBP2](const PointCollection& nnbp1)
//     {
//         std::vector<std::vector<double>> dis = NNBP2 | nnbp1;
//         return SumAllRowMin(dis);

//     });

//     return std::distance(min.begin(),std::min_element(min.begin(),min.end()));
//     }


// }


} // namespace geom
