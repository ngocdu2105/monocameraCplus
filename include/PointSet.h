#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iterator>

// #include "include/PointSetArray.h"


namespace geom {


class PointSet;

class PointSetArray {
public:
    std::vector<PointSet> sets;

    const PointSet& operator[](size_t i) const;
    PointSet& operator[](size_t i);

    size_t indexOfMinDistanceTo(const PointSet& target) const;
    // PointSetArray operator+(const PointSet& p) const;
};

class PointSet {
private:
    std::vector<cv::Point> points;

public:
    using value_type = cv::Point;

    PointSet() = default;
    explicit PointSet(const std::vector<cv::Point>& pts);

    void add(const cv::Point& p);
    PointSet& append(const cv::Point& p);

    PointSet translate(const cv::Point& offset) const;
    PointSet operator+(const cv::Point& offset) const { return translate(offset); }
    PointSet operator-(const cv::Point& offset) const;

    const cv::Point& operator[](size_t index) const;
    cv::Point& operator[](size_t index);

    auto begin() { return points.begin(); }
    auto end() { return points.end(); }
    auto begin() const { return points.begin(); }
    auto end() const { return points.end(); }
    PointSetArray operator+(const PointSet& offsetVecter) const;
    std::vector<std::vector<double>> computeDistanceMatrix(const PointSet& other) const;
    // size_t indexOfMinDistanceTo(const PointSet& other) const;
    size_t size() const { return points.size(); }
    const std::vector<cv::Point>& data() const { return points; }
    void print() const;
    PointSet slice(size_t start, size_t end) const;
    PointSet operator()(size_t start, size_t end) const;
    ~PointSet() { points.clear(); }
};











} // namespace geom
