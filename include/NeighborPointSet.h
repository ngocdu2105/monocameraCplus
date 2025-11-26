#pragma once
#include "include/PointSet.h"

namespace geom {

struct NeighborPointSet {
    PointSet primaryA;
    PointSet primaryB;
    PointSet secondaryA;
    PointSet secondaryB;

    void printPrimaryA() const;
    void printPrimaryB() const;
    void printSecondaryA() const;
    void printSecondaryB() const;

    PointSet slicePrimaryA(size_t begin, size_t end) const;
};

// Factory function
NeighborPointSet makeNeighborSet(
    const std::vector<cv::Point>& primaryA,
    const std::vector<cv::Point>& primaryB,
    const std::vector<cv::Point>& secondaryA,
    const std::vector<cv::Point>& secondaryB
);

PointSetArray makePointSetArray(const std::vector<PointSet>& sets);

} // namespace geom
