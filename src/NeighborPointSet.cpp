#include "include/NeighborPointSet.h"
#include <cassert>
#include <numeric>

namespace geom {

// ---------------- Print methods ----------------
void NeighborPointSet::printPrimaryA() const { primaryA.print(); }
void NeighborPointSet::printPrimaryB() const { primaryB.print(); }
void NeighborPointSet::printSecondaryA() const { secondaryA.print(); }
void NeighborPointSet::printSecondaryB() const { secondaryB.print(); }

// ---------------- Slice ----------------
PointSet NeighborPointSet::slicePrimaryA(size_t begin, size_t end) const {
    assert(begin <= end && end <= primaryA.size());
    std::vector<cv::Point> subset(primaryA.data().begin() + begin,
                                  primaryA.data().begin() + end);
    return PointSet(subset);
}

// ---------------- Factory ----------------
NeighborPointSet makeNeighborSet(
    const std::vector<cv::Point>& primaryA,
    const std::vector<cv::Point>& primaryB,
    const std::vector<cv::Point>& secondaryA,
    const std::vector<cv::Point>& secondaryB
) {
    assert(!primaryA.empty() && !primaryB.empty() &&
           !secondaryA.empty() && !secondaryB.empty());

    NeighborPointSet nbs;
    nbs.primaryA = PointSet(primaryA);
    nbs.primaryB = PointSet(primaryB);
    nbs.secondaryA = PointSet(secondaryA);
    nbs.secondaryB = PointSet(secondaryB);

    return nbs;
}

PointSetArray makePointSetArray(const std::vector<PointSet>& sets) {
    assert(!sets.empty());
    PointSetArray psa;
    psa.sets = sets;
    return psa;
}

} // namespace geom
