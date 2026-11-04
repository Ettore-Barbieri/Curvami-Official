#pragma once
#include <cmath>
#include <vector>

// THE fold-angle utility (C++ side). Route every fold angle through it.
//
// The +/-180 branch cut has produced four separate wrong, publishable-looking
// results in this project:
//
//   1. ComputeFoldAngleDeg averaged raw atan2 values per Gauss point, so near
//      flat-fold the mean swept through zero while the crease barely moved
//      (-177.7 -> -126.8 -> +178.4 across steps where every passive crease moved
//      less than 1 deg per step).
//   2. The target check compared a raw value against +180, so a crease
//      approaching -180 read as 357 deg away and the run never terminated - it
//      ran past flat-fold, reversed, and unfolded.
//   3. diagnose.py took abs() of exported per-GP angles straddling the cut
//      (-179.65 and +178.90 for one physical state), reporting a crease at
//      "171 centre / 172.9 ends" that the solver had at -179.802.
//   4. The original per-crease pairing compared one crease's curvature against
//      another crease's angle.
//
// Each was caught by accident. No ad-hoc atan2 averaging, no abs(), no scattered
// modulo arithmetic. analysis/fold_angles.py is the Python half and must agree.
namespace FoldAngle {

    // Weighted circular mean, in radians, in (-pi, pi].
    inline double circularMeanRad(const std::vector<double>& theta,
                                  const std::vector<double>& weight) {
        double s = 0.0, c = 0.0;
        const size_t n = theta.size();
        for (size_t i = 0; i < n; ++i) {
            const double w = (i < weight.size()) ? weight[i] : 1.0;
            s += std::sin(theta[i]) * w;
            c += std::cos(theta[i]) * w;
        }
        return std::atan2(s, c);
    }

    // Shift one angle onto the branch within pi of `ref`. Values away from the cut
    // come back unchanged; a value just past flat stays past flat rather than
    // being folded back, which matters because there is no contact model.
    inline double unwrapAboutRad(double theta, double ref) {
        double d = theta - ref;
        while (d >  M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        return ref + d;
    }

    // Weighted mean of a set of per-Gauss-point angles, all put on one branch
    // first. This is what a crease's reported angle must be.
    inline double weightedMeanAboutCircularMeanRad(const std::vector<double>& theta,
                                                   const std::vector<double>& weight) {
        if (theta.empty()) return 0.0;
        const double ref = circularMeanRad(theta, weight);
        double acc = 0.0, wsum = 0.0;
        for (size_t i = 0; i < theta.size(); ++i) {
            const double w = (i < weight.size()) ? weight[i] : 1.0;
            acc  += unwrapAboutRad(theta[i], ref) * w;
            wsum += w;
        }
        return (wsum > 1e-14) ? acc / wsum : 0.0;
    }

    // Difference a - b taken the short way round, in (-180, 180]. Use this for
    // every "have we reached the target" test: -179.5 is 0.5 deg from +180.
    inline double wrappedDistanceDeg(double a_deg, double b_deg) {
        return std::remainder(a_deg - b_deg, 360.0);
    }

}  // namespace FoldAngle
