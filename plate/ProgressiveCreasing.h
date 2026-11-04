#pragma once

// The build stamp (CURVAMI_GIT_COMMIT), generated at every build by
// cmake/BuildStamp.cmake.
#if __has_include("curvami_build_stamp.h")
#include "curvami_build_stamp.h"
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "Meshless.h"
#include "NonLinearSolver.h"
#include "FoldAngleUtil.h"
#include "curvami/Enrichment.h"

#ifndef CURVAMI_GIT_COMMIT
#define CURVAMI_GIT_COMMIT "unknown"
#endif

// =========================================================
// PROGRESSIVE CREASING
// =========================================================
// docs/Progressive_Creasing_Brief_for_Code.md, Step 1. A crease is written along
// a fixed path; it exists only between the tips. At each step:
//   - the tips move to the next vertices of the crease polyline (the vertices are
//     crease-line mesh nodes, which must sit on sheet nodes - checked at setup);
//   - the cut is the polyline truncated at the tips, with the existing end cap of
//     Enrichment::evaluateIndicator at any tip inside the sheet;
//   - every RKPM shape function is rebuilt with that cut, and the resulting jump
//     in position is reported. The coefficients are carried over unchanged, or,
//     with `refit`, re-fitted by least squares to the previous deformed positions
//     at the face Gauss points (Refitter below);
//   - hinge and continuity constraint are on at crease Gauss points between the
//     tips, off elsewhere; theta0 = theta0* on all active points;
//   - one Newton solve (LineSearch). If it fails, theta0 on the newest piece only
//     is raised from 0 to theta0* over a few sub-steps; if that fails too, stop.
// The tip only advances; no energy criterion, no arc-length.
namespace Progressive {

    struct Config {
        // The lines to make and how each grows (mode, tip step) are given per line:
        // see struct Line. Modes: one_end (tip from the first polyline vertex to
        // the last), one_end_reverse, middle_out (two tips from the middle vertex),
        // ends_in (two tips from the two ends, meeting at the middle vertex),
        // all_at_once (the full crease from the start, theta0 raised uniformly).
        int num_steps = 40;          // all_at_once only: theta0 ramp steps
        int fallback_substeps = 4;   // newest-piece theta0 sub-steps after a failed solve
        double L_ref = 1.0;          // facet width L, for the normalised diagnostics
        // Re-fit of the coefficients after each rebuild: none | l2 | h1 (Refitter).
        // Production runs use none (decision of 29 Sep): both fits start the
        // newest crease point far from theta0 and can send the first Newton step
        // far off (see the brief, "Re-fit test"). Kept for comparison only.
        std::string refit = "none";

        // Guard: a converged step is refused, and the run stops with a report, if
        // its largest local fold angle passes guard_max_theta_deg, its total energy
        // exceeds guard_energy_factor times the previous step's, or it moved some
        // face Gauss point by more than guard_max_motion * L_ref, or by more than
        // guard_max_motion_topology * L_ref on a step where the number of cut ends
        // on the sheet edge changes (the crease reaching the edge: completion of
        // example C moved 0.20-0.45 L with falling energy).
        double guard_max_theta_deg = 180.0;
        double guard_energy_factor = 3.0;
        double guard_max_motion = 0.5;
        double guard_max_motion_topology = 1.0;

        // Diagnosis of one step (>= 0 to enable): at that step, report the crease
        // state (continuity jump, fold angles at the crease points) for the start
        // with the coefficients carried over and, with refit, for the re-fitted
        // start; solve the step from each, printing the Newton path; then stop.
        int diagnose_step = -1;
        std::string log_file;        // per-step CSV
    };

    inline std::vector<double> cumulativeArcLength(const Enrichment::FoldLine& path) {
        std::vector<double> S(path.vertices.size(), 0.0);
        for (size_t i = 1; i < S.size(); ++i)
            S[i] = S[i - 1] + (path.vertices[i] - path.vertices[i - 1]).norm();
        return S;
    }

    // Arc length of the closest point of the path to P.
    inline double projectArcLength(const Eigen::Vector2d& P, const Enrichment::FoldLine& path,
                                   const std::vector<double>& S) {
        double best_d = std::numeric_limits<double>::infinity(), best_s = 0.0;
        for (size_t i = 0; i + 1 < path.vertices.size(); ++i) {
            const Eigen::Vector2d A = path.vertices[i], v = path.vertices[i + 1] - A;
            const double v2 = v.squaredNorm();
            if (v2 == 0.0) continue;
            const double t = std::max(0.0, std::min(1.0, (P - A).dot(v) / v2));
            const double d = (P - (A + t * v)).norm();
            if (d < best_d) { best_d = d; best_s = S[i] + t * std::sqrt(v2); }
        }
        return best_s;
    }

    // The cut between vertices a and b. An end that is a tip inside the sheet gets
    // the end cap (extend = false); an end that is also an end of the full path
    // keeps the full path's flags, so at a = 0, b = N the cut is the full crease.
    inline Enrichment::FoldLine truncate(const Enrichment::FoldLine& full, int a, int b) {
        const int N = static_cast<int>(full.vertices.size()) - 1;
        Enrichment::FoldLine cut;
        cut.vertices.assign(full.vertices.begin() + a, full.vertices.begin() + b + 1);
        cut.extend_start = (a == 0) ? full.extend_start : false;
        cut.extend_end   = (b == N) ? full.extend_end   : false;
        cut.is_closed    = full.is_closed && a == 0 && b == N;
        return cut;
    }

    // Deformed-minus-reference position at every face Gauss point, with the
    // shape functions currently stored.
    inline std::vector<Eigen::Vector3d> faceDisplacements(const Meshless& plate, const Eigen::VectorXd& x) {
        std::vector<Eigen::Vector3d> u(plate.faces.gps.size(), Eigen::Vector3d::Zero());
        for (size_t g = 0; g < plate.faces.gps.size(); ++g) {
            const auto& gp = plate.faces.gps[g];
            for (size_t j = 0; j < gp.neighbor_node_ids.size(); ++j)
                u[g] += gp.Psi[j] * x.segment<3>(gp.neighbor_node_ids[j] * 3);
        }
        return u;
    }

    // Largest pointwise distance between two displacement sets, and where.
    inline double maxDistance(const Meshless& plate, const std::vector<Eigen::Vector3d>& u1,
                              const std::vector<Eigen::Vector3d>& u2, Eigen::Vector2d* where = nullptr) {
        double best = 0.0;
        for (size_t g = 0; g < u1.size(); ++g) {
            const double d = (u1[g] - u2[g]).norm();
            if (d > best) { best = d; if (where) *where = plate.faces.gps[g].X_ref; }
        }
        return best;
    }

    // Negative modes of the reduced Hessian, read from the last KKT factorization
    // of the Newton solve that just converged: INFOG(12) minus the constraint
    // count (see StepController). No extra factorization. That factorization is at
    // the last iterate before convergence, so it differs from the converged state
    // only by one converged Newton correction. It measures the stability of the
    // real problem only with fold_exact_tangent on: the rank-one hinge tangent
    // alone can never be negative.
    inline int lastNegativeModes(const NonLinearSolver& solver) {
        return solver.kkt_solver.negativePivots() - static_cast<int>(solver.ind_rows.size());
    }

    // Mean (shared circular-mean utility) and largest |theta| over the active
    // crease points of one fold, in degrees.
    inline void activeFoldAngles(const Meshless& plate, const Fold& fold, double& mean_deg, double& max_deg) {
        std::vector<double> th, w;
        for (size_t i = 0; i < fold.left.gps.size(); ++i) {
            if (!Constraints::gpActive(&fold.gp_active, i)) continue;
            const auto& gL = fold.left.gps[i];
            const auto& gR = fold.right.gps[i];
            auto kinL = Kinematics::compute(gL, plate.nodes);
            auto kinR = Kinematics::compute(gR, plate.nodes);
            Eigen::Matrix<double, 12, 1> F;
            F.segment<3>(0) = kinL.F0.col(0); F.segment<3>(3) = kinL.F0.col(1);
            F.segment<3>(6) = kinR.F0.col(0); F.segment<3>(9) = kinR.F0.col(1);
            th.push_back(Constraints::foldAngleAndGrad(F, gL.tangent.x(), gL.tangent.y(), nullptr));
            w.push_back(gL.weight);
        }
        mean_deg = max_deg = 0.0;
        if (th.empty()) return;
        const double ref = FoldAngle::circularMeanRad(th, w);
        mean_deg = FoldAngle::weightedMeanAboutCircularMeanRad(th, w) * 180.0 / M_PI;
        for (double t : th)
            max_deg = std::max(max_deg, std::abs(FoldAngle::unwrapAboutRad(t, ref)) * 180.0 / M_PI);
    }

    // Displacement and its first derivatives (3x2: d/dX, d/dY) at a set of points,
    // with the shape functions currently stored on them.
    struct FieldSample {
        std::vector<Eigen::Vector3d> u;
        std::vector<Eigen::Matrix<double, 3, 2>> G;
    };

    inline FieldSample sampleField(const std::vector<GaussPointData>& gps, const Eigen::VectorXd& x) {
        FieldSample f;
        f.u.assign(gps.size(), Eigen::Vector3d::Zero());
        f.G.assign(gps.size(), Eigen::Matrix<double, 3, 2>::Zero());
        for (size_t g = 0; g < gps.size(); ++g) {
            const auto& gp = gps[g];
            for (size_t j = 0; j < gp.neighbor_node_ids.size(); ++j) {
                const Eigen::Vector3d a = x.segment<3>(gp.neighbor_node_ids[j] * 3);
                f.u[g] += gp.Psi[j] * a;
                f.G[g].col(0) += gp.dPsi[j].x() * a;
                f.G[g].col(1) += gp.dPsi[j].y() * a;
            }
        }
        return f;
    }

    // What the re-fit reproduces: the previous deformed state, sampled with the
    // shape functions it was computed with (before the rebuild).
    struct RefitTarget {
        FieldSample face;
        std::vector<FieldSample> left, right;   // per fold: the crease points' two sides
    };

    inline RefitTarget sampleTarget(const Meshless& plate, const Eigen::VectorXd& x) {
        RefitTarget t;
        t.face = sampleField(plate.faces.gps, x);
        for (const auto& f : plate.folds) {
            t.left.push_back(sampleField(f.left.gps, x));
            t.right.push_back(sampleField(f.right.gps, x));
        }
        return t;
    }

    // Least-squares re-fit of the coefficients after the shape functions change,
    // for the three displacement components at once. Two modes:
    //
    //  l2: minimise sum_g w_g |u(a)_g - u_g|^2 over the face Gauss points
    //      (w_g the quadrature weight): the L2 fit of the displacement.
    //
    //  h1: also match the first derivatives, and include the crease Gauss points
    //      on both sides:
    //        sum_g w_g ( |u(a)_g - u_g|^2 + l^2 |G(a)_g - G_g|^2 )
    //      over face points (w_g = area weight) and crease points (w_g = line
    //      weight x support radius rho, the area of a strip one support wide),
    //      with l = L_ref so both terms are in the same units. Targets are the
    //      previous state's values and slopes; at a crease point the cut has just
    //      reached, that is the previous continuous field on both sides, so the
    //      re-fitted start has no gap there and the previous local slopes. The fold
    //      angle depends on the slopes, which the l2 fit leaves free (step 9 of the
    //      first re-fit run: newest point at 20 deg against 36 carried over).
    //
    // Normal equations M a = b. The face part of M has the node-pair pattern of
    // the stiffness and is filled in place through the assembler's per-Gauss-point
    // CSR map; the crease part (h1) is added from triplets.
    class Refitter {
    public:
        Refitter(const Meshless& plate, const std::string& mode, double l)
            : h1_(mode == "h1"), l2len_(l * l) {
            const auto& K = plate.face_assembler.K;          // 3x3 blocks, upper
            const int nn = static_cast<int>(plate.nodes.size());
            // Node-level pattern: column j holds the nodes of K's column 3j, one per
            // 3x3 block (rows 3i, 3i+1, 3i+2 are consecutive).
            std::vector<Eigen::Triplet<double>> trips;
            for (int j = 0; j < nn; ++j) {
                const int beg = K.outerIndexPtr()[3 * j], end = K.outerIndexPtr()[3 * j + 1];
                for (int p = beg; p < end; p += 3)
                    trips.emplace_back(K.innerIndexPtr()[p] / 3, j, 0.0);
            }
            M_.resize(nn, nn);
            M_.setFromTriplets(trips.begin(), trips.end());
            M_.makeCompressed();
            if (!h1_) ldlt_.analyzePattern(M_);
        }

        // Overwrites x with the fitted coefficients. False if M is not positive
        // definite (x is then left unchanged).
        bool fit(const Meshless& plate, const RefitTarget& t, Eigen::VectorXd& x) {
            const int nn = static_cast<int>(plate.nodes.size());
            const double l2 = h1_ ? l2len_ : 0.0;
            std::fill(M_.valuePtr(), M_.valuePtr() + M_.nonZeros(), 0.0);
            Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(nn, 3);
            const int* outer = M_.outerIndexPtr();
            double* val = M_.valuePtr();
            const auto& gps = plate.faces.gps;
            for (size_t g = 0; g < gps.size(); ++g) {
                const auto& gp = gps[g];
                const auto& map = plate.face_assembler.csrMapping(g);
                const size_t n = gp.neighbor_node_ids.size();
                const double w = gp.weight;
                size_t mo = 0;
                for (size_t i = 0; i < n; ++i) {
                    const int ni = gp.neighbor_node_ids[i];
                    rhs.row(ni) += (w * gp.Psi[i]) * t.face.u[g].transpose();
                    if (h1_) rhs.row(ni) += (w * l2) * (gp.dPsi[i].x() * t.face.G[g].col(0)
                                                        + gp.dPsi[i].y() * t.face.G[g].col(1)).transpose();
                    for (size_t j = i; j < n; ++j) {
                        const int nj = gp.neighbor_node_ids[j];
                        double m = gp.Psi[i] * gp.Psi[j];
                        if (h1_) m += l2 * gp.dPsi[i].dot(gp.dPsi[j]);
                        val[outer[std::max(ni, nj)] + map[mo++]] += w * m;
                    }
                }
            }

            if (!h1_) {
                ldlt_.factorize(M_);
            } else {
                std::vector<Eigen::Triplet<double>> trips;
                auto add = [&](const std::vector<GaussPointData>& cgps, const FieldSample& tgt) {
                    for (size_t g = 0; g < cgps.size(); ++g) {
                        const auto& gp = cgps[g];
                        const double w = gp.weight * plate.rho;
                        const size_t n = gp.neighbor_node_ids.size();
                        for (size_t i = 0; i < n; ++i) {
                            const int ni = gp.neighbor_node_ids[i];
                            rhs.row(ni) += (w * gp.Psi[i]) * tgt.u[g].transpose()
                                         + (w * l2) * (gp.dPsi[i].x() * tgt.G[g].col(0)
                                                       + gp.dPsi[i].y() * tgt.G[g].col(1)).transpose();
                            for (size_t j = 0; j < n; ++j) {
                                const int nj = gp.neighbor_node_ids[j];
                                if (ni > nj) continue;       // upper triangle
                                trips.emplace_back(ni, nj, w * (gp.Psi[i] * gp.Psi[j] + l2 * gp.dPsi[i].dot(gp.dPsi[j])));
                            }
                        }
                    }
                };
                for (size_t fi = 0; fi < plate.folds.size(); ++fi) {
                    add(plate.folds[fi].left.gps, t.left[fi]);
                    add(plate.folds[fi].right.gps, t.right[fi]);
                }
                Eigen::SparseMatrix<double> Mc(nn, nn);
                Mc.setFromTriplets(trips.begin(), trips.end());
                Mh1_ = M_ + Mc;
                ldlt_.compute(Mh1_);     // the ordering is cheap next to the fill
            }
            if (ldlt_.info() != Eigen::Success) return false;
            const Eigen::MatrixXd a = ldlt_.solve(rhs);
            if (!a.allFinite()) return false;
            for (int i = 0; i < nn; ++i) x.segment<3>(3 * i) = a.row(i).transpose();
            return true;
        }

    private:
        bool h1_;
        double l2len_;
        Eigen::SparseMatrix<double> M_, Mh1_;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> ldlt_;
    };

    // State of the crease at the active points for coefficients x: the largest
    // displacement jump |uL - uR| across the crease (the continuity constraint's
    // violation) and the range of the local fold angle, over all active points and
    // over the newest ones only.
    // Gap |uL - uR| / L and local fold angle (deg) at crease point i. Needs
    // plate.updateKinematics(x) first.
    inline void creasePointState(const Meshless& plate, const Fold& fold, size_t i, double L,
                                 double& gap, double& th) {
        const auto& gL = fold.left.gps[i];
        const auto& gR = fold.right.gps[i];
        Eigen::Vector3d uL = Eigen::Vector3d::Zero(), uR = Eigen::Vector3d::Zero();
        for (size_t j = 0; j < gL.Psi.size(); ++j) uL += gL.Psi[j] * plate.nodes[gL.neighbor_node_ids[j]].u;
        for (size_t j = 0; j < gR.Psi.size(); ++j) uR += gR.Psi[j] * plate.nodes[gR.neighbor_node_ids[j]].u;
        gap = (uL - uR).norm() / L;
        auto kinL = Kinematics::compute(gL, plate.nodes);
        auto kinR = Kinematics::compute(gR, plate.nodes);
        Eigen::Matrix<double, 12, 1> F;
        F.segment<3>(0) = kinL.F0.col(0); F.segment<3>(3) = kinL.F0.col(1);
        F.segment<3>(6) = kinR.F0.col(0); F.segment<3>(9) = kinR.F0.col(1);
        th = Constraints::foldAngleAndGrad(F, gL.tangent.x(), gL.tangent.y(), nullptr) * 180.0 / M_PI;
    }

    // A crease point of the run: (fold index in plate.folds, Gauss point index).
    typedef std::pair<int, size_t> CreasePoint;

    inline void printCreaseState(const char* label, Meshless& plate, const std::vector<int>& folds,
                                 const Eigen::VectorXd& x, const std::vector<CreasePoint>& newest, double L) {
        plate.updateKinematics(x);
        double gap_all = 0.0, th_lo = 1e9, th_hi = -1e9;
        for (int f : folds) {
            const Fold& fold = plate.folds[f];
            for (size_t i = 0; i < fold.left.gps.size(); ++i) {
                if (!Constraints::gpActive(&fold.gp_active, i)) continue;
                double g, t; creasePointState(plate, fold, i, L, g, t);
                gap_all = std::max(gap_all, g); th_lo = std::min(th_lo, t); th_hi = std::max(th_hi, t);
            }
        }
        printf("   [%s] active crease points: max |uL-uR| = %.3e L, fold angle %.3f .. %.3f deg\n",
               label, gap_all, th_lo, th_hi);
        for (const auto& c : newest) {
            double g, t; creasePointState(plate, plate.folds[c.first], c.second, L, g, t);
            printf("   [%s]   newest point s = %.4f: |uL-uR| = %.3e L, fold angle %.3f deg\n",
                   label, plate.folds[c.first].gp_s[c.second], g, t);
        }
    }

    // One line of the making: one cut (a single polyline, `full`) that carries one
    // or more folds, each with its own rest angle and kind. The tip(s) travel along
    // the whole polyline; a crease point takes the rest angle of its own fold.
    struct Line {
        std::vector<int> folds;      // indices in plate.folds
        Enrichment::FoldLine full;   // the cut, built exactly as the all-at-once cut is
        std::string mode = "one_end";
        int tip_step = 1;
        std::vector<int> skip;       // path vertices where a tip must not stop (stepped over)
    };

    // The crease exists between path vertices a and b (outside false), or, for two
    // tips converging from the two ends, on [0, a] and [b, N] (outside true).
    struct Stage { int a, b; double frac; bool outside = false; };

    inline std::vector<Stage> schedule(const std::string& mode, int tip_step, int num_steps, int N,
                                       const std::vector<int>& skip = {}) {
        std::vector<Stage> st;
        const int m = std::max(1, tip_step);
        auto skipped = [&](int v) { return std::find(skip.begin(), skip.end(), v) != skip.end(); };
        if (mode == "all_at_once") {
            for (int k = 0; k <= num_steps; ++k)
                st.push_back({0, N, static_cast<double>(k) / num_steps});
        } else if (mode == "one_end") {
            for (int k = 0;; ++k) {
                int b = std::min(k * m, N);
                if (b == N || !skipped(b)) st.push_back({0, b, 1.0});
                if (b == N) break;
            }
        } else if (mode == "one_end_reverse") {
            for (int k = 0;; ++k) {
                int a = std::max(N - k * m, 0);
                if (a == 0 || !skipped(a)) st.push_back({a, N, 1.0});
                if (a == 0) break;
            }
        } else if (mode == "middle_out") {
            const int c = N / 2;
            if (N % 2 != 0)
                printf("!!! WARNING: crease path has an odd number of segments (%d); middle_out starts at vertex %d, off-centre.\n", N, c);
            for (int k = 0;; ++k) {
                int a = std::max(c - k * m, 0), b = std::min(c + k * m, N);
                st.push_back({a, b, 1.0});
                if (a == 0 && b == N) break;
            }
        } else if (mode == "ends_in") {
            // Two tips leave the two ends and converge on the middle vertex, where
            // the two pieces join into the full cut.
            const int c = N / 2;
            if (N % 2 != 0)
                printf("!!! WARNING: crease path has an odd number of segments (%d); ends_in meets at vertex %d, off-centre.\n", N, c);
            for (int k = 0;; ++k) {
                int a = std::min(k * m, c), b = std::max(N - k * m, c);
                const bool last = (a == c && b == c);
                if (last || (!skipped(a) && !skipped(b))) st.push_back({a, b, 1.0, true});
                if (last) break;
            }
        } else {
            throw std::runtime_error("Progressive creasing: unknown mode '" + mode + "'");
        }
        return st;
    }

    // A step of the whole making: line `line` is cut between its vertices a and b;
    // the lines before it are complete, the lines after it are absent (no cut, no
    // hinge, no constraint).
    struct GStage { int line, a, b; double frac; bool outside; };

    // `rebuild` recomputes every shape function for a given set of cuts (it is
    // SetupHelper::recomputeShapeFunctions; passed in to keep this header free of
    // the setup layer). `static_creases` are cuts that do not change. The lines are
    // made in the order given. On entry the shape functions must be those of ALL
    // cuts in full (as main.cpp leaves them).
    inline void run(NonLinearSolver& solver, Meshless& plate, Eigen::VectorXd& x,
                    const std::vector<Line>& lines,
                    const std::vector<Enrichment::FoldLine>& static_creases,
                    const Config& cfg,
                    const std::function<void(const std::vector<Enrichment::FoldLine>&)>& rebuild,
                    const std::function<void(int, const Eigen::VectorXd&, const std::vector<double>&,
                                             const std::vector<int>&, double, double, double)>& on_converged)
    {
        if (lines.empty()) throw std::runtime_error("Progressive creasing: no line to make");
        const int nlines = static_cast<int>(lines.size());
        auto thetaStar = [&](int f) { return plate.folds[f].fold_sign * plate.folds[f].angle_target; };

        // Per line: arc length along its cut; per fold: arc length of its crease points.
        std::vector<std::vector<double>> Sl(nlines);
        std::vector<int> Nl(nlines);
        std::vector<int> all_folds;
        double Lc = 0.0, worst_tip = 0.0;
        size_t ngp = 0;
        for (int l = 0; l < nlines; ++l) {
            const Enrichment::FoldLine& full = lines[l].full;
            Nl[l] = static_cast<int>(full.vertices.size()) - 1;
            if (Nl[l] < 1) throw std::runtime_error("Progressive creasing: crease path has no segments");
            Sl[l] = cumulativeArcLength(full);
            Lc += Sl[l].back();
            for (int f : lines[l].folds) {
                Fold& fold = plate.folds.at(f);
                const size_t n = fold.left.gps.size();
                fold.gp_s.resize(n);
                for (size_t i = 0; i < n; ++i) fold.gp_s[i] = projectArcLength(fold.left.gps[i].X_ref, full, Sl[l]);
                fold.gp_active.assign(n, 0);
                fold.gp_theta0.assign(n, 0.0);
                all_folds.push_back(f);
                ngp += n;
            }
            // Every path vertex is a potential tip: how far from a sheet node?
            for (const auto& V : full.vertices) {
                double d = std::numeric_limits<double>::infinity();
                for (const auto& n : plate.nodes) d = std::min(d, (n.X_ref - V).norm());
                worst_tip = std::max(worst_tip, d);
            }
        }
        const bool multi = all_folds.size() > 1;

        // The whole making: each line's own schedule, one after the other. A later
        // line's empty first stage is dropped: it is the previous line's last state.
        std::vector<GStage> st;
        for (int l = 0; l < nlines; ++l) {
            const auto ls = schedule(lines[l].mode, lines[l].tip_step, cfg.num_steps, Nl[l], lines[l].skip);
            for (size_t k = 0; k < ls.size(); ++k) {
                // Nothing of this line yet: inside, a == b; outside, a = 0 and b = N.
                const bool empty = ls[k].outside ? (ls[k].a == 0 && ls[k].b == Nl[l]) : (ls[k].b == ls[k].a);
                if (l > 0 && empty) continue;
                st.push_back({l, ls[k].a, ls[k].b, ls[k].frac, ls[k].outside});
            }
        }

        printf("\n=================================================================\n");
        printf(" >>> PROGRESSIVE CREASING: %d line(s), %zu fold(s) <<<\n", nlines, all_folds.size());
        printf("=================================================================\n");
        for (int l = 0; l < nlines; ++l) {
            const Enrichment::FoldLine& full = lines[l].full;
            printf(" line %d: mode %s, tip_step %d, L_c = %.6g, %d path segments\n",
                   l, lines[l].mode.c_str(), lines[l].tip_step, Sl[l].back(), Nl[l]);
            printf("   path starts at (%.4f, %.4f), ends at (%.4f, %.4f); extend flags %d/%d, closed %d\n",
                   full.vertices.front().x(), full.vertices.front().y(),
                   full.vertices.back().x(), full.vertices.back().y(),
                   full.extend_start, full.extend_end, full.is_closed);
            for (int f : lines[l].folds)
                printf("   fold %d (crease tag %d): theta0* = %.4f deg, %zu crease GPs, s from %.4f to %.4f\n",
                       f, plate.folds[f].crease_tag, thetaStar(f) * 180.0 / M_PI, plate.folds[f].left.gps.size(),
                       plate.folds[f].gp_s.empty() ? 0.0 : *std::min_element(plate.folds[f].gp_s.begin(), plate.folds[f].gp_s.end()),
                       plate.folds[f].gp_s.empty() ? 0.0 : *std::max_element(plate.folds[f].gp_s.begin(), plate.folds[f].gp_s.end()));
            for (int v : lines[l].skip)
                printf("   tip never stops at path vertex %d (%.4f, %.4f): stepped over\n",
                       v, full.vertices.at(v).x(), full.vertices.at(v).y());
        }
        printf(" total crease length %.6g, %zu crease GPs, %zu steps\n", Lc, ngp, st.size());
        printf(" largest distance from a path vertex to the nearest sheet node: %.3e (%.3e L)\n",
               worst_tip, worst_tip / cfg.L_ref);
        if (!plate.fold_exact_tangent)
            printf("!!! WARNING: fold_exact_tangent is off; the negative-mode count cannot see an instability"
                   " (the rank-one hinge tangent is never negative).\n");
        if (worst_tip > 1e-9 * cfg.L_ref)
            printf("!!! WARNING: crease path vertices are not on sheet nodes; tips will not sit on nodes.\n");

        FILE* log = cfg.log_file.empty() ? nullptr : std::fopen(cfg.log_file.c_str(), "w");
        if (log) {
            std::fprintf(log, "# progressive creasing, git_commit=%s, mode=%s, L_ref=%.10g, L_c=%.10g, N=%d, tip_step=%d, theta0_star_deg=%.10g, fold_exact_tangent=%d, refit=%s, ls_on_fail=%s, guard_max_theta_deg=%g, guard_energy_factor=%g, guard_max_motion=%g, guard_max_motion_topology=%g, lines=%d\n",
                         CURVAMI_GIT_COMMIT, lines[0].mode.c_str(), cfg.L_ref, Lc, Nl[0], lines[0].tip_step,
                         thetaStar(all_folds[0]) * 180.0 / M_PI,
                         plate.fold_exact_tangent ? 1 : 0, cfg.refit.c_str(),
                         solver.ls_fail_on_no_descent ? "fail" : "accept", cfg.guard_max_theta_deg,
                         cfg.guard_energy_factor, cfg.guard_max_motion, cfg.guard_max_motion_topology, nlines);
            std::fprintf(log, "step,a,b,ell,ell_over_Lc,theta0_frac,newton_iters,substeps,E_mem,E_ben,E_hin,"
                              "theta_mean_deg,theta_max_deg,neg_modes,jump_over_L,jump_X,jump_Y,change_over_L,"
                              "jump_refit_over_L,refit_seconds,start_gap_newest_over_L,start_theta_newest_min_deg,start_theta_newest_max_deg,guard_retry,line");
            if (multi)
                for (int f : all_folds)
                    std::fprintf(log, ",theta_mean_tag%d,theta_max_tag%d", plate.folds[f].crease_tag, plate.folds[f].crease_tag);
            std::fprintf(log, "\n");
        }

        std::unique_ptr<Refitter> refitter;
        if (cfg.refit == "l2" || cfg.refit == "h1") refitter = std::make_unique<Refitter>(plate, cfg.refit, cfg.L_ref);
        else if (cfg.refit != "none") throw std::runtime_error("creasing.refit must be none, l2 or h1");

        double prev_E_tot = 0.0;
        int prev_edge_ends = 0;          // cut ends on the sheet edge at the previous step
        int prev_edge_line = 0;
        // The shape functions on entry are every cut in full.
        int prev_line = nlines - 1, prev_a = 0, prev_b = Nl[nlines - 1];
        bool prev_outside = false;
        std::vector<std::vector<char>> prev_active(plate.folds.size());
        for (int f : all_folds) prev_active[f].assign(plate.folds[f].left.gps.size(), 0);

        for (size_t k = 0; k < st.size(); ++k) {
            const GStage& s = st[k];
            const Line& line = lines[s.line];
            const Enrichment::FoldLine& full = line.full;
            const std::vector<double>& S = Sl[s.line];
            const int N = Nl[s.line];
            // Outside (two converging tips): the pieces [0, a] and [b, N]; once the
            // tips meet (a >= b) the line is complete.
            const bool whole = s.outside && s.a >= s.b;
            double ell = s.outside ? (whole ? S[N] : S[s.a] + (S[N] - S[s.b])) : S[s.b] - S[s.a];
            for (int l = 0; l < s.line; ++l) ell += Sl[l].back();

            // --- 1. move the tips: rebuild the shape functions, report the jump ---
            // jump: with the coefficients carried over; jump_refit: what is left
            // after the re-fit (equal to jump when the re-fit is off).
            double jump = 0.0, jump_refit = 0.0, refit_s = 0.0;
            Eigen::Vector2d jump_at = Eigen::Vector2d::Zero();
            Eigen::VectorXd x_carried = x;                 // for diagnose_step
            std::vector<Eigen::Vector3d> u_prev;           // idem
            if (s.line != prev_line || s.a != prev_a || s.b != prev_b || s.outside != prev_outside) {
                const auto u_old = faceDisplacements(plate, x);
                const RefitTarget target = refitter ? sampleTarget(plate, x) : RefitTarget{};
                std::vector<Enrichment::FoldLine> cuts = static_creases;
                for (int l = 0; l < s.line; ++l) cuts.push_back(lines[l].full);
                if (!s.outside) {
                    if (s.b > s.a) cuts.push_back(truncate(full, s.a, s.b));
                } else if (whole) {
                    cuts.push_back(full);
                } else {
                    if (s.a > 0) cuts.push_back(truncate(full, 0, s.a));
                    if (s.b < N) cuts.push_back(truncate(full, s.b, N));
                }
                rebuild(cuts);
                const auto u_new = faceDisplacements(plate, x);
                jump = maxDistance(plate, u_old, u_new, &jump_at) / cfg.L_ref;
                x_carried = x;
                u_prev = u_old;
                jump_refit = jump;
                if (refitter) {
                    const auto t0 = std::chrono::steady_clock::now();
                    Eigen::VectorXd x_fit = x;
                    if (refitter->fit(plate, target, x_fit)) {
                        x = x_fit;
                        jump_refit = maxDistance(plate, u_old, faceDisplacements(plate, x)) / cfg.L_ref;
                    } else {
                        printf("!!! WARNING: re-fit failed (mass matrix not positive definite); coefficients carried over.\n");
                    }
                    refit_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    printf(" >> re-fit: jump %.3e L -> %.3e L in %.3f s\n", jump, jump_refit, refit_s);
                }
                prev_line = s.line; prev_a = s.a; prev_b = s.b; prev_outside = s.outside;
            }

            // --- 2. creases on: completed lines in full, this line between its tips ---
            std::vector<CreasePoint> newest;
            for (int l = 0; l < nlines; ++l) {
                for (int f : lines[l].folds) {
                    Fold& fold = plate.folds[f];
                    for (size_t i = 0; i < fold.left.gps.size(); ++i) {
                        bool on;
                        double frac = 1.0;
                        if (l < s.line) on = true;
                        else if (l > s.line) on = false;
                        else if (!s.outside) { on = s.b > s.a && fold.gp_s[i] > S[s.a] && fold.gp_s[i] < S[s.b]; frac = s.frac; }
                        else { on = whole || fold.gp_s[i] < S[s.a] || fold.gp_s[i] > S[s.b]; frac = s.frac; }
                        fold.gp_active[i] = on ? 1 : 0;
                        fold.gp_theta0[i] = on ? frac * thetaStar(f) : 0.0;
                        if (on && !prev_active[f][i]) newest.push_back({f, i});
                    }
                }
            }

            printf("\n--- Progressive step %zu/%zu | line %d | vertices %s[%d, %d] | ell/L_c = %.4f | theta0 frac = %.4f | jump = %.3e L ---\n",
                   k, st.size() - 1, s.line, s.outside ? "outside " : "", s.a, s.b, ell / Lc, s.frac, jump);

            // Cut ends on the sheet edge: a step that changes their number is a
            // topology change (the crease reaching the edge) with its own motion limit.
            const int edge_ends = s.outside
                ? (((s.a > 0 || whole) && full.extend_start) ? 1 : 0) + (((s.b < N || whole) && full.extend_end) ? 1 : 0)
                : ((s.b > s.a) ? (((s.a == 0 && full.extend_start) ? 1 : 0)
                                  + ((s.b == N && full.extend_end) ? 1 : 0)) : 0);
            const bool topology_step = (edge_ends != prev_edge_ends) || (s.line != prev_edge_line);
            const double motion_limit = topology_step ? cfg.guard_max_motion_topology : cfg.guard_max_motion;

            // The state a step converged to, and the guard's verdict on it. theta:
            // with one fold, its mean and largest |theta|; with several, the
            // length-weighted mean of the folds' |mean| and the largest |theta| of all.
            struct StepState {
                double E_mem = 0, E_ben = 0, E_hin = 0, th_mean = 0, th_max = 0, change = 0; int neg = 0;
                std::vector<double> f_mean, f_max;
            };
            auto evaluate = [&](const Eigen::VectorXd& xs, const std::vector<Eigen::Vector3d>& u0) {
                StepState r;
                plate.updateKinematics(xs);
                r.change = maxDistance(plate, u0, faceDisplacements(plate, xs)) / cfg.L_ref;
                r.neg = lastNegativeModes(solver);
                plate.computeStrainEnergies(xs, r.E_mem, r.E_ben);
                r.E_hin = plate.computeTotalHingeEnergy(xs);
                if (!multi) {
                    activeFoldAngles(plate, plate.folds[all_folds[0]], r.th_mean, r.th_max);
                } else {
                    double wsum = 0.0;
                    for (int f : all_folds) {
                        double m = 0.0, mx = 0.0;
                        activeFoldAngles(plate, plate.folds[f], m, mx);
                        r.f_mean.push_back(m); r.f_max.push_back(mx);
                        double w = 0.0;
                        for (size_t i = 0; i < plate.folds[f].left.gps.size(); ++i)
                            if (plate.folds[f].gp_active[i]) w += plate.folds[f].left.gps[i].weight;
                        r.th_mean += w * std::abs(m); wsum += w;
                        r.th_max = std::max(r.th_max, mx);
                    }
                    if (wsum > 0.0) r.th_mean /= wsum;
                }
                return r;
            };
            auto guardWhy = [&](const StepState& r) {
                std::string why;
                char buf[160];
                const double E_tot = r.E_mem + r.E_ben + r.E_hin;
                if (r.th_max > cfg.guard_max_theta_deg) {
                    std::snprintf(buf, sizeof(buf), " max fold angle %.2f deg > %.1f;", r.th_max, cfg.guard_max_theta_deg);
                    why += buf;
                }
                if (prev_E_tot > 1e-10 && E_tot > cfg.guard_energy_factor * prev_E_tot) {
                    std::snprintf(buf, sizeof(buf), " total energy %.3e -> %.3e (x%.1f > x%.1f);",
                                  prev_E_tot, E_tot, E_tot / prev_E_tot, cfg.guard_energy_factor);
                    why += buf;
                }
                if (r.change > motion_limit) {
                    std::snprintf(buf, sizeof(buf), " step motion %.3f L > %.3f L%s;", r.change, motion_limit,
                                  topology_step ? " (topology-change step)" : "");
                    why += buf;
                }
                return why;
            };

            // Raise theta0 on the newest crease points from 0 to its target over
            // fallback_substeps sub-steps at the same ell, starting from xs.
            auto setNewest = [&](double fac) {
                for (const auto& c : newest) plate.folds[c.first].gp_theta0[c.second] = fac * s.frac * thetaStar(c.first);
            };
            auto rampNewest = [&](Eigen::VectorXd& xs, int& iters) {
                bool okk = true;
                for (int j = 1; j <= cfg.fallback_substeps && okk; ++j) {
                    const double f = static_cast<double>(j) / cfg.fallback_substeps;
                    setNewest(f);
                    printf(">> fallback sub-step %d/%d: theta0 on the newest points at %.4f of its target\n",
                           j, cfg.fallback_substeps, f);
                    okk = solver.LineSearch(plate, 0.0, xs, false);
                    iters += solver.last_ls_attempts;
                    if (!okk) printf(">> fallback sub-step %d failed.\n", j);
                }
                return okk;
            };

            // One step from xs. Newton at the full target. The fallback (theta0
            // ramped on the newest points, from the same start) runs when Newton
            // fails, or when it converges to a state the guard refuses; a state from
            // the fallback is guarded again. Returns false when no acceptable state
            // was found; `why` then holds the guard's reason, or is empty when
            // Newton itself failed. iters counts every Newton iteration done.
            auto fullStep = [&](Eigen::VectorXd& xs, int& iters, int& substeps, int& retry,
                                StepState& r, std::string& why) {
                setNewest(1.0);
                const Eigen::VectorXd x0 = xs;
                const auto u0 = faceDisplacements(plate, xs);
                substeps = 0; retry = 0; why.clear();
                bool okk = solver.LineSearch(plate, 0.0, xs, false);
                iters = solver.last_ls_attempts;
                if (okk) {
                    r = evaluate(xs, u0);
                    why = guardWhy(r);
                    if (why.empty()) return true;
                    printf("\n!!! GUARD: step %zu (ell/L_c = %.4f) converged to a refused state:%s"
                           " (max theta %.2f deg, motion %.3f L) - discarded.\n",
                           k, ell / Lc, why.c_str(), r.th_max, r.change);
                    if (log) std::fprintf(log, "# GUARD REFUSED at step %zu, ell_over_Lc=%.10g:%s theta_max=%.6g, change_over_L=%.6g, iters=%d\n",
                                          k, ell / Lc, why.c_str(), r.th_max, r.change, iters);
                    retry = 1;
                } else {
                    printf(">> Newton failed.\n");
                }
                if (newest.empty() || cfg.fallback_substeps <= 0) return false;
                printf(">> Fallback: raising theta0 on the newest %zu crease point(s) over %d sub-steps from the step's start.\n",
                       newest.size(), cfg.fallback_substeps);
                xs = x0;
                okk = rampNewest(xs, iters);
                substeps = cfg.fallback_substeps;
                if (!okk) { why.clear(); return false; }
                r = evaluate(xs, u0);
                why = guardWhy(r);
                if (!why.empty()) {
                    printf("!!! GUARD: the fallback's state is refused too:%s\n", why.c_str());
                    return false;
                }
                return true;
            };

            // --- diagnosis of one step: both starts, then stop ---
            if (static_cast<int>(k) == cfg.diagnose_step) {
                printf("\n=================================================================\n");
                printf(" >>> DIAGNOSING STEP %zu (ell/L_c = %.4f) <<<\n", k, ell / Lc);
                printf("=================================================================\n");
                std::vector<std::pair<const char*, Eigen::VectorXd>> starts = {{"carried-over", x_carried}};
                if (refitter) starts.push_back({"re-fit", x});
                for (auto& st0 : starts) {
                    const char* label = st0.first;
                    Eigen::VectorXd xs = st0.second;
                    printf("\n--- start: %s ---\n", label);
                    if (!u_prev.empty())
                        printf("   [%s] face points: max distance from the previous deformed state = %.3e L\n",
                               label, maxDistance(plate, u_prev, faceDisplacements(plate, xs)) / cfg.L_ref);
                    printCreaseState(label, plate, all_folds, xs, newest, cfg.L_ref);
                    int d_iters = 0, d_sub = 0, d_retry = 0;
                    StepState r;
                    std::string why;
                    const bool conv = fullStep(xs, d_iters, d_sub, d_retry, r, why);
                    printf("   [%s] RESULT: accepted %d after %d iterations (guard retry %d, fallback sub-steps %d)%s%s\n",
                           label, conv ? 1 : 0, d_iters, d_retry, d_sub,
                           why.empty() ? "" : " | refused:", why.c_str());
                    if (conv) {
                        printf("   [%s] E_mem %.4e E_ben %.4e E_hin %.4e | theta mean %.3f max %.3f deg | neg modes %d | motion %.3e L\n",
                               label, r.E_mem, r.E_ben, r.E_hin, r.th_mean, r.th_max, r.neg, r.change);
                        printCreaseState(label, plate, all_folds, xs, newest, cfg.L_ref);
                    }
                }
                printf("\n>>> Diagnosis of step %zu done; stopping.\n", k);
                break;
            }

            // --- 3. solve ---
            // The start at the newest crease points: continuity gap and fold angle.
            double sg = 0.0, st_lo = 0.0, st_hi = 0.0;
            if (!newest.empty()) {
                plate.updateKinematics(x);
                st_lo = 1e9; st_hi = -1e9;
                for (const auto& c : newest) {
                    double g, t; creasePointState(plate, plate.folds[c.first], c.second, cfg.L_ref, g, t);
                    sg = std::max(sg, g); st_lo = std::min(st_lo, t); st_hi = std::max(st_hi, t);
                }
                printf(" >> start at the newest crease points: max gap %.3e L, fold angle %.3f .. %.3f deg\n",
                       sg, st_lo, st_hi);
            }
            const Eigen::VectorXd x_start = x;
            int iters = 0, substeps = 0, guard_retry = 0;
            StepState r;
            std::string why;
            const bool ok = fullStep(x, iters, substeps, guard_retry, r, why);

            if (!ok) {
                if (why.empty()) {
                    printf("\n!!! PROGRESSIVE CREASING STOPPED: no equilibrium at ell = %.6g (ell/L_c = %.4f), step %zu, line %d, vertices [%d, %d].\n",
                           ell, ell / Lc, k, s.line, s.a, s.b);
                    if (log) std::fprintf(log, "# STOPPED at step %zu, ell=%.10g, ell_over_Lc=%.10g, line=%d, a=%d, b=%d\n",
                                          k, ell, ell / Lc, s.line, s.a, s.b);
                } else {
                    printf("\n!!! PROGRESSIVE CREASING STOPPED by the guard at step %zu (ell/L_c = %.4f):%s\n",
                           k, ell / Lc, why.c_str());
                    if (log) std::fprintf(log, "# GUARD STOP at step %zu, ell_over_Lc=%.10g:%s theta_max=%.6g, change_over_L=%.6g\n",
                                          k, ell / Lc, why.c_str(), r.th_max, r.change);
                }
                x = x_start;
                break;
            }
            for (int f : all_folds) prev_active[f] = plate.folds[f].gp_active;
            prev_edge_ends = edge_ends;
            prev_edge_line = s.line;

            // --- 4. report ---
            const double E_mem = r.E_mem, E_ben = r.E_ben, E_hin = r.E_hin;
            const double th_mean = r.th_mean, th_max = r.th_max, change = r.change;
            const int neg = r.neg;

            printf(" >> step %zu: iters %d (guard retry %d, sub-steps %d) | E_mem %.4e E_ben %.4e E_hin %.4e | theta mean %.3f max %.3f deg"
                   " | neg modes %d | jump %.3e L at (%.4f, %.4f) | step change %.3e L\n",
                   k, iters, guard_retry, substeps, E_mem, E_ben, E_hin, th_mean, th_max, neg, jump, jump_at.x(), jump_at.y(), change);
            if (multi) {
                printf("    per crease (tag: mean / max deg):");
                for (size_t q = 0; q < all_folds.size(); ++q)
                    printf("  %d: %.3f / %.3f", plate.folds[all_folds[q]].crease_tag, r.f_mean[q], r.f_max[q]);
                printf("\n");
            }
            if (log) {
                std::fprintf(log, "%zu,%d,%d,%.10g,%.10g,%.10g,%d,%d,%.10e,%.10e,%.10e,%.10g,%.10g,%d,%.10e,%.10g,%.10g,%.10e,%.10e,%.6g,%.10e,%.10g,%.10g,%d,%d",
                             k, s.a, s.b, ell, ell / Lc, s.frac, iters, substeps, E_mem, E_ben, E_hin,
                             th_mean, th_max, neg, jump, jump_at.x(), jump_at.y(), change, jump_refit, refit_s,
                             sg, st_lo, st_hi, guard_retry, s.line);
                if (multi)
                    for (size_t q = 0; q < all_folds.size(); ++q) std::fprintf(log, ",%.10g,%.10g", r.f_mean[q], r.f_max[q]);
                std::fprintf(log, "\n");
                std::fflush(log);
            }
            prev_E_tot = E_mem + E_ben + E_hin;

            std::vector<double> angles(plate.folds.size(), 0.0);
            std::vector<int> drivers(plate.folds.size(), 0);
            for (size_t i = 0; i < plate.folds.size(); ++i) {
                angles[i] = solver.ComputeFoldAngleDeg(plate, i);
                drivers[i] = plate.folds[i].is_driver ? 1 : 0;
            }
            if (on_converged) on_converged(static_cast<int>(k), x, angles, drivers, E_mem, E_ben, E_hin);
        }
        if (log) std::fclose(log);
    }
}
