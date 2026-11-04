#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Meshless.h"
#include "Kinematics.h"
#include "Constitutive.h"
#include "curvami/FRNN.h"
#include "curvami/GlobalRKPM.h"
#include "curvami/Enrichment.h"

// =========================================================
// STRESS ALONG RAYS FROM A CRACK TIP (post-processing on a converged state)
// =========================================================
// Coupled_Criterion_Plan.md (docs/), "Stress sampler"; Crack_Brief_for_Code.md, E9
// step 2. For the stress condition of the coupled criterion: the opening stress
// across a candidate path, before the crack jumps along it.
//
// Rays leave the tip at angles phi from the crack direction d (counter-clockwise),
// with points at r = dr, 2 dr, ... <= r_max. The points are not Gauss points of the
// run: their shape functions are built here with the run's own cuts, support and
// order. On a ray, T = (cos, sin) of its direction and N = e_z x T are the reference
// tangent and normal of the candidate path.
//
// At each point and each surface z = -h/2 (bottom) and +h/2 (top, the side of b):
//   E(z) = E0 + z E1_el            Green-Lagrange strain (E1_el: natural curvature removed)
//   S(z) = Q E(z)                  second Piola-Kirchhoff stress, plane stress
//   F(z) = F0 + z b_,alpha         3 x 2, with [b_,1 b_,2] = F0 C0^-1 E1 (geometric E1)
//   J(z) = sqrt(det F(z)^T F(z))   area ratio of that surface
//   sigma(z) = F S F^T / J         Cauchy stress (3 x 3, in the surface's tangent plane)
//   n = F C(z)^-1 N / |.|          current normal to the path in that plane (Nanson)
//   t = F T / |F T|                current tangent of the path
//   sig_nn = n . sigma n,  tau_nt = t . sigma n
// Also the mid-surface values N_nn / h and N_nt / h and the moment M_nn (reference
// components of the resultants along N and T), and S_nn at the two surfaces, for
// comparison. Q is A / h: the sheet is taken as uniform through the thickness.
//
// Config, one object or a list (several tips):
//   "stress_sampler": {"tip": [x, y], "direction": [dx, dy], "r_max": ..., "dr": ...,
//                      "angles_deg": [...] (optional: default -170 to 170 every 10),
//                      "steps": [...] (optional: default every exported step)}
// Output: one CSV per ray and step, <output>_<tag>_s<step>_p<phi>.csv (tag S, or S0,
// S1, ... for a list). A ray stops at the bounding box of the nodes. A point lying on
// a cut (a ray crossing a crease exactly at a sample) is not written: it has no side,
// and its kinematics would mix the two facets.
// It changes nothing in the solution.
namespace StressSampler {

    struct Settings {
        Eigen::Vector2d tip = Eigen::Vector2d::Zero();
        Eigen::Vector2d dir = Eigen::Vector2d::UnitX();
        double r_max = 0.0, dr = 0.0;
        std::vector<double> angles_deg;
        std::vector<int> steps;
        std::string stem;             // <output>_<tag>
    };

    struct SurfaceStress { double sig_nn, tau_nt, S_nn, J; };

    // Cauchy opening and shear stress across the path (reference normal N, tangent T)
    // at height z, from the kinematics at the mid-surface point.
    inline SurfaceStress atSurface(const Kinematics::LocalKinematics& kin, const Eigen::Matrix3d& Q, double z,
                                   const Eigen::Vector2d& N, const Eigen::Vector2d& T) {
        const Eigen::Matrix2d E = kin.E0 + z * kin.E1_el;
        const Eigen::Vector3d s = Q * Eigen::Vector3d(E(0, 0), E(1, 1), 2.0 * E(0, 1));
        Eigen::Matrix2d S;
        S << s(0), s(2), s(2), s(1);
        const Eigen::Matrix<double, 3, 2> F = kin.F0 + z * (kin.F0 * kin.C0_inv * kin.E1);
        const Eigen::Matrix2d C = F.transpose() * F;
        const double J = std::sqrt(C.determinant());
        const Eigen::Matrix3d sigma = F * S * F.transpose() / J;
        const Eigen::Vector3d n = (F * C.inverse() * N).normalized();
        const Eigen::Vector3d t = (F * T).normalized();
        return {n.dot(sigma * n), t.dot(sigma * n), N.dot(S * N), J};
    }

    // Distance from X to the polyline of a cut.
    inline double distanceToCut(const Eigen::Vector2d& X, const Enrichment::FoldLine& cut) {
        double best = std::numeric_limits<double>::infinity();
        const size_t n = cut.vertices.size();
        for (size_t i = 0; i + 1 < n || (cut.is_closed && i < n); ++i) {
            const Eigen::Vector2d A = cut.vertices[i], v = cut.vertices[(i + 1) % n] - A;
            const double v2 = v.squaredNorm();
            const double t = v2 > 0.0 ? std::max(0.0, std::min(1.0, (X - A).dot(v) / v2)) : 0.0;
            best = std::min(best, (X - (A + t * v)).norm());
        }
        return best;
    }

    inline void report(Meshless& plate, const Eigen::VectorXd& x, int step, const Settings& s,
                       const std::vector<Enrichment::FoldLine>& creases, const char* git_commit) {
        if (s.r_max <= 0.0 || s.dr <= 0.0) return;
        if (!s.steps.empty() && std::find(s.steps.begin(), s.steps.end(), step) == s.steps.end()) return;
        plate.updateKinematics(x);

        const double h = plate.material.total_thickness();
        const Eigen::Matrix3d Q = plate.material.A / h;
        if (plate.material.plies.size() > 1)
            printf("!!! WARNING: stress sampler: several plies; the surface stress uses Q = A / h.\n");
        Eigen::Vector2d lo = plate.nodes[0].X_ref, hi = lo;
        for (const auto& n : plate.nodes) { lo = lo.cwiseMin(n.X_ref); hi = hi.cwiseMax(n.X_ref); }

        std::vector<double> angles = s.angles_deg;
        if (angles.empty()) for (int a = -170; a <= 170; a += 10) angles.push_back(a);
        const double phi0 = std::atan2(s.dir.y(), s.dir.x());
        const int nr = static_cast<int>(std::floor(s.r_max / s.dr + 1e-9));

        printf("\n >> Stress sampler at step %d, tip (%.6g, %.6g), d = (%.4f, %.4f): %zu rays, r up to %.4g every %.4g\n",
               step, s.tip.x(), s.tip.y(), s.dir.x(), s.dir.y(), angles.size(), s.r_max, s.dr);
        GlobalFRNN frnn(plate.rho);
        for (double a : angles) {
            const double phi = phi0 + a * M_PI / 180.0;
            const Eigen::Vector2d T(std::cos(phi), std::sin(phi)), N(-T.y(), T.x());
            std::vector<GaussPointData> gps;
            std::vector<double> rr;
            int skipped = 0;
            for (int i = 1; i <= nr; ++i) {
                const Eigen::Vector2d X = s.tip + (i * s.dr) * T;
                if (X.x() < lo.x() || X.x() > hi.x() || X.y() < lo.y() || X.y() > hi.y()) break;
                bool on_cut = false;
                for (const auto& c : creases) on_cut = on_cut || distanceToCut(X, c) < 1e-9 * plate.rho;
                if (on_cut) { ++skipped; continue; }
                GaussPointData gp;
                gp.id = static_cast<int>(gps.size());
                gp.X_ref = X;
                gp.weight = 0.0;
                gps.push_back(gp);
                rr.push_back(i * s.dr);
            }
            if (gps.empty()) continue;
            try {
                std::vector<size_t> ig, js;
                std::vector<double> sv;
                frnn.execute(gps, plate.nodes, plate.rho, ig, js, sv);
                GlobalRKPM::solve(gps, plate.nodes, ig, js, sv, plate.rho, plate.order, creases);
            } catch (const std::exception& e) {
                printf("!!! WARNING: stress sampler: ray at %g deg skipped (%s)\n", a, e.what());
                continue;
            }

            char name[64];
            std::snprintf(name, sizeof(name), "_s%d_p%+04d.csv", step, static_cast<int>(std::lround(a)));
            const std::string path = s.stem + name;
            FILE* f = std::fopen(path.c_str(), "w");
            if (!f) { printf("!!! WARNING: could not write %s\n", path.c_str()); continue; }
            std::fprintf(f, "# stress along a ray from the tip, git_commit=%s, step=%d, tip_x=%.12g, tip_y=%.12g, "
                            "dir_x=%.12g, dir_y=%.12g, phi_deg=%.6g, h=%.12g, rho=%.12g, order=%d\n",
                         git_commit, step, s.tip.x(), s.tip.y(), s.dir.x(), s.dir.y(), a, h, plate.rho, plate.order);
            std::fprintf(f, "r,X,Y,sig_nn_top,sig_nn_bot,tau_nt_top,tau_nt_bot,sig_nn_max,Nnn_over_h,Nnt_over_h,Mnn,"
                            "S_nn_top,S_nn_bot,J_top,J_bot,n_neighbours\n");
            double peak = 0.0;
            for (size_t i = 0; i < gps.size(); ++i) {
                const auto kin = Kinematics::compute(gps[i], plate.nodes);
                Constitutive::Resultants res;
                Constitutive::evaluate(kin, plate.material, res);
                Eigen::Matrix2d Nr, Mr;
                Nr << res.N(0), res.N(2), res.N(2), res.N(1);
                Mr << res.M(0), res.M(2), res.M(2), res.M(1);
                const SurfaceStress top = atSurface(kin, Q, 0.5 * h, N, T), bot = atSurface(kin, Q, -0.5 * h, N, T);
                const double mx = std::max(top.sig_nn, bot.sig_nn);
                peak = std::max(peak, mx);
                std::fprintf(f, "%.10g,%.10g,%.10g,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.12g,%.12g,%zu\n",
                             rr[i], gps[i].X_ref.x(), gps[i].X_ref.y(), top.sig_nn, bot.sig_nn, top.tau_nt, bot.tau_nt, mx,
                             N.dot(Nr * N) / h, T.dot(Nr * N) / h, N.dot(Mr * N), top.S_nn, bot.S_nn, top.J, bot.J,
                             gps[i].neighbor_node_ids.size());
            }
            std::fclose(f);
            printf("    ray %+4d deg: %zu points to r = %.4g, largest sig_nn %.6e%s\n",
                   static_cast<int>(std::lround(a)), gps.size(), rr.back(), peak,
                   skipped ? " (points on a cut left out)" : "");
        }
    }
}
