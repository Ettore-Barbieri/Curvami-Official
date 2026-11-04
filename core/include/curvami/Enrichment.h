#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include "curvami/MeshfreeData.h"

namespace Enrichment {

    struct FoldLine {
        std::vector<Eigen::Vector2d> vertices;
        bool extend_start = false;
        bool extend_end = false;
        bool is_closed = false;
    };

    struct IndicatorData {
        double f; double d;
        Eigen::Vector2d grad_f; Eigen::Matrix2d hess_f;
        bool is_valid = false;
    };

    inline IndicatorData evaluateIndicator(const Eigen::Vector2d& P, const FoldLine& fold) {
        IndicatorData out;
        out.d = std::numeric_limits<double>::infinity();
        
        size_t N = fold.vertices.size();
        if (N < 2) return out;

        Eigen::Vector2d C_best;
        size_t idx_best = 0;
        double t_best = 0.0;
        Eigen::Vector2d seg_n_best;

        for (size_t i = 0; i < N - 1; ++i) {
            Eigen::Vector2d A = fold.vertices[i];
            Eigen::Vector2d B = fold.vertices[i+1];
            Eigen::Vector2d v = B - A;
            
            double v_sq = v.squaredNorm();
            if (v_sq == 0) continue;

            Eigen::Vector2d w = P - A;
            double t = w.dot(v) / v_sq;

            double t_min = (i == 0 && fold.extend_start) ? -std::numeric_limits<double>::infinity() : 0.0;
            double t_max = (i == N - 2 && fold.extend_end) ? std::numeric_limits<double>::infinity() : 1.0;

            double t_clamped = std::max(t_min, std::min(t_max, t));
            Eigen::Vector2d C = A + t_clamped * v;

            double current_dist_sq = (P - C).squaredNorm();
            if (current_dist_sq < out.d * out.d) {
                out.d = std::sqrt(current_dist_sq);
                C_best = C;
                idx_best = i;
                t_best = t_clamped;
                seg_n_best = Eigen::Vector2d(-v.y(), v.x()) / std::sqrt(v_sq);
            }
        }

        out.is_valid = (out.d > 1e-12);
        if (!out.is_valid) {
            out.f = 0.0; out.grad_f.setZero(); out.hess_f.setZero();
            return out;
        }

        Eigen::Vector2d u_hat = (P - C_best) / out.d;
        double dot_prod = u_hat.dot(seg_n_best);

        bool is_start_cap = (idx_best == 0 && t_best == 0.0 && !fold.extend_start && !fold.is_closed); // <-- ADD !fold.is_closed
        bool is_end_cap = (idx_best == N - 2 && t_best == 1.0 && !fold.extend_end && !fold.is_closed); // <-- ADD !fold.is_closed
        
        
        if (is_start_cap || is_end_cap) {
            out.f = dot_prod;
            out.grad_f = (seg_n_best - out.f * u_hat) / out.d;
            double d2 = out.d * out.d;
            double ux = u_hat.x(), uy = u_hat.y();
            double nx = seg_n_best.x(), ny = seg_n_best.y();
            
            out.hess_f(0,0) = (3 * out.f * ux * ux - 2 * nx * ux - out.f) / d2;
            out.hess_f(1,1) = (3 * out.f * uy * uy - 2 * ny * uy - out.f) / d2;
            double h12 = (3 * out.f * ux * uy - (nx * uy + ux * ny)) / d2;
            out.hess_f(0,1) = out.hess_f(1,0) = h12;
        } else {
            out.f = (dot_prod >= 0.0) ? 1.0 : -1.0;
            out.grad_f.setZero(); out.hess_f.setZero();
        }

        return out;
    }

}
