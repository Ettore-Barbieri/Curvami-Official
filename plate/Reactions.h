#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdio>
#include <string>
#include <vector>

#include "Meshless.h"

// =========================================================
// SUPPORT REACTIONS (post-processing on a converged state)
// =========================================================
// The solver does not keep the multipliers of the constraints. At equilibrium the
// nodal forces of the supports are  J^T mu = F_int - lambda F_ext  (plate and hinge
// internal forces; J the constraint Jacobian of Meshless::evaluateConstraints), so mu
// is recovered here by least squares, (J J^T) mu = J (F_int - lambda F_ext), with the
// minimum-norm solution where rows are dependent. A displacement row has no weight,
// so its mu is the force that support point exerts ON the plate along that component;
// the resultant of an edge is the sum over its points. A "rot" row gives the moment
// conjugate to the normal slope at that point. Written to <output>_reactions.csv when
// the config has "report_reactions": true (LineSearch only). It changes nothing in
// the solution.
namespace Reactions {

    inline void write(Meshless& plate, const Eigen::VectorXd& x, double lambda, bool flagRig,
                      const std::string& path, const char* git_commit) {
        Eigen::SparseMatrix<double> J;
        Eigen::VectorXd c;
        plate.evaluateConstraints(x, c, J, flagRig);
        if (J.rows() == 0) return;
        const Eigen::VectorXd r = plate.assembleSystemInternalOnly(x, flagRig) - lambda * plate.computeExternalForce(x);
        const Eigen::MatrixXd A = Eigen::MatrixXd(J * Eigen::SparseMatrix<double>(J.transpose()));
        const Eigen::VectorXd mu = A.completeOrthogonalDecomposition().solve(J * r);
        const double res = (Eigen::VectorXd(J.transpose() * mu) - r).lpNorm<Eigen::Infinity>();
        const double scale = r.lpNorm<Eigen::Infinity>();

        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) { printf("!!! WARNING: could not write %s\n", path.c_str()); return; }
        std::fprintf(f, "# support reactions (force of the support on the plate), git_commit=%s, lambda=%.12g, "
                        "residual_inf=%.6e, nodal_force_inf=%.6e, rows=%d\n",
                     git_commit, lambda, res, scale, static_cast<int>(J.rows()));
        std::fprintf(f, "kind,tag,component,n_points,resultant,X_first,Y_first\n");

        int row = 0;
        // rows in the order of Meshless::evaluateConstraints: points, edges, then folds
        auto domain = [&](const char* kind, const IntegrationDomain& dom) {
            const BoundaryCondition& bc = dom.bc;
            if (!(bc.active_u || bc.active_v || bc.active_w || bc.active_rot) || dom.gps.empty()) return;
            double sum[4] = {0.0, 0.0, 0.0, 0.0};
            for (size_t i = 0; i < dom.gps.size(); ++i) {
                if (bc.active_u)   sum[0] += mu(row++);
                if (bc.active_v)   sum[1] += mu(row++);
                if (bc.active_w)   sum[2] += mu(row++);
                if (bc.active_rot) sum[3] += mu(row++);
            }
            const char* names[4] = {"u", "v", "w", "rot"};
            const bool on[4] = {bc.active_u, bc.active_v, bc.active_w, bc.active_rot};
            for (int k = 0; k < 4; ++k)
                if (on[k])
                    std::fprintf(f, "%s,%d,%s,%zu,%.12e,%.10g,%.10g\n", kind, dom.gps[0].entity_tag, names[k],
                                 dom.gps.size(), sum[k], dom.gps[0].X_ref.x(), dom.gps[0].X_ref.y());
        };
        for (const auto& p : plate.points) domain("point", p);
        for (const auto& e : plate.edges)  domain("edge", e);
        std::fclose(f);
        printf("\n >> Support reactions written to %s (%d constraint rows, residual %.2e of %.2e)\n",
               path.c_str(), static_cast<int>(J.rows()), res, scale);
    }
}
