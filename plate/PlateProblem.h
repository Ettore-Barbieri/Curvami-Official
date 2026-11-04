#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Meshless.h"
#include "curvami/solver/Problem.h"

// =========================================================
// MESHLESS DISPLACEMENT TRACKING
// =========================================================
// Mean displacement along the load direction over every loaded integration
// domain; the mean deflection w when nothing is loaded. 3 DOFs per node.
inline double trackingDisplacement(const Meshless& plate, const Eigen::VectorXd& x) {
    double u_track = 0.0;
    int count = 0;
    
    auto extract_disp = [&](const auto& dom) {
        if (dom.bc.force.norm() > 1e-12) {
            Eigen::Vector3d load_dir = dom.bc.force.normalized();
            for (const auto& gp : dom.gps) {
                Eigen::Vector3d u_gp = Eigen::Vector3d::Zero();
                for (size_t i = 0; i < gp.neighbor_node_ids.size(); ++i) {
                    int id = gp.neighbor_node_ids[i];
                    u_gp += gp.Psi[i] * x.segment<3>(id * 3);
                }
                u_track += u_gp.dot(load_dir);
                count++;
            }
        }
    };
    
    for (const auto& p : plate.points) extract_disp(p);
    for (const auto& e : plate.edges) extract_disp(e);
    for (const auto& f : plate.folds) { extract_disp(f.left); extract_disp(f.right); }
    
    if (count > 0) return u_track / static_cast<double>(count);
    
    double avg_z = 0.0;
    for (size_t i = 0; i < plate.nodes.size(); ++i) {
        avg_z += x(i * 3 + 2);
    }
    return avg_z / static_cast<double>(plate.nodes.size());
}

// =========================================================
// PLATE AS A GENERIC PROBLEM
// =========================================================
// Adapter for the core continuation solvers. flagRig (rigid folds enforced as
// constraints instead of elastic hinges) is chosen per solve, so an adapter is
// built per call rather than stored on the plate.
class PlateProblem : public Problem {
public:
    PlateProblem(Meshless& plate, bool flagRig) : plate_(plate), flagRig_(flagRig) {}

    int numDofs() const override { return static_cast<int>(plate_.nodes.size()) * 3; }

    void assemble(Eigen::SparseMatrix<double>& K, Eigen::VectorXd& F_int,
                  const Eigen::VectorXd& x) override {
        plate_.assembleSystem(K, F_int, x, flagRig_);
    }

    Eigen::VectorXd internalForce(const Eigen::VectorXd& x) override {
        return plate_.assembleSystemInternalOnly(x, flagRig_);
    }

    Eigen::VectorXd externalForce(const Eigen::VectorXd& x) override {
        return plate_.computeExternalForce(x);
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c,
                     Eigen::SparseMatrix<double>& J) override {
        plate_.evaluateConstraints(x, c, J, flagRig_);
    }

    // The plate's reference loads (point/edge/fold forces) do not depend on x.
    // The target-directed driver moments do, but they are applied inside
    // TargetDirectedArcLength, not through this adapter; giving them their load
    // stiffness is a separate behaviour change.
    bool loadStiffness(Eigen::SparseMatrix<double>& K_load, const Eigen::VectorXd& x) override {
        (void)K_load; (void)x;
        return false;
    }

    double monitor(const Eigen::VectorXd& x) const override {
        return trackingDisplacement(plate_, x);
    }

private:
    Meshless& plate_;
    bool flagRig_;
};
