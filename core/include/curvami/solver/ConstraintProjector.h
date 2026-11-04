#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>

// =========================================================
// CONSTRAINT PROJECTOR (shared dense factorization)
// =========================================================
// The geometric corrector and the residual projection both need (J J^T)^-1.
// Factorizing it once per iteration and reusing it for both halves the dense
// work, which is O(nc^2 * ns) for the Gram matrix alone.
class ConstraintProjector {
public:
    ConstraintProjector() = default;

    explicit ConstraintProjector(const Eigen::SparseMatrix<double>& J_ind) { reset(J_ind); }

    void reset(const Eigen::SparseMatrix<double>& J_ind) {
        J_ = Eigen::MatrixXd(J_ind);
        cod_ready_ = false;
        ldlt_ok_   = false;
        if (J_.rows() == 0) return;

        ldlt_.compute(J_ * J_.transpose());
        ldlt_ok_ = (ldlt_.info() == Eigen::Success);
    }

    // Minimum-norm step xb with J*xb = -c.
    Eigen::VectorXd corrector(const Eigen::VectorXd& c_ind, int ns) {
        if (c_ind.size() == 0) {
            return Eigen::VectorXd::Zero(ns);
        }
        if (ldlt_ok_) {
            Eigen::VectorXd mu  = ldlt_.solve(-c_ind);
            Eigen::VectorXd out = J_.transpose() * mu;
            if (out.allFinite()) return out;
        }
        // Robust fallback for nearly rank-deficient filtered constraints.
        return cod().solve(-c_ind);
    }

    // R projected onto the null space of J.
    Eigen::VectorXd project(const Eigen::VectorXd& R) {
        if (J_.rows() == 0) {
            return R;
        }
        if (ldlt_ok_) {
            Eigen::VectorXd mu  = ldlt_.solve(J_ * R);
            Eigen::VectorXd out = R - J_.transpose() * mu;
            if (out.allFinite()) return out;
        }
        // Robust fallback for nearly rank-deficient filtered constraints.
        return R - cod().solve(J_ * R);
    }

private:
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>& cod() {
        if (!cod_ready_) {
            cod_.compute(J_);
            cod_ready_ = true;
        }
        return cod_;
    }

    Eigen::MatrixXd J_;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_;
    bool ldlt_ok_   = false;
    bool cod_ready_ = false;
};
