#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <dmumps_c.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

// =========================================================
// PERSISTENT SYMMETRIC MUMPS KKT SOLVER
// =========================================================
// The symbolic analysis (fill-reducing ordering) depends only on the sparsity
// pattern, which is fixed by the meshless topology and the active constraint
// set. Re-running it on every Newton iteration - as a fresh -1 / 6 / -2 cycle
// does - throws that work away for nothing. This wrapper keeps one MUMPS
// instance alive, re-analyzing only when the pattern actually changes, and
// otherwise reusing the symbolic structure for factorization (job=2) and
// solve (job=3).
class MumpsSymmetricSolver {
public:
    MumpsSymmetricSolver() = default;
    ~MumpsSymmetricSolver() { destroy(); }

    MumpsSymmetricSolver(const MumpsSymmetricSolver&) = delete;
    MumpsSymmetricSolver& operator=(const MumpsSymmetricSolver&) = delete;

    // Solves KKT * X = B in place. B is overwritten with X.
    bool solve(const Eigen::SparseMatrix<double>& KKT, Eigen::VectorXd& rhs_sol) {
        return solveRaw(KKT, rhs_sol.data(), 1) && rhs_sol.allFinite();
    }

    // Multi-RHS form; Eigen's column-major storage matches MUMPS' lrhs layout.
    bool solve(const Eigen::SparseMatrix<double>& KKT, Eigen::MatrixXd& rhs_sol) {
        return solveRaw(KKT, rhs_sol.data(), static_cast<int>(rhs_sol.cols()))
        && rhs_sol.allFinite();
    }

    // Block form: solve the saddle-point system [K J^T; J 0] without ever
    // materialising it. MUMPS (sym=2) reads only the upper triangle, which is
    // exactly upper(K) plus the whole J^T block - the J block lies strictly below
    // the diagonal and was always discarded. Building the full KKT cost a triplet
    // vector of nnz(K)+2*nnz(J) at 16 B each, a setFromTriplets, and (when
    // symmetrising) a materialised transpose and sum, every Newton iteration.
    bool solve(const Eigen::SparseMatrix<double>& K,
               const Eigen::SparseMatrix<double>& J, Eigen::VectorXd& rhs_sol) {
        return solveRawBlocks(K, J, rhs_sol.data(), 1) && rhs_sol.allFinite();
    }

    bool solve(const Eigen::SparseMatrix<double>& K,
               const Eigen::SparseMatrix<double>& J, Eigen::MatrixXd& rhs_sol) {
        return solveRawBlocks(K, J, rhs_sol.data(), static_cast<int>(rhs_sol.cols()))
        && rhs_sol.allFinite();
    }

    int  lastError()     const { return last_error_; }
    // INFOG(12): number of negative pivots from the sym=2 factorization. Free -
    // it is a byproduct of the factorization we already perform.
    int  negativePivots() const { return neg_pivots_; }
    long analysisCount() const { return analysis_count_; }
    long solveCount()    const { return solve_count_; }

    void destroy() {
        if (initialized_) {
            id_.job = -2;
            dmumps_c(&id_);
            initialized_ = false;
        }
        irn_.clear();
        jcn_.clear();
        analyzed_n_ = -1;
    }

private:
    // INFO(1) codes meaning "internal workspace too small"; the documented
    // remedy is to raise ICNTL(14) and repeat the factorization.
    static bool isWorkspaceError(int info1) {
        return info1 == -8  || info1 == -9  || info1 == -11
        || info1 == -12 || info1 == -14 || info1 == -15
        || info1 == -20;
    }

    // Rebuilds the 1-based upper-triangle coordinate arrays. Returns true when
    // the pattern differs from the one the current analysis was built for.
    bool refreshMatrix(const Eigen::SparseMatrix<double>& KKT) {
        irn_new_.clear();
        jcn_new_.clear();
        a_.clear();

        for (int k = 0; k < KKT.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(KKT, k); it; ++it) {
                if (it.row() <= it.col()) {
                    irn_new_.push_back(static_cast<int>(it.row()) + 1);
                    jcn_new_.push_back(static_cast<int>(it.col()) + 1);
                    a_.push_back(it.value());
                }
            }
        }

        if (irn_new_ == irn_ && jcn_new_ == jcn_) {
            return false; // Same pattern: the existing symbolic analysis still applies.
        }

        // swap (not assign) so both buffers keep their capacity across iterations
        irn_.swap(irn_new_);
        jcn_.swap(jcn_new_);
        return true;
    }

    // Emits the same coordinate arrays the old path produced, in the same order:
    // iterating KKT columns and keeping row <= col yields upper(K) column by column,
    // then the J^T block column by column. J is transposed first (it is far smaller
    // than K) so its entries come out with ascending row index within each column,
    // matching what a CSC scan of the assembled KKT gave.
    bool refreshFromBlocks(const Eigen::SparseMatrix<double>& K,
                           const Eigen::SparseMatrix<double>& J) {
        const int ns = static_cast<int>(K.rows());

        irn_new_.clear(); jcn_new_.clear(); a_.clear();

        for (int c = 0; c < K.outerSize(); ++c) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(K, c); it; ++it) {
                if (it.row() <= c) {
                    irn_new_.push_back(static_cast<int>(it.row()) + 1);
                    jcn_new_.push_back(c + 1);
                    a_.push_back(it.value());
                }
            }
        }

        Jt_ = J.transpose();
        Jt_.makeCompressed();
        for (int jr = 0; jr < Jt_.outerSize(); ++jr) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Jt_, jr); it; ++it) {
                irn_new_.push_back(static_cast<int>(it.row()) + 1);
                jcn_new_.push_back(ns + jr + 1);
                a_.push_back(it.value());
            }
        }

        if (irn_new_ == irn_ && jcn_new_ == jcn_) return false;
        irn_.swap(irn_new_);
        jcn_.swap(jcn_new_);
        return true;
    }

    bool solveRawBlocks(const Eigen::SparseMatrix<double>& K,
                        const Eigen::SparseMatrix<double>& J, double* rhs, int nrhs) {
        const int n = static_cast<int>(K.rows()) + static_cast<int>(J.rows());
        return runMumps(n, refreshFromBlocks(K, J), rhs, nrhs);
    }

    bool solveRaw(const Eigen::SparseMatrix<double>& KKT, double* rhs, int nrhs) {
        const int n = static_cast<int>(KKT.rows());
        return runMumps(n, refreshMatrix(KKT), rhs, nrhs);
    }

    bool runMumps(int n, bool pattern_changed, double* rhs, int nrhs) {
        last_error_ = 0;

        if (!initialized_) {
            id_.comm_fortran = -4; // Dummy MPI (sequential mode)
            id_.par = 1;           // Required by the sequential library
            id_.sym = 2;           // Symmetric indefinite
            id_.job = -1;
            dmumps_c(&id_);
            if (id_.info[0] < 0) {
                last_error_ = id_.info[0];
                return false;
            }
            initialized_ = true;

            // Suppress verbose Fortran terminal logs
            id_.icntl[0] = -1; id_.icntl[1] = -1; id_.icntl[2] = -1; id_.icntl[3] = 0;
        }

        id_.n   = n;
        id_.nz  = static_cast<int>(a_.size());
        id_.irn = irn_.data();
        id_.jcn = jcn_.data();
        id_.a   = a_.data();

        if (pattern_changed || n != analyzed_n_) {
            id_.job = 1; // Symbolic analysis
            dmumps_c(&id_);
            if (id_.info[0] < 0) {
                last_error_ = id_.info[0];
                destroy();
                return false;
            }
            analyzed_n_ = n;
            ++analysis_count_;
            if (analysis_count_ > 1) {
                printf("   [MUMPS] KKT sparsity pattern changed -> re-running symbolic analysis (#%ld).\n",
                       analysis_count_);
            }
        }

        id_.job = 2; // Numerical factorization
        dmumps_c(&id_);

        for (int retry = 0; retry < 3 && isWorkspaceError(id_.info[0]); ++retry) {
            const int old_icntl14 = id_.icntl[13];
            id_.icntl[13] = std::min(std::max(2 * old_icntl14, old_icntl14 + 20), 200);
            printf("   [MUMPS] Workspace too small (INFO(1)=%d) -> ICNTL(14) %d -> %d, refactorizing.\n",
                   id_.info[0], old_icntl14, id_.icntl[13]);
            id_.job = 2;
            dmumps_c(&id_);
        }

        if (id_.info[0] < 0) {
            last_error_ = id_.info[0];
            destroy();
            return false;
        }
        neg_pivots_ = id_.infog[11];   // INFOG(12), 0-based in the C interface

        // nrhs/lrhs must be reset on every solve: the instance is reused and
        // would otherwise inherit the previous call's block width.
        id_.nrhs = nrhs;
        id_.lrhs = n;
        id_.rhs  = rhs;
        id_.job  = 3; // Solve
        dmumps_c(&id_);

        if (id_.info[0] < 0) {
            last_error_ = id_.info[0];
            destroy();
            return false;
        }

        ++solve_count_;
        return true;
    }

    DMUMPS_STRUC_C id_{};
    bool initialized_ = false;
    int  analyzed_n_  = -1;
    int  last_error_  = 0;
    int  neg_pivots_  = 0;
    long analysis_count_ = 0;
    long solve_count_    = 0;

    Eigen::SparseMatrix<double> Jt_;   // scratch for the block path
    std::vector<int>    irn_,     jcn_;
    std::vector<int>    irn_new_, jcn_new_;
    std::vector<double> a_;
};
