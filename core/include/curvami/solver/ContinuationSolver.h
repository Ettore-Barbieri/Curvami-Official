#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "curvami/RestartManager.h"
#include "curvami/solver/MumpsSymmetricSolver.h"
#include "curvami/solver/ConstraintProjector.h"
#include "curvami/solver/StepController.h"
#include "curvami/solver/Problem.h"

// =========================================================
// GENERIC CONSTRAINED CONTINUATION SOLVERS
// =========================================================
// Newton with backtracking (LineSearch) and arc-length continuation (ArcLength)
// for any Problem: F_int(x) - lambda * F_ext(x) = 0 with c(x) = 0, the
// constraints enforced through a KKT system solved by symmetric MUMPS.
// Constraint rows are filtered to an independent set once per solve
// (InitializeTopology) and reused (filterRows).
class ContinuationSolver : public StepController {
public:
    int kmax = 5000;             // continuation step cap (ArcLength)

    // Newton-iteration cap for LineSearch. It used to be kmax, so shortening a
    // continuation by lowering kmax also truncated every LineSearch, which then
    // rolled back to its start state and silently corrupted what followed.
    int ls_max_iters = 500;
    std::string restart_file = "";
    
    double psi = 1.0;
    double div = 1e7;
    
    double tol_force  = 1e-8;
    double tol_energy = 1e-8;
    double tol_geom   = 1e-12;
    
    std::vector<int> ind_rows;

    // Newton iterations taken by the last LineSearch call (diagnostic only):
    // last_ls_iters on convergence (-1 otherwise), last_ls_attempts always,
    // including a call that failed.
    int last_ls_iters = 0;
    int last_ls_attempts = 0;

    // What LineSearch does when all six halvings fail to reduce the residual:
    // false (default, the original behaviour) accepts the last, worse trial;
    // true rolls back and reports failure, so the caller can react. Accepting a
    // worse trial let a Newton path wander to an unphysical state once
    // (progressive creasing, 29 Sep).
    bool ls_fail_on_no_descent = false;

    // One persistent MUMPS instance shared by every solve in this run, so the
    // symbolic analysis is paid once instead of once per Newton iteration.
    MumpsSymmetricSolver kkt_solver;

    // Scratch for Problem::loadStiffness, kept so its storage is reused.
    Eigen::SparseMatrix<double> K_load_;


    // Reduced-Hessian inertia from the last MUMPS factorization, for the step
    // controller: INFOG(12) counts negative pivots of the KKT, nc of which belong
    // to the constraint block.
    void confirmInertiaOnly(int n_const) {
        recordInertia(kkt_solver.negativePivots() - n_const);
    }

    bool confirmReversal(int n_const, const char* legacy_msg, const char* label) {
        return confirmReversalByInertia(kkt_solver.negativePivots() - n_const, legacy_msg, label);
    }

    // K <- K - lambda * K_load when the problem supplies a load stiffness (see
    // Problem::loadStiffness). When it does not, K is left untouched, so problems
    // without one follow exactly the same arithmetic as before.
    void applyLoadStiffness(Problem& prob, Eigen::SparseMatrix<double>& K,
                            const Eigen::VectorXd& x, double lambda) {
        if (!prob.loadStiffness(K_load_, x)) return;
        if (K_load_.rows() != K.rows() || K_load_.cols() != K.cols()) {
            throw std::invalid_argument("loadStiffness: K_load is "
                                        + std::to_string(K_load_.rows()) + "x" + std::to_string(K_load_.cols())
                                        + ", K is " + std::to_string(K.rows()) + "x" + std::to_string(K.cols()));
        }
        K = K - lambda * K_load_;
    }

    static void checkSize(const Problem& prob, int ns, const char* who) {
        if (ns != prob.numDofs()) {
            throw std::invalid_argument(std::string(who) + ": state vector has "
                                        + std::to_string(ns) + " entries, problem has "
                                        + std::to_string(prob.numDofs()) + " DOFs");
        }
    }

    // =========================================================
    // TOPOLOGY INITIALIZATION
    // =========================================================
    void InitializeTopology(const Eigen::SparseMatrix<double>& J) {
        ind_rows.clear();
        
        if (J.rows() == 0 || J.cols() == 0) {
            return;
        }
        
        Eigen::MatrixXd J_dense = Eigen::MatrixXd(J);
        Eigen::MatrixXd J_trans = J_dense.transpose();
        
        if (!J_trans.allFinite()) {
            throw std::runtime_error("Constraint Jacobian contains NaN or Inf.");
        }
        
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(J_trans);
        Eigen::VectorXd diag_R = qr.matrixQR().diagonal().array().abs();
        
        const double max_diag = (diag_R.size() > 0) ? diag_R.maxCoeff() : 0.0;
        const double tol =
        std::max(J.rows(), J.cols())
        * std::numeric_limits<double>::epsilon()
        * max_diag;
        
        const auto& perm = qr.colsPermutation().indices();
        
        for (int i = 0; i < diag_R.size(); ++i) {
            if (diag_R(i) > tol) {
                ind_rows.push_back(perm(i));
            }
        }
        
        std::sort(ind_rows.begin(), ind_rows.end());
    }
    
    Eigen::SparseMatrix<double> filterRows(const Eigen::SparseMatrix<double>& J) const {
        Eigen::SparseMatrix<double> J_ind(static_cast<int>(ind_rows.size()), J.cols());
        
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<size_t>(J.nonZeros()));
        std::vector<int> row_map(J.rows(), -1);
        
        for (int i = 0; i < static_cast<int>(ind_rows.size()); ++i) {
            row_map[ind_rows[i]] = i;
        }
        
        for (int k = 0; k < J.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(J, k); it; ++it) {
                const int mapped_row = row_map[it.row()];
                if (mapped_row >= 0) {
                    trips.emplace_back(mapped_row, static_cast<int>(it.col()), it.value());
                }
            }
        }
        
        J_ind.setFromTriplets(trips.begin(), trips.end());
        J_ind.makeCompressed();
        
        return J_ind;
    }
    
    // =========================================================
    // GEOMETRIC CORRECTOR AND RESIDUAL PROJECTION
    // =========================================================
    // Convenience wrappers for single use of a Jacobian. When both the corrector
    // and the projection are needed for the same J_ind, build one
    // ConstraintProjector instead so (J J^T) is factorized only once.
    Eigen::VectorXd ComputeGeometricCorrector(const Eigen::SparseMatrix<double>& J_ind,
                                              const Eigen::VectorXd& c_ind,
                                              int ns) const {
        return ConstraintProjector(J_ind).corrector(c_ind, ns);
    }

    Eigen::VectorXd GetProjectedResidual(const Eigen::SparseMatrix<double>& J_ind,
                                         const Eigen::VectorXd& R) const {
        return ConstraintProjector(J_ind).project(R);
    }

    // =========================================================
        // NEWTON SOLVER WITH BACKTRACKING
        // =========================================================
        bool LineSearch(Problem& prob,
                        double lambda,
                        Eigen::VectorXd& d) {
            const int ns = static_cast<int>(d.size());
            checkSize(prob, ns, "LineSearch");
            Eigen::VectorXd d0 = d;
            last_ls_iters = -1;   // stays -1 unless the solve converges
            
            Eigen::SparseMatrix<double> J_all;
            Eigen::VectorXd c_all;
            
            prob.constraints(d, c_all, J_all);
            InitializeTopology(J_all);
            
            const int nc = static_cast<int>(ind_rows.size());
            
            printf("\nIter      Max Force (inf)      Energy Chg           Geom. Error\n");
            printf("-------------------------------------------------------------------------\n");
            
            double energy_change = 1.0;
            
            for (int j = 1; j <= ls_max_iters; ++j) {
                last_ls_attempts = j;
                prob.constraints(d, c_all, J_all);
                
                Eigen::SparseMatrix<double> J_ind = filterRows(J_all);
                
                Eigen::VectorXd c_ind(static_cast<int>(ind_rows.size()));
                for (int i = 0; i < static_cast<int>(ind_rows.size()); ++i) {
                    c_ind(i) = c_all(ind_rows[i]);
                }
                
                Eigen::VectorXd Fint = prob.internalForce(d);
                Eigen::VectorXd Fext = prob.externalForce(d);
                Eigen::VectorXd R_vec = Fint - lambda * Fext;

                // One (J J^T) factorization serves both the projection below and
                // the geometric corrector further down.
                ConstraintProjector proj(J_ind);
                Eigen::VectorXd R_proj_current = proj.project(R_vec);

                const double max_force = R_proj_current.lpNorm<Eigen::Infinity>();
                const double geom_error = c_ind.lpNorm<Eigen::Infinity>();
                
                if (j > 1
                    && max_force <= tol_force
                    && energy_change <= tol_energy
                    && geom_error <= tol_geom) {
                    printf(">> Convergence! Max Force = %.2e | Energy Chg = %.2e | Geom Err = %.2e\n",
                           max_force, energy_change, geom_error);
                    last_ls_iters = j;
                    return true;
                }
                
                if (max_force >= div || std::isnan(max_force)) {
                    printf(">> Divergence detected. Rolling back.\n");
                    d = d0;
                    return false;
                }
                
                Eigen::SparseMatrix<double> K;
                Eigen::VectorXd F_int_dummy;
                
                prob.assemble(K, F_int_dummy, d);
                applyLoadStiffness(prob, K, d, lambda);

                Eigen::VectorXd xb = proj.corrector(c_ind, ns);

                Eigen::VectorXd RHS = Eigen::VectorXd::Zero(ns + nc);
                RHS.head(ns) = -(R_vec + K.selfadjointView<Eigen::Upper>() * xb);


                Eigen::VectorXd sol = RHS; // MUMPS overwrites this with the solution

                if (!kkt_solver.solve(K, J_ind, sol)) {
                    printf(">> KKT solve failed (MUMPS INFO(1) = %d). Rolling back.\n",
                           kkt_solver.lastError());
                    d = d0;
                    return false;
                }

                Eigen::VectorXd dx_tang = sol.head(ns);
                
                const double current_merit = max_force;
                double alpha = 1.0;
                double trial_merit = max_force;
                
                Eigen::VectorXd d_trial = d;
                bool descended = false;
                
                for (int iarm = 1; iarm <= 6; ++iarm) {
                    d_trial = d + xb + alpha * dx_tang;
                    
                    Eigen::VectorXd R_trial =
                    prob.internalForce(d_trial)
                    - lambda * prob.externalForce(d_trial);
                    
                    prob.constraints(d_trial, c_all, J_all);
                    Eigen::SparseMatrix<double> J_ind_trial = filterRows(J_all);
                    
                    trial_merit = GetProjectedResidual(J_ind_trial, R_trial).lpNorm<Eigen::Infinity>();
                    
                    if (trial_merit <= 1.05 * current_merit) {
                        descended = true;
                        if (iarm > 1) {
                            printf("   -> Backtracking Success at alpha = %.3f (Trial: %.2e <= Current: %.2e)\n",
                                   alpha, trial_merit, current_merit);
                        }
                        break;
                    }
                    
                    if (iarm == 1) {
                        printf("   -> Backtracking Initiated (Trial: %.2e > Current: %.2e)\n",
                               trial_merit, current_merit);
                    }
                    
                    alpha *= 0.5;
                }

                if (!descended && ls_fail_on_no_descent) {
                    printf(">> No descent after 6 halvings (last trial %.2e > current %.2e);"
                           " failing (ls_on_fail = fail). Rolling back.\n", trial_merit, current_merit);
                    d = d0;
                    return false;
                }
                
                d = d_trial;
                energy_change = std::abs((xb + alpha * dx_tang).dot(R_vec));
                
                printf("%3d      %12.5e         %12.5e         %12.5e\n",
                       j, trial_merit, energy_change, geom_error);
            }
            
            printf(">> LineSearch reached iteration limit (ls_max_iters = %d). Rolling back to initial state.\n",
                   ls_max_iters);
            d = d0;
            return false;
        }
    
    
    // =========================================================
        // STANDARD ARC-LENGTH CONTINUATION
        // =========================================================
        // on_converged(step, lambda, x, W_ext) runs after every accepted step.
        void ArcLength(Problem& prob,
                       double lambda_0,
                       Eigen::VectorXd x0,
                       double lambda_tot,
                       std::function<void(int, double, const Eigen::VectorXd&, double)> on_converged = nullptr) {
            
            const int jmax = max_iters;
            const int ns = static_cast<int>(x0.size());
            checkSize(prob, ns, "ArcLength");
            
            Eigen::VectorXd x = x0;
            double lambda = lambda_0;
            int k = 1;
            
            Eigen::VectorXd Delta_U_old = Eigen::VectorXd::Zero(ns);
            double Delta_lambda_old = 0.0;
            
            Eigen::SparseMatrix<double> J_all;
            Eigen::VectorXd c_all;
            
            prob.constraints(x, c_all, J_all);
            InitializeTopology(J_all);
            resetInertiaTracking();
            
            const int n_const = static_cast<int>(ind_rows.size());
            
            double W_ext = 0.0;
            Eigen::VectorXd F_ref = prob.externalForce(x0);
            Eigen::VectorXd x_prev_step = x;
            
            if (!restart_file.empty()) {
                if (RestartManager::load(restart_file, k, lambda, ds, Delta_lambda_old, W_ext, x, Delta_U_old, ns)) {
                    printf("\n=================================================================================\n");
                    printf(">>> RESTART SUCCESSFUL: Resuming Arc-Length from Step %d (Lambda = %.4f) <<<\n", k, lambda);
                    printf("=================================================================================\n");
                    // Work increments are measured from the previous step. After a
                    // restart that is the loaded state, not x0 (which is zero in
                    // restart mode): W_ext used to carry a constant offset.
                    x_prev_step = x;
                } else {
                    printf("\n>>> No restart file found at %s. Starting from scratch. <<<\n", restart_file.c_str());
                }
            }
            
            while (k <= kmax && lambda <= lambda_tot) {
                double current_disp = prob.monitor(x);
                
                printf("\n=================================================================================\n");
                printf(" LOAD STEP %3d | Lambda: %10.5e | Disp: %10.5e | ds = %10.5e | psi = %.2f\n",
                       k, lambda, current_disp, ds, psi);
                printf("=================================================================================\n");
                
                Eigen::VectorXd Fext = prob.externalForce(x);
                
                Eigen::SparseMatrix<double> K_mat;
                Eigen::VectorXd Fint_dummy;
                
                prob.assemble(K_mat, Fint_dummy, x);
                applyLoadStiffness(prob, K_mat, x, lambda);
                prob.constraints(x, c_all, J_all);
                
                Eigen::SparseMatrix<double> J_ind = filterRows(J_all);
                
                Eigen::VectorXd c_ind(n_const);
                for (int i = 0; i < n_const; ++i) {
                    c_ind(i) = c_all(ind_rows[i]);
                }
                
                Eigen::VectorXd xb_p = ComputeGeometricCorrector(J_ind, c_ind, ns);
                

                const int n_kkt = static_cast<int>(K_mat.rows()) + static_cast<int>(J_ind.rows());

                // --- PREDICTOR SOLVE ---
                Eigen::VectorXd sol_T = Eigen::VectorXd::Zero(n_kkt);
                sol_T.head(ns) = Fext;

                if (!kkt_solver.solve(K_mat, J_ind, sol_T)) {
                    printf(">> Predictor KKT solve failed (MUMPS INFO(1) = %d).\n",
                           kkt_solver.lastError());
                    break;
                }

                Eigen::VectorXd d_T = sol_T.head(ns);

                const double ff_norm = Fext.dot(Fext);
                double sgn = 1.0;
                
                if (k == 1) {
                    const double s = d_T.dot(Fext);
                    sgn = (s >= 0.0) ? 1.0 : -1.0;
                } else {
                    if (Delta_U_old.dot(d_T) >= 0.0) {
                        sgn = 1.0;
                        confirmInertiaOnly(n_const);
                    } else if (confirmReversal(n_const, " >>> SNAP-BACK DETECTED: Reversing Load Increment <<<\n", "SNAP-BACK DETECTED")) {
                        sgn = -1.0;
                        noteReversal();
                    } else {
                        sgn = 1.0;
                    }
                }
                
                double Delta_lambda = sgn * ds / std::sqrt(d_T.dot(d_T) + psi * psi * ff_norm);
                Eigen::VectorXd Delta_U = Delta_lambda * d_T + xb_p;
                
                int icorr_ok = 0;
                int k_corr = jmax;
                
                printf("Iter      Max Force (inf)      Energy Chg           Geom. Error          Arc-Length Err\n");
                printf("------------------------------------------------------------------------------------------\n");
                
                // --- CORRECTOR ITERATIONS ---
                for (int j = 1; j <= jmax; ++j) {
                    const double curr_lambda = lambda + Delta_lambda;
                    const Eigen::VectorXd x_curr = x + Delta_U;
                    
                    prob.constraints(x_curr, c_all, J_all);
                    Eigen::SparseMatrix<double> J_ind_c = filterRows(J_all);
                    Eigen::VectorXd c_ind_c(n_const);
                    for (int i = 0; i < n_const; ++i) c_ind_c(i) = c_all(ind_rows[i]);
                    
                    Eigen::VectorXd Fext_c = prob.externalForce(x_curr);
                    
                    Eigen::SparseMatrix<double> K_c;
                    Eigen::VectorXd Fint_c;
                    prob.assemble(K_c, Fint_c, x_curr);
                    applyLoadStiffness(prob, K_c, x_curr, curr_lambda);
                    Eigen::VectorXd R_vec = Fint_c - curr_lambda * Fext_c;
                    
                    const double v_res = Delta_U.dot(Delta_U) + psi * psi * Delta_lambda * Delta_lambda * ff_norm - ds * ds;
                    
                    Eigen::VectorXd xb_c = ComputeGeometricCorrector(J_ind_c, c_ind_c, ns);
                    

                    const int n_kkt_c = static_cast<int>(K_c.rows()) + static_cast<int>(J_ind_c.rows());

                    Eigen::MatrixXd sol_c = Eigen::MatrixXd::Zero(n_kkt_c, 2);
                    sol_c.col(0).head(ns) = Fext_c;
                    sol_c.col(1).head(ns) = -(R_vec + K_c.selfadjointView<Eigen::Upper>() * xb_c);

                    if (!kkt_solver.solve(K_c, J_ind_c, sol_c)) {
                        printf(">> Corrector KKT solve failed (MUMPS INFO(1) = %d).\n",
                               kkt_solver.lastError());
                        icorr_ok = 0; k_corr = j; break;
                    }

                    Eigen::VectorXd daP = sol_c.col(0).head(ns);
                    Eigen::VectorXd daPsi = sol_c.col(1).head(ns) + xb_c;
                    Eigen::VectorXd lambda_mult = sol_c.col(1).tail(n_const);

                    const double arc_denom = Delta_U.dot(daP) + psi * psi * Delta_lambda * ff_norm;
                    const double arc_numer = 0.5 * v_res + Delta_U.dot(daPsi);
                    const double dl_c = -arc_numer / arc_denom;
                    
                    Delta_lambda += dl_c;
                    Delta_U += daPsi + dl_c * daP;
                    
                    const double max_force = (R_vec + J_ind_c.transpose() * lambda_mult).lpNorm<Eigen::Infinity>();
                    const double energy_change = std::abs(daPsi.dot(R_vec));
                    const double geom_error = c_ind_c.lpNorm<Eigen::Infinity>();
                    const double al_error = std::abs(v_res);
                    
                    if (j > 1 && max_force <= tol_force && energy_change <= tol_energy && geom_error <= tol_geom && al_error <= tol_geom) {
                        printf(">> Convergence! Max Force = %.2e | Energy = %.2e | Geom = %.2e\n", max_force, energy_change, geom_error);
                        icorr_ok = 1; k_corr = j; break;
                    }
                    if (max_force >= div || std::isnan(max_force)) {
                        printf(">> Divergence detected.\n");
                        icorr_ok = 0; k_corr = j; break;
                    }
                    printf("%3d      %12.5e         %12.5e         %12.5e         %12.5e\n", j, max_force, energy_change, geom_error, al_error);
                }
                
                double current_cos_theta = 1.0;
                if (k > 1 && icorr_ok == 1) {
                    const double old_load = psi * Delta_lambda_old;
                    const double new_load = psi * Delta_lambda;
                    const double old_norm = std::sqrt(Delta_U_old.squaredNorm() + old_load * old_load);
                    const double new_norm = std::sqrt(Delta_U.squaredNorm() + new_load * new_load);
                    if (old_norm > 0.0 && new_norm > 0.0) {
                        current_cos_theta = (Delta_U_old.dot(Delta_U) + old_load * new_load) / (old_norm * new_norm);
                        current_cos_theta = std::max(-1.0, std::min(1.0, current_cos_theta));
                    }
                }
                
                ASLA(icorr_ok, k_corr, current_cos_theta);
                
                if (icorr_ok == 1) {
                    ApplyMatlabCurvatureController(k, current_cos_theta);
                    noteAcceptedStep();
                    Delta_U_old = Delta_U; Delta_lambda_old = Delta_lambda; x += Delta_U; lambda += Delta_lambda;
                    
                    double avg_lambda = 0.5 * ((lambda - Delta_lambda) + lambda);
                    double dW = avg_lambda * F_ref.dot(x - x_prev_step);
                    W_ext += dW; x_prev_step = x;
                    
                    if (on_converged) on_converged(k, lambda, x, W_ext);
                    ++k;
                    if (!restart_file.empty()) RestartManager::save(restart_file, k, lambda, ds, Delta_lambda_old, W_ext, x, Delta_U_old);
                    if (stalledAtDsmin(x, Delta_U)) {
                        printf("!!! No progress: ds at minimum and the state has not moved over %d"
                               " accepted steps. Terminating.\n", no_progress_window);
                        break;
                    }
                } else {
                    if (ds <= dsmin) { printf("!!! FATAL: ds reached minimum threshold. Terminating.\n"); break; }
                }
            }
        }
    
};
