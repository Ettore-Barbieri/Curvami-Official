#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/UmfPackSupport>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Meshless.h"
#include "FoldAngleUtil.h"
#include "curvami/RestartManager.h"
#include "curvami/solver/ContinuationSolver.h"
#include "PlateProblem.h"

class NonLinearSolver : public ContinuationSolver {
public:
    using ContinuationSolver::LineSearch;
    using ContinuationSolver::ArcLength;

    // Plate entry points: wrap the plate in a PlateProblem for this call. flagRig
    // is per call, not per plate (KinematicDeployment always passes false).
    bool LineSearch(Meshless& plate, double lambda, Eigen::VectorXd& d, bool flagRig) {
        PlateProblem prob(plate, flagRig);
        return ContinuationSolver::LineSearch(prob, lambda, d);
    }

    // The energies are computed after every accepted step whether or not a
    // callback is set, exactly as before the split.
    void ArcLength(Meshless& plate,
                   double lambda_0,
                   Eigen::VectorXd x0,
                   double lambda_tot,
                   bool flagRig,
                   std::function<void(int, std::vector<double>, const Eigen::VectorXd&, double, double, double, double)> on_converged = nullptr) {
        PlateProblem prob(plate, flagRig);
        ContinuationSolver::ArcLength(prob, lambda_0, std::move(x0), lambda_tot,
            [&](int step, double lambda, const Eigen::VectorXd& x, double W_ext) {
                double E_hin = plate.computeTotalHingeEnergy(x);
                double E_mem = 0.0; double E_ben = 0.0;
                plate.computeStrainEnergies(x, E_mem, E_ben);

                if (on_converged) on_converged(step, std::vector<double>{lambda}, x, W_ext, E_mem, E_ben, E_hin);
            });
    }

    // =========================================================
    // MESHLESS DISPLACEMENT TRACKING
    // =========================================================
    double ComputeTrackingDisplacement(const Meshless& plate, const Eigen::VectorXd& x) const {
        return trackingDisplacement(plate, x);
    }
    
    // =========================================================
    // LENGTH-WEIGHTED AVERAGE FOLD ANGLE (degrees)
    // =========================================================
    double ComputeFoldAngleDeg(const Meshless& plate, size_t fold_idx) const {
        const auto& fold = plate.folds[fold_idx];
        if (fold.left.gps.empty() || fold.right.gps.empty()) return 0.0;

        const size_t n = fold.left.gps.size();
        std::vector<double> theta, weight;
        theta.reserve(n); weight.reserve(n);

        for (size_t gp_idx = 0; gp_idx < n; ++gp_idx) {
            // A progressive crease reports only the part that exists.
            if (!Constraints::gpActive(&fold.gp_active, gp_idx)) continue;
            const auto& gL = fold.left.gps[gp_idx];
            const auto& gR = fold.right.gps[gp_idx];

            auto kinL = Kinematics::compute(gL, plate.nodes);
            auto kinR = Kinematics::compute(gR, plate.nodes);

            Eigen::Vector3d aT = kinL.F0.col(0) * gL.tangent.x() + kinL.F0.col(1) * gL.tangent.y();
            double c_val = kinL.b.dot(kinR.b);
            double s_val = (kinL.b.cross(kinR.b)).dot(aT.normalized());

            theta.push_back(std::atan2(-s_val, c_val));
            weight.push_back(gL.weight);
        }
        if (theta.empty()) return 0.0;

        // Near flat-fold the per-GP values straddle the atan2 cut (+179 / -179)
        // and a plain average reads ~0 deg. The shared utility puts them on one
        // branch about their circular mean first; away from the cut that changes
        // nothing.
        return FoldAngle::weightedMeanAboutCircularMeanRad(theta, weight) * 180.0 / M_PI;
    }

    // =========================================================
    // KKT ASSEMBLY
    // =========================================================
    // =========================================================
        // INITIAL FOLDED CONFIGURATION
        // =========================================================
        Eigen::VectorXd GetInitialConditionWithFolds(Meshless& plate) {
            printf("\n--- Computing Initial Folded Condition (Symmetric MUMPS KKT) ---\n");
            
            const int ns = static_cast<int>(plate.nodes.size()) * 3;
            Eigen::VectorXd x = Eigen::VectorXd::Zero(ns);
            
            Eigen::SparseMatrix<double> J_all;
            Eigen::VectorXd c_all;
            
            plate.evaluateConstraints(x, c_all, J_all, true);
            
            if (c_all.size() == 0) {
                return x;
            }
            
            InitializeTopology(J_all);
            
            Eigen::SparseMatrix<double> J_ind = filterRows(J_all);
            
            Eigen::VectorXd c_ind(static_cast<int>(ind_rows.size()));
            for (int i = 0; i < static_cast<int>(ind_rows.size()); ++i) {
                c_ind(i) = c_all(ind_rows[i]);
            }
            
            Eigen::SparseMatrix<double> K;
            Eigen::VectorXd F_int;
            
            plate.assembleSystem(K, F_int, x, true);
            
            Eigen::VectorXd xb = ComputeGeometricCorrector(J_ind, c_ind, ns);
            
            int n_const_size = static_cast<int>(ind_rows.size());
            Eigen::VectorXd KKT_rhs = Eigen::VectorXd::Zero(ns + n_const_size);
            KKT_rhs.head(ns) = -(K.selfadjointView<Eigen::Upper>() * xb);
            

            Eigen::VectorXd sol = KKT_rhs; // MUMPS overwrites this with the solution

            if (!kkt_solver.solve(K, J_ind, sol)) {
                throw std::runtime_error("Initial folded-condition native MUMPS solve failed with error code: "
                                         + std::to_string(kkt_solver.lastError()));
            }

            x = xb + sol.head(ns);
            
            const double geom_error = (J_ind * x + c_ind).lpNorm<Eigen::Infinity>();
            
            printf(">>> Initial Condition Found! Geom Error = %.2e\n", geom_error);
            
            return x;
        }
    
    // =========================================================
        // TARGET-DIRECTED ARC-LENGTH DEPLOYMENT
        // =========================================================
        void TargetDirectedArcLength(Meshless& plate,
                                     double lambda_0,
                                     Eigen::VectorXd x0,
                                     double lambda_tot,
                                     bool flagRig,
                                     double target_tol_deg = 2.0,
                                     std::function<void(int, std::vector<double>, const Eigen::VectorXd&, double, double, double, double)> on_converged = nullptr) {
            
            const int jmax = max_iters;
            const int ns = static_cast<int>(x0.size());
            
            printf("\n=================================================================================\n");
            printf(" >>> INITIATING TARGET-DIRECTED ARC-LENGTH DEPLOYMENT (SYMMETRIC MUMPS) <<<\n");
            printf("=================================================================================\n");
            
            plate.updateKinematics(x0);
            std::vector<int> active_folds;
            Eigen::VectorXd moment_ratios(plate.folds.size());
            
            int driver_count = 0;
            for (size_t i = 0; i < plate.folds.size(); ++i) {
                if (plate.folds[i].is_driver) {
                    active_folds.push_back(static_cast<int>(i));
                    double distance_to_travel = plate.folds[i].angle_target - plate.folds[i].Ang0;
                    moment_ratios(driver_count) = distance_to_travel;
                    driver_count++;
                }
            }
            moment_ratios.conservativeResize(driver_count);
            
            if (moment_ratios.cwiseAbs().maxCoeff() > 1e-12) {
                moment_ratios /= moment_ratios.cwiseAbs().maxCoeff();
            }
            
            if (active_folds.empty()) {
                printf("!!! ERROR: No active drivers found.\n");
                return;
            }
            
            printf("\n--- Programmed Actuation Distribution ---\n");
            for (size_t i = 0; i < active_folds.size(); ++i) {
                printf(" -> Fold %d Actuation Multiplier: %+.2f\n", active_folds[i], moment_ratios(i));
            }
            
            Eigen::VectorXd x = x0;
            double lambda = lambda_0;
            int k = 1;
            
            Eigen::VectorXd Delta_U_old = Eigen::VectorXd::Zero(ns);
            double Delta_lambda_old = 0.0;
            
            Eigen::SparseMatrix<double> J_all;
            Eigen::VectorXd c_all;
            
            plate.evaluateConstraints(x, c_all, J_all, flagRig);
            InitializeTopology(J_all);
            resetInertiaTracking();
            const int n_const = static_cast<int>(ind_rows.size());
            
            double W_ext = 0.0;
            Eigen::VectorXd x_prev_step = x;
            
            if (!restart_file.empty()) {
                if (RestartManager::load(restart_file, k, lambda, ds, Delta_lambda_old, W_ext, x, Delta_U_old, ns)) {
                    printf("\n>>> RESTART SUCCESSFUL: Resuming from Step %d (Lambda = %.4f) <<<\n", k, lambda);
                    // Work increments are measured from the previous step. After a
                    // restart that is the loaded state, not x0 (which is zero in
                    // restart mode): W_ext used to carry a constant offset.
                    x_prev_step = x;
                }
            }
            
            while (k <= kmax && lambda <= lambda_tot) {
                plate.updateKinematics(x);
                
                auto get_follower_forces = [&](const Eigen::VectorXd& x_eval) {
                    plate.updateKinematics(x_eval);
                    Eigen::VectorXd F_m = Eigen::VectorXd::Zero(ns);
                    for (size_t i = 0; i < active_folds.size(); ++i) {
                        int fold_idx = active_folds[i];
                        F_m += moment_ratios(i) * ComputeFoldGradient(plate, fold_idx);
                    }
                    return F_m;
                };
                
                Eigen::VectorXd F_ref = get_follower_forces(x);
                
                printf("\n=================================================================================\n");
                printf(" LOAD STEP %3d | Actuation Parameter (Lambda): %10.5e | ds = %10.5e\n", k, lambda, ds);
                printf("=================================================================================\n");
                
                Eigen::VectorXd Fext = F_ref;
                
                Eigen::SparseMatrix<double> K_mat;
                Eigen::VectorXd Fint_dummy;
                plate.assembleSystem(K_mat, Fint_dummy, x, flagRig);
                plate.evaluateConstraints(x, c_all, J_all, flagRig);
                Eigen::SparseMatrix<double> J_ind = filterRows(J_all);
                
                Eigen::VectorXd c_ind(n_const);
                for (int i = 0; i < n_const; ++i) c_ind(i) = c_all(ind_rows[i]);
                
                Eigen::VectorXd xb_p = ComputeGeometricCorrector(J_ind, c_ind, ns);
                
                const int n_kkt = static_cast<int>(K_mat.rows()) + static_cast<int>(J_ind.rows());

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
                    if (Delta_U_old.dot(d_T) >= 0.0) { sgn = 1.0; confirmInertiaOnly(n_const); }
                    else if (confirmReversal(n_const, " >>> SNAP-BACK DETECTED: Reversing Load Increment <<<\n", "SNAP-BACK DETECTED")) { sgn = -1.0; noteReversal(); }
                    else sgn = 1.0;
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
                    
                    plate.evaluateConstraints(x_curr, c_all, J_all, flagRig);
                    Eigen::SparseMatrix<double> J_ind_c = filterRows(J_all);
                    Eigen::VectorXd c_ind_c(n_const);
                    for (int i = 0; i < n_const; ++i) c_ind_c(i) = c_all(ind_rows[i]);
                    
                    Eigen::VectorXd Fext_c = get_follower_forces(x_curr);
                    
                    Eigen::SparseMatrix<double> K_c;
                    Eigen::VectorXd Fint_c;
                    plate.assembleSystem(K_c, Fint_c, x_curr, flagRig);
                    
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
                        current_cos_theta = std::max(-1.0, std::min(1.0, (Delta_U_old.dot(Delta_U) + old_load * new_load) / (old_norm * new_norm)));
                    }
                }
                
                ASLA(icorr_ok, k_corr, current_cos_theta);
                
                if (icorr_ok == 1) {
                    ApplyMatlabCurvatureController(k, current_cos_theta);
                    noteAcceptedStep();
                    
                    Delta_U_old = Delta_U;
                    Delta_lambda_old = Delta_lambda;
                    x += Delta_U;
                    lambda += Delta_lambda;
                    
                    double avg_lambda = 0.5 * ((lambda - Delta_lambda) + lambda);
                    W_ext += avg_lambda * F_ref.dot(x - x_prev_step);
                    x_prev_step = x;
                    
                    plate.updateKinematics(x);
                    std::vector<double> output_data;
                    output_data.push_back(lambda);
                    
                    for (size_t i = 0; i < plate.folds.size(); ++i) {
                        output_data.push_back(ComputeFoldAngleDeg(plate, i));
                    }

                    double E_mem = 0.0, E_ben = 0.0;
                    plate.computeStrainEnergies(x, E_mem, E_ben);
                    if (on_converged) on_converged(k, output_data, x, W_ext, E_mem, E_ben, plate.computeTotalHingeEnergy(x));
                    
                    ++k;
                    if (!restart_file.empty()) RestartManager::save(restart_file, k, lambda, ds, Delta_lambda_old, W_ext, x, Delta_U_old);
                    
                    bool all_reached = true;
                    for (size_t i = 0; i < plate.folds.size(); ++i) {
                        if (plate.folds[i].is_driver) {
                            double target_deg = plate.folds[i].angle_target * 180.0 / M_PI;
                            // Wrapped distance: -179.5 is 0.5 deg from +180, not 359.5.
                            double err_deg = FoldAngle::wrappedDistanceDeg(output_data[i+1], target_deg);
                            if (std::abs(err_deg) > target_tol_deg) {
                                all_reached = false;
                                break;
                            }
                        }
                    }
                    if (all_reached) {
                        printf("\n>>> TARGET STATE ACHIEVED! Terminating deployment. <<<\n");
                        break;
                    }

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
    
    
    
    // =========================================================
    // ANALYTICAL FOLD KINEMATIC GRADIENT (J-INTEGRAL B-MATRIX)
    // =========================================================
    Eigen::VectorXd ComputeFoldGradient(const Meshless& plate, int fold_idx) const {
        int ns = static_cast<int>(plate.nodes.size()) * 3;
        Eigen::VectorXd J_i = Eigen::VectorXd::Zero(ns);
        
        const auto& fold = plate.folds[fold_idx];
        if (fold.left.gps.empty() || fold.right.gps.empty()) return J_i;
        
        // Helper lambda to construct a skew-symmetric cross-product matrix
        auto cross_mat = [](const Eigen::Vector3d& v) {
            Eigen::Matrix3d m;
            m <<   0.0, -v.z(),  v.y(),
            v.z(),    0.0, -v.x(),
            -v.y(),  v.x(),    0.0;
            return m;
        };
        
        // Loop over ALL integration points to distribute the moment along the crease
        for (size_t gp_idx = 0; gp_idx < fold.left.gps.size(); ++gp_idx) {
            const auto& gL = fold.left.gps[gp_idx];
            const auto& gR = fold.right.gps[gp_idx];
            
            auto kinL = Kinematics::compute(gL, plate.nodes);
            auto kinR = Kinematics::compute(gR, plate.nodes);
            
            Eigen::Vector3d a1L = kinL.F0.col(0), a2L = kinL.F0.col(1), bL = kinL.b;
            Eigen::Vector3d a1R = kinR.F0.col(0), a2R = kinR.F0.col(1), bR = kinR.b;
            
            double tx = gL.tangent.x();
            double ty = gL.tangent.y();
            Eigen::Vector3d aT = a1L * tx + a2L * ty;
            
            double aT_norm = aT.norm();
            if (aT_norm < 1e-14) continue; // Degenerate tangent
            
            Eigen::Vector3d t = aT / aT_norm;
            Eigen::Matrix3d Pt = (Eigen::Matrix3d::Identity() - t * t.transpose()) / aT_norm;
            
            double c = bL.dot(bR);
            Eigen::Vector3d n_LR = bL.cross(bR);
            double s = n_LR.dot(t);
            double denom = c * c + s * s;
            
            if (denom < 1e-14) continue; // Collinear degeneracy
            
            // The integration weight (length segment) for distributing the force
            double weight = gL.weight;
            
            // --- 1. Compute Gradients for LEFT Panel Nodes ---
            for (size_t i = 0; i < gL.neighbor_node_ids.size(); ++i) {
                int I = gL.neighbor_node_ids[i];
                double dPsix = gL.dPsi[i].x();
                double dPsiy = gL.dPsi[i].y();
                
                Eigen::Vector3d WL = dPsiy * a1L - dPsix * a2L;
                Eigen::Matrix3d GL = -cross_mat(WL);
                
                double dPsi_t = dPsix * tx + dPsiy * ty;
                Eigen::Matrix3d dt_du = dPsi_t * Pt;
                
                Eigen::Vector3d dc_du = GL.transpose() * bR;
                Eigen::Vector3d ds_du = GL.transpose() * (bR.cross(t)) + dt_du.transpose() * n_LR;
                
                // Apply the integration weight
                J_i.segment<3>(I * 3) += weight * ((s * dc_du - c * ds_du) / denom);
            }
            
            // --- 2. Compute Gradients for RIGHT Panel Nodes ---
            for (size_t i = 0; i < gR.neighbor_node_ids.size(); ++i) {
                int I = gR.neighbor_node_ids[i];
                double dPsix = gR.dPsi[i].x();
                double dPsiy = gR.dPsi[i].y();
                
                Eigen::Vector3d WR = dPsiy * a1R - dPsix * a2R;
                Eigen::Matrix3d GR = -cross_mat(WR);
                
                Eigen::Vector3d dc_du = GR.transpose() * bL;
                Eigen::Vector3d ds_du = GR.transpose() * (t.cross(bL));
                
                // Apply the integration weight
                J_i.segment<3>(I * 3) += weight * ((s * dc_du - c * ds_du) / denom);
            }
        }
        
        return J_i;
    }
    
    // =========================================================
        // KINEMATIC DEPLOYMENT (STIFF DRIVER, SOFT PASSIVES)
        // =========================================================
        void KinematicDeployment(Meshless& plate, Eigen::VectorXd& x, int num_steps = 50,
                                 std::function<void(int, const Eigen::VectorXd&, const std::vector<double>&, const std::vector<int>&, double, double, double)> on_converged = nullptr) {
            
            int last_completed_global_step = -1;
            
            double d_lambda = 0.0, d_ds = 0.0, d_dl = 0.0, d_wext = 0.0;
            Eigen::VectorXd d_dU = Eigen::VectorXd::Zero(x.size());
            
            if (!restart_file.empty()) {
                if (RestartManager::load(restart_file, last_completed_global_step, d_lambda, d_ds, d_dl, d_wext, x, d_dU,
                                         static_cast<int>(x.size()))) {
                    printf("\n=================================================================\n");
                    printf(" >>> RESTART SUCCESSFUL: Resuming from Global Step %d <<<\n", last_completed_global_step + 1);
                }
            }
            
            if (last_completed_global_step == -1) {
                printf("\n=================================================================\n");
                printf(" >>> INITIATING KINEMATIC DEPLOYMENT (ENERGY MINIMIZATION) <<<\n");
                printf("=================================================================\n");
            }
            
            for (int local_step = 0; local_step <= num_steps; ++local_step) {
                
                if (last_completed_global_step >= 0 && local_step == 0) {
                    continue;
                }
                
                int current_global_step = (last_completed_global_step == -1) ? 0 : last_completed_global_step + 1;
                
                for (size_t i = 0; i < plate.folds.size(); ++i) {
                    auto& f = plate.folds[i];
                    if (f.is_driver) {
                        double progress = static_cast<double>(local_step) / num_steps;
                        f.Ang0 = f.fold_sign * (f.angle_start + progress * (f.angle_target - f.angle_start));
                        
                        if (local_step > 0) {
                            printf("\n--- Global Step %d | Phase Step %d/%d | Actuating Driver %zu to %.2f deg ---\n",
                                   current_global_step, local_step, num_steps, i, std::abs(f.Ang0) * 180.0 / M_PI);
                        }
                    }
                }
                
                // Dispatches LineSearch (which now utilizes native parallel MUMPS internally)
                bool converged = LineSearch(plate, 0.0, x, false);
                
                if (!converged) {
                    printf("!!! Solver diverged at global step %d. Kinematic lock likely.\n", current_global_step);
                    break;
                }
                
                plate.updateKinematics(x);
                std::vector<double> current_angles(plate.folds.size(), 0.0);
                std::vector<int> current_drivers(plate.folds.size(), 0);
                
                for (size_t i = 0; i < plate.folds.size(); ++i) {
                    current_drivers[i] = plate.folds[i].is_driver ? 1 : 0;
                    
                    if (plate.folds[i].left.gps.empty() || plate.folds[i].right.gps.empty()) {
                        current_angles[i] = plate.folds[i].Ang0 * 180.0 / M_PI;
                        continue;
                    }

                    current_angles[i] = ComputeFoldAngleDeg(plate, i);

                    if (!plate.folds[i].is_driver && local_step > 0) {
                        printf("  [Fold %zu] Yielded Angle: %6.2f deg\n", i, current_angles[i]);
                    }
                }
                
                if (!restart_file.empty()) {
                    RestartManager::save(restart_file, current_global_step, d_lambda, d_ds, d_dl, d_wext, x, d_dU);
                }
                
                if (on_converged) {
                    double E_mem = 0.0, E_ben = 0.0;
                    plate.computeStrainEnergies(x, E_mem, E_ben);
                    double E_hin = plate.computeTotalHingeEnergy(x);
                    on_converged(current_global_step, x, current_angles, current_drivers, E_mem, E_ben, E_hin);
                }
                
                last_completed_global_step = current_global_step;
            }
        }
    
    // =========================================================
        // PURE KINEMATIC ARC-LENGTH (FORCE-FREE W/ SNAP-BACK)
        // =========================================================
        void KinematicArcLength(Meshless& plate,
                                const Eigen::VectorXd& x0,
                                bool flagRig,
                                std::function<void(int, std::vector<double>, const Eigen::VectorXd&, double, double, double, double)> on_converged = nullptr) {
            
            const int jmax = max_iters;
            const int ns = static_cast<int>(x0.size());
            
            printf("\n=================================================================================\n");
            printf(" >>> INITIATING PURE KINEMATIC ARC-LENGTH (ZERO EXTERNAL FORCES - SYMMETRIC MUMPS) <<<\n");
            printf("=================================================================================\n");
            
            Eigen::VectorXd x = x0;
            double mu = 0.0;
            int k = 1;
            
            Eigen::VectorXd Delta_U_old = Eigen::VectorXd::Zero(ns);
            double Delta_mu_old = 0.0;
            
            Eigen::SparseMatrix<double> J_all;
            Eigen::VectorXd c_all;
            plate.evaluateConstraints(x, c_all, J_all, flagRig);
            InitializeTopology(J_all);
            resetInertiaTracking();
            const int n_const = static_cast<int>(ind_rows.size());
            
            double W_ext = 0.0;
            
            auto update_driver_angles = [&](double mu_val) {
                for (auto& f : plate.folds) {
                    if (f.is_driver) {
                        f.Ang0 = f.fold_sign * (f.angle_start + mu_val * (f.angle_target - f.angle_start));
                    }
                }
            };
            
            auto compute_F_pseudo = [&](const Meshless& plate_obj) {
                Eigen::VectorXd F_p = Eigen::VectorXd::Zero(ns);
                for (size_t i = 0; i < plate_obj.folds.size(); ++i) {
                    if (plate_obj.folds[i].is_driver) {
                        double dtheta0_dmu = plate_obj.folds[i].fold_sign * (plate_obj.folds[i].angle_target - plate_obj.folds[i].angle_start);
                        F_p -= plate_obj.folds[i].Kt * dtheta0_dmu * ComputeFoldGradient(plate_obj, static_cast<int>(i));
                    }
                }
                return F_p;
            };
            
            if (!restart_file.empty()) {
                if (RestartManager::load(restart_file, k, mu, ds, Delta_mu_old, W_ext, x, Delta_U_old, ns)) {
                    printf("\n>>> RESTART SUCCESSFUL: Resuming from Step %d (mu = %.4f) <<<\n", k, mu);
                }
            }
            
            while (k <= kmax && mu <= 1.05) {
                printf("\n=================================================================================\n");
                printf(" LOAD STEP %3d | Deployment Parameter (mu): %10.5e | ds = %10.5e\n", k, mu, ds);
                printf("=================================================================================\n");
                
                update_driver_angles(mu);
                Eigen::VectorXd F_pseudo = compute_F_pseudo(plate);
                
                Eigen::SparseMatrix<double> K_mat;
                Eigen::VectorXd Fint_dummy;
                plate.assembleSystem(K_mat, Fint_dummy, x, flagRig);
                plate.evaluateConstraints(x, c_all, J_all, flagRig);
                Eigen::SparseMatrix<double> J_ind = filterRows(J_all);
                
                Eigen::VectorXd c_ind(n_const);
                for (int i = 0; i < n_const; ++i) c_ind(i) = c_all(ind_rows[i]);
                
                Eigen::VectorXd xb_p = ComputeGeometricCorrector(J_ind, c_ind, ns);
                
                const int n_kkt = static_cast<int>(K_mat.rows()) + static_cast<int>(J_ind.rows());

                Eigen::VectorXd sol_T = Eigen::VectorXd::Zero(n_kkt);
                sol_T.head(ns) = F_pseudo;

                if (!kkt_solver.solve(K_mat, J_ind, sol_T)) {
                    printf(">> Predictor KKT solve failed (MUMPS INFO(1) = %d).\n",
                           kkt_solver.lastError());
                    break;
                }

                Eigen::VectorXd d_T = sol_T.head(ns);

                double sgn = 1.0;
                if (k > 1) {
                    if ((Delta_U_old.dot(d_T) + psi * psi * Delta_mu_old * 1.0) < 0.0) {
                        if (confirmReversal(n_const, " >>> KINEMATIC SNAP-BACK DETECTED: Navigating Limit Point! <<<\n", "KINEMATIC SNAP-BACK DETECTED")) { sgn = -1.0; noteReversal(); }
                    } else {
                        confirmInertiaOnly(n_const);
                    }
                }
                
                double Delta_mu = sgn * ds / std::sqrt(d_T.dot(d_T) + psi * psi);
                Eigen::VectorXd Delta_U = Delta_mu * d_T + xb_p;
                
                Eigen::VectorXd Delta_U_pred = Delta_U;
                double Delta_mu_pred = Delta_mu;
                
                int icorr_ok = 0;
                int k_corr = jmax;
                
                printf("Iter      Max Force (inf)      Energy Chg           Geom. Error          mu Drift\n");
                printf("------------------------------------------------------------------------------------------\n");
                
                for (int j = 1; j <= jmax; ++j) {
                    const double curr_mu = mu + Delta_mu;
                    const Eigen::VectorXd x_curr = x + Delta_U;
                    
                    plate.evaluateConstraints(x_curr, c_all, J_all, flagRig);
                    Eigen::SparseMatrix<double> J_ind_c = filterRows(J_all);
                    Eigen::VectorXd c_ind_c(n_const);
                    for (int i = 0; i < n_const; ++i) c_ind_c(i) = c_all(ind_rows[i]);
                    
                    update_driver_angles(curr_mu);
                    Eigen::VectorXd F_pseudo_c = compute_F_pseudo(plate);
                    
                    Eigen::SparseMatrix<double> K_c;
                    Eigen::VectorXd Fint_c;
                    plate.assembleSystem(K_c, Fint_c, x_curr, flagRig);
                    
                    Eigen::VectorXd R_vec = Fint_c;
                    Eigen::VectorXd xb_c = ComputeGeometricCorrector(J_ind_c, c_ind_c, ns);
                    
                    const int n_kkt_c = static_cast<int>(K_c.rows()) + static_cast<int>(J_ind_c.rows());

                    Eigen::MatrixXd sol_c = Eigen::MatrixXd::Zero(n_kkt_c, 2);
                    sol_c.col(0).head(ns) = F_pseudo_c;
                    sol_c.col(1).head(ns) = -(R_vec + K_c.selfadjointView<Eigen::Upper>() * xb_c);

                    if (!kkt_solver.solve(K_c, J_ind_c, sol_c)) {
                        printf(">> Corrector KKT solve failed (MUMPS INFO(1) = %d).\n",
                               kkt_solver.lastError());
                        icorr_ok = 0; k_corr = j; break;
                    }

                    Eigen::VectorXd daP = sol_c.col(0).head(ns);
                    Eigen::VectorXd daPsi = sol_c.col(1).head(ns) + xb_c;
                    Eigen::VectorXd lambda_mult = sol_c.col(1).tail(n_const);

                    const double arc_denom = Delta_U_pred.dot(daP) + psi * psi * Delta_mu_pred;
                    const double arc_numer = Delta_U_pred.dot(daPsi);
                    const double dm_c = -arc_numer / arc_denom;
                    
                    Delta_mu += dm_c;
                    Delta_U += daPsi + dm_c * daP;
                    
                    const double max_force = (R_vec + J_ind_c.transpose() * lambda_mult).lpNorm<Eigen::Infinity>();
                    const double energy_change = std::abs(daPsi.dot(R_vec));
                    const double geom_error = c_ind_c.lpNorm<Eigen::Infinity>();
                    const double mu_drift = std::abs(dm_c);
                    
                    if (j > 1 && max_force <= tol_force && energy_change <= tol_energy && geom_error <= tol_geom) {
                        printf(">> Convergence! Max Force = %.2e | Energy = %.2e | Geom = %.2e\n", max_force, energy_change, geom_error);
                        icorr_ok = 1; k_corr = j; break;
                    }
                    if (max_force >= div || std::isnan(max_force)) {
                        printf(">> Divergence detected.\n");
                        icorr_ok = 0; k_corr = j; break;
                    }
                    
                    printf("%3d      %12.5e         %12.5e         %12.5e         %12.5e\n", j, max_force, energy_change, geom_error, mu_drift);
                }
                
                double current_cos_theta = 1.0;
                if (k > 1 && icorr_ok == 1) {
                    const double old_load = psi * Delta_mu_old;
                    const double new_load = psi * Delta_mu;
                    const double old_norm = std::sqrt(Delta_U_old.squaredNorm() + old_load * old_load);
                    const double new_norm = std::sqrt(Delta_U.squaredNorm() + new_load * new_load);
                    if (old_norm > 0.0 && new_norm > 0.0) {
                        current_cos_theta = std::max(-1.0, std::min(1.0, (Delta_U_old.dot(Delta_U) + old_load * new_load) / (old_norm * new_norm)));
                    }
                }
                
                ASLA(icorr_ok, k_corr, current_cos_theta);
                
                if (icorr_ok == 1) {
                    ApplyMatlabCurvatureController(k, current_cos_theta);
                    noteAcceptedStep();
                    Delta_U_old = Delta_U; Delta_mu_old = Delta_mu; x += Delta_U; mu += Delta_mu;
                    
                    plate.updateKinematics(x);
                    std::vector<double> output_data; output_data.push_back(mu);
                    
                    for (size_t i = 0; i < plate.folds.size(); ++i) {
                        output_data.push_back(ComputeFoldAngleDeg(plate, i));
                    }

                    double E_mem = 0.0, E_ben = 0.0; plate.computeStrainEnergies(x, E_mem, E_ben);
                    if (on_converged) on_converged(k, output_data, x, 0.0, E_mem, E_ben, plate.computeTotalHingeEnergy(x));
                    
                    ++k;
                    if (!restart_file.empty()) RestartManager::save(restart_file, k, mu, ds, Delta_mu_old, 0.0, x, Delta_U_old);
                    
                    if (mu >= 1.0) {
                        printf("\n>>> TARGET KINEMATIC MANIFOLD COMPLETELY TRACED (mu >= 1.0)! <<<\n");
                        break;
                    }
                } else {
                    if (ds <= dsmin) { printf("!!! FATAL: ds reached minimum threshold. Terminating.\n"); break; }
                }
            }
        }
    };
