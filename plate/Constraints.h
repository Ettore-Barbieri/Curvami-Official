#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>
#include "PlateData.h"
#include "Kinematics.h"

namespace Constraints {

    typedef Eigen::Triplet<double> T;

    // Per-Gauss-point crease state for progressive creasing. A null or empty
    // vector means "not set": every point active, rest angle the fold's scalar.
    inline bool gpActive(const std::vector<char>* active, size_t i) {
        return !active || active->empty() || (*active)[i] != 0;
    }
    inline double gpTheta0(double theta0, const std::vector<double>* theta0_gp, size_t i) {
        return (theta0_gp && !theta0_gp->empty()) ? (*theta0_gp)[i] : theta0;
    }

    inline void evaluateDirichlet(
        const std::vector<GaussPointData>& bnd_gps, const std::vector<Node>& nodes,
        const BoundaryCondition& bc, int& row_ptr, std::vector<double>& c_vec, std::vector<T>& J_triplets,
        std::vector<double>* dcdtheta = nullptr)
    {
        for (const auto& gp : bnd_gps) {
            Eigen::Vector3d u_curr = Eigen::Vector3d::Zero();
            Eigen::Vector3d dudX = Eigen::Vector3d::Zero(), dudY = Eigen::Vector3d::Zero();
            
            for (size_t j = 0; j < gp.neighbor_node_ids.size(); ++j) {
                int id = gp.neighbor_node_ids[j];
                u_curr += nodes[id].u * gp.Psi[j];
                dudX += nodes[id].u * gp.dPsi[j].x();
                dudY += nodes[id].u * gp.dPsi[j].y();
            }

            if (bc.active_u) { c_vec.push_back(u_curr.x() - bc.val_u); if(dcdtheta) dcdtheta->push_back(0.0); for(size_t j=0; j<gp.Psi.size(); ++j) J_triplets.push_back(T(row_ptr, gp.neighbor_node_ids[j]*3+0, gp.Psi[j])); row_ptr++; }
            if (bc.active_v) { c_vec.push_back(u_curr.y() - bc.val_v); if(dcdtheta) dcdtheta->push_back(0.0); for(size_t j=0; j<gp.Psi.size(); ++j) J_triplets.push_back(T(row_ptr, gp.neighbor_node_ids[j]*3+1, gp.Psi[j])); row_ptr++; }
            if (bc.active_w) { c_vec.push_back(u_curr.z() - bc.val_w); if(dcdtheta) dcdtheta->push_back(0.0); for(size_t j=0; j<gp.Psi.size(); ++j) J_triplets.push_back(T(row_ptr, gp.neighbor_node_ids[j]*3+2, gp.Psi[j])); row_ptr++; }

            if (bc.active_rot) {
                double nx = gp.normal.x(), ny = gp.normal.y();
                c_vec.push_back((dudX.z()*nx + dudY.z()*ny) - bc.val_rot);
                if(dcdtheta) dcdtheta->push_back(0.0);
                for(size_t j=0; j<gp.Psi.size(); ++j) J_triplets.push_back(T(row_ptr, gp.neighbor_node_ids[j]*3+2, gp.dPsi[j].x()*nx + gp.dPsi[j].y()*ny));
                row_ptr++;
            }
        }
    }

    inline void computeNormalDerivatives(const Eigen::Vector3d& a1, const Eigen::Vector3d& a2,
                                         const Eigen::Vector3d& b, double norm_n,
                                         Eigen::Matrix<double, 3, 6>& db_dF)
    {
        Eigen::Matrix<double, 3, 6> dn_dF;
        for (int i = 0; i < 3; ++i) {
            dn_dF.col(i) = Eigen::Vector3d::Unit(i).cross(a2);
            dn_dF.col(i+3) = a1.cross(Eigen::Vector3d::Unit(i));
        }
        db_dF = (Eigen::Matrix3d::Identity() - b * b.transpose()) / norm_n * dn_dF;
    }

    // theta and its gradient as functions of the two deformation gradients only.
    // F packs (a1L, a2L, a1R, a2R); tx/ty are the crease tangent components.
    // Used both for the residual and, when the exact tangent is requested, as the
    // function that gets differenced - so the Hessian is guaranteed to be the
    // derivative of exactly the gradient the residual uses.
    inline double foldAngleAndGrad(const Eigen::Matrix<double, 12, 1>& F,
                                   double tx, double ty,
                                   Eigen::Matrix<double, 12, 1>* grad)
    {
        const Eigen::Vector3d a1L = F.segment<3>(0), a2L = F.segment<3>(3);
        const Eigen::Vector3d a1R = F.segment<3>(6), a2R = F.segment<3>(9);

        const Eigen::Vector3d nL = a1L.cross(a2L), nR = a1R.cross(a2R);
        const double normL = nL.norm(), normR = nR.norm();
        const Eigen::Vector3d bL = nL / normL, bR = nR / normR;

        const Eigen::Vector3d aT = a1L * tx + a2L * ty;
        const double norm_aT = aT.norm();
        const Eigen::Vector3d t = aT / norm_aT;

        const double c = bL.dot(bR);
        const double s = (bL.cross(bR)).dot(t);
        const double theta = std::atan2(-s, c);

        if (grad) {
            Eigen::Matrix<double, 3, 6> dbL_dFL, dbR_dFR;
            computeNormalDerivatives(a1L, a2L, bL, normL, dbL_dFL);
            computeNormalDerivatives(a1R, a2R, bR, normR, dbR_dFR);

            Eigen::Matrix<double, 3, 6> daT_dFL;
            daT_dFL << tx, 0, 0, ty, 0, 0,
                       0, tx, 0, 0, ty, 0,
                       0, 0, tx, 0, 0, ty;
            const Eigen::Matrix<double, 3, 6> dt_dFL =
                (Eigen::Matrix3d::Identity() - t * t.transpose()) / norm_aT * daT_dFL;

            const Eigen::Matrix<double, 1, 6> dc_dFL = bR.transpose() * dbL_dFL;
            const Eigen::Matrix<double, 1, 6> dc_dFR = bL.transpose() * dbR_dFR;
            const Eigen::Matrix<double, 1, 6> ds_dFL =
                bR.cross(t).transpose() * dbL_dFL + bL.cross(bR).transpose() * dt_dFL;
            const Eigen::Matrix<double, 1, 6> ds_dFR = t.cross(bL).transpose() * dbR_dFR;

            // d/dx atan2(-s, c) = -(c*ds - s*dc) / (c^2 + s^2).
            // The divisor is only 1 when the crease tangent t is exactly parallel
            // to bL x bR; it is not, so omitting it biases the hinge force.
            // ComputeFoldGradient in NonLinearSolver.h divides by the same denom.
            const double denom = c * c + s * s;
            const double inv_denom = (denom > 1e-14) ? 1.0 / denom : 0.0;

            grad->head<6>() = -(c * ds_dFL - s * dc_dFL).transpose() * inv_denom;
            grad->tail<6>() = -(c * ds_dFR - s * dc_dFR).transpose() * inv_denom;
        }
        return theta;
    }

    // One row of a tie that is not pointwise: a weighted sum of the pointwise jumps
    // u+ - u- at the crease Gauss points `terms` (index, weight). Each row gives
    // three constraints, one per displacement component.
    struct TieRow { std::vector<std::pair<size_t, double>> terms; };

    // Active crease points in order along the crease, with their arc position
    // (measured along the tangent of the first point; the crease is straight).
    inline std::vector<std::pair<double, size_t>> orderedCreasePoints(
        const std::vector<GaussPointData>& gpL, const std::vector<char>* active)
    {
        std::vector<std::pair<double, size_t>> pts;
        Eigen::Vector2d T0 = Eigen::Vector2d::Zero();
        for (size_t i = 0; i < gpL.size(); ++i) {
            if (!gpActive(active, i)) continue;
            if (pts.empty()) T0 = gpL[i].tangent;
            pts.push_back({T0.dot(gpL[i].X_ref), i});
        }
        std::sort(pts.begin(), pts.end());
        return pts;
    }

    inline std::vector<TieRow> buildTieRows(const std::vector<GaussPointData>& gpL, const FoldTie& tie,
                                            const std::vector<char>* active)
    {
        const auto pts = orderedCreasePoints(gpL, active);
        std::vector<TieRow> rows;
        if (tie.type == FoldTie::Subset) {
            const size_t k = static_cast<size_t>(std::max(1, tie.every));
            for (size_t m = 0; m < pts.size(); m += k) rows.push_back({{{pts[m].second, 1.0}}});
        }
        else if ((tie.type == FoldTie::MortarP0 || tie.type == FoldTie::MortarP1) && !pts.empty()) {
            // the crease runs from s0 to s1: each Gauss point stands for its own weight
            double s0 = 1e300, s1 = -1e300;
            for (const auto& p : pts) {
                s0 = std::min(s0, p.first - 0.5 * gpL[p.second].weight);
                s1 = std::max(s1, p.first + 0.5 * gpL[p.second].weight);
            }
            const int nseg = std::max(1, static_cast<int>(std::lround((s1 - s0) / tie.length)));
            const double len = (s1 - s0) / nseg;
            if (tie.type == FoldTie::MortarP0) {
                rows.resize(static_cast<size_t>(nseg));
                for (const auto& p : pts) {
                    const int j = std::min(nseg - 1, std::max(0, static_cast<int>(std::floor((p.first - s0) / len))));
                    rows[static_cast<size_t>(j)].terms.push_back({p.second, gpL[p.second].weight});
                }
            } else {
                rows.resize(static_cast<size_t>(nseg) + 1);
                for (const auto& p : pts) {
                    const double x = (p.first - s0) / len;
                    const int j = std::min(nseg - 1, std::max(0, static_cast<int>(std::floor(x))));
                    const double t = x - j, w = gpL[p.second].weight;
                    if (1.0 - t > 0.0) rows[static_cast<size_t>(j)].terms.push_back({p.second, w * (1.0 - t)});
                    if (t > 0.0)       rows[static_cast<size_t>(j) + 1].terms.push_back({p.second, w * t});
                }
            }
            rows.erase(std::remove_if(rows.begin(), rows.end(), [](const TieRow& r) { return r.terms.empty(); }), rows.end());
        }
        return rows;
    }

    inline size_t tieRowCount(const std::vector<GaussPointData>& gpL, const FoldTie& tie,
                              const std::vector<char>* active)
    {
        if (tie.type != FoldTie::Pointwise) return buildTieRows(gpL, tie, active).size();
        size_t n = 0;
        for (size_t i = 0; i < gpL.size(); ++i) if (gpActive(active, i)) ++n;
        return n;
    }

    // evaluateFold for a tie that is not pointwise: the tie rows first, then (rigid
    // folds only) the fold-angle row of every active crease point, as in evaluateFold.
    inline void evaluateFoldTied(
        const std::vector<GaussPointData>& gpL, const std::vector<GaussPointData>& gpR,
        const std::vector<Node>& nodes, double theta0, bool flagRig,
        int& row_ptr, std::vector<double>& c_vec, std::vector<T>& J_triplets,
        std::vector<double>* dcdtheta, double dtheta_dlambda,
        std::vector<int>* specific_fold_angle_rows,
        const std::vector<double>* theta0_gp, const std::vector<char>* active, const FoldTie& tie)
    {
        for (const TieRow& row : buildTieRows(gpL, tie, active)) {
            Eigen::Vector3d jump = Eigen::Vector3d::Zero();
            for (const auto& term : row.terms) {
                const auto& gL = gpL[term.first]; const auto& gR = gpR[term.first];
                for (size_t j = 0; j < gL.Psi.size(); ++j) jump += term.second * gL.Psi[j] * nodes[gL.neighbor_node_ids[j]].u;
                for (size_t j = 0; j < gR.Psi.size(); ++j) jump -= term.second * gR.Psi[j] * nodes[gR.neighbor_node_ids[j]].u;
            }
            for (int d = 0; d < 3; ++d) {
                c_vec.push_back(jump[d]);
                if (dcdtheta) dcdtheta->push_back(0.0);
                for (const auto& term : row.terms) {
                    const auto& gL = gpL[term.first]; const auto& gR = gpR[term.first];
                    for (size_t j = 0; j < gL.Psi.size(); ++j) J_triplets.push_back(T(row_ptr, gL.neighbor_node_ids[j]*3+d, term.second * gL.Psi[j]));
                    for (size_t j = 0; j < gR.Psi.size(); ++j) J_triplets.push_back(T(row_ptr, gR.neighbor_node_ids[j]*3+d, -term.second * gR.Psi[j]));
                }
                row_ptr++;
            }
        }

        if (!flagRig) return;
        for (size_t i = 0; i < gpL.size(); ++i) {
            if (!gpActive(active, i)) continue;
            const auto& gL = gpL[i]; const auto& gR = gpR[i];
            auto kinL = Kinematics::compute(gL, nodes);
            auto kinR = Kinematics::compute(gR, nodes);
            Eigen::Matrix<double, 12, 1> F;
            F.segment<3>(0) = kinL.F0.col(0); F.segment<3>(3) = kinL.F0.col(1);
            F.segment<3>(6) = kinR.F0.col(0); F.segment<3>(9) = kinR.F0.col(1);
            Eigen::Matrix<double, 12, 1> gradF;
            const double theta = foldAngleAndGrad(F, gL.tangent.x(), gL.tangent.y(), &gradF);
            if (specific_fold_angle_rows) specific_fold_angle_rows->push_back(row_ptr);
            c_vec.push_back(theta - gpTheta0(theta0, theta0_gp, i));
            if (dcdtheta) dcdtheta->push_back(-dtheta_dlambda);
            auto add_J = [&](int block, const GaussPointData& gp) {
                for (size_t j = 0; j < gp.neighbor_node_ids.size(); ++j) {
                    int id = gp.neighbor_node_ids[j];
                    double dX = gp.dPsi[j].x(), dY = gp.dPsi[j].y();
                    for (int d = 0; d < 3; ++d)
                        J_triplets.push_back(T(row_ptr, id * 3 + d,
                            gradF(block + d) * dX + gradF(block + d + 3) * dY));
                }
            };
            add_J(0, gL); add_J(6, gR);
            row_ptr++;
        }
    }

    inline void evaluateFold(
        const std::vector<GaussPointData>& gpL, const std::vector<GaussPointData>& gpR,
        const std::vector<Node>& nodes, double theta0, bool flagRig,
        int& row_ptr, std::vector<double>& c_vec, std::vector<T>& J_triplets,
        std::vector<double>* dcdtheta = nullptr, double dtheta_dlambda = 1.0,
        std::vector<int>* specific_fold_angle_rows = nullptr,
        const std::vector<double>* theta0_gp = nullptr,
        const std::vector<char>* active = nullptr,
        const FoldTie* tie = nullptr)
    {
        if (tie && tie->type != FoldTie::Pointwise) {
            evaluateFoldTied(gpL, gpR, nodes, theta0, flagRig, row_ptr, c_vec, J_triplets, dcdtheta,
                             dtheta_dlambda, specific_fold_angle_rows, theta0_gp, active, *tie);
            return;
        }
        for (size_t i = 0; i < gpL.size(); ++i) {
            if (!gpActive(active, i)) continue;
            const auto& gL = gpL[i]; const auto& gR = gpR[i];
            auto kinL = Kinematics::compute(gL, nodes);
            auto kinR = Kinematics::compute(gR, nodes);

            Eigen::Vector3d uL = Eigen::Vector3d::Zero(), uR = Eigen::Vector3d::Zero();
            for(size_t j=0; j<gL.Psi.size(); ++j) uL += nodes[gL.neighbor_node_ids[j]].u * gL.Psi[j];
            for(size_t j=0; j<gR.Psi.size(); ++j) uR += nodes[gR.neighbor_node_ids[j]].u * gR.Psi[j];
            
            Eigen::Vector3d jump = uL - uR;
            for (int d = 0; d < 3; ++d) {
                c_vec.push_back(jump[d]);
                if (dcdtheta) dcdtheta->push_back(0.0);
                for(size_t j=0; j<gL.Psi.size(); ++j) J_triplets.push_back(T(row_ptr, gL.neighbor_node_ids[j]*3+d, gL.Psi[j]));
                for(size_t j=0; j<gR.Psi.size(); ++j) J_triplets.push_back(T(row_ptr, gR.neighbor_node_ids[j]*3+d, -gR.Psi[j]));
                row_ptr++;
            }

            if (flagRig) {
                Eigen::Matrix<double, 12, 1> F;
                F.segment<3>(0) = kinL.F0.col(0); F.segment<3>(3) = kinL.F0.col(1);
                F.segment<3>(6) = kinR.F0.col(0); F.segment<3>(9) = kinR.F0.col(1);

                Eigen::Matrix<double, 12, 1> gradF;
                const double theta = foldAngleAndGrad(F, gL.tangent.x(), gL.tangent.y(), &gradF);

                if (specific_fold_angle_rows) specific_fold_angle_rows->push_back(row_ptr);

                c_vec.push_back(theta - gpTheta0(theta0, theta0_gp, i));

                if (dcdtheta) dcdtheta->push_back(-dtheta_dlambda);

                auto add_J = [&](int block, const GaussPointData& gp) {
                    for (size_t j = 0; j < gp.neighbor_node_ids.size(); ++j) {
                        int id = gp.neighbor_node_ids[j];
                        double dX = gp.dPsi[j].x(), dY = gp.dPsi[j].y();
                        for (int d = 0; d < 3; ++d)
                            J_triplets.push_back(T(row_ptr, id * 3 + d,
                                gradF(block + d) * dX + gradF(block + d + 3) * dY));
                    }
                };
                add_J(0, gL); add_J(6, gR);
                row_ptr++;
            }
        }
    }

    inline double computeHingeEnergy(
        const std::vector<GaussPointData>& gpL, const std::vector<GaussPointData>& gpR,
        const std::vector<Node>& nodes, double theta0, double Kt,
        const std::vector<double>* theta0_gp = nullptr,
        const std::vector<char>* active = nullptr)
    {
        double E_hinge = 0.0;

        for (size_t i = 0; i < gpL.size(); ++i) {
            if (!gpActive(active, i)) continue;
            const auto& gL = gpL[i]; const auto& gR = gpR[i];
            auto kinL = Kinematics::compute(gL, nodes);
            auto kinR = Kinematics::compute(gR, nodes);

            Eigen::Vector3d a1L = kinL.F0.col(0), a2L = kinL.F0.col(1), bL = kinL.b;
            Eigen::Vector3d a1R = kinR.F0.col(0), a2R = kinR.F0.col(1), bR = kinR.b;

            Eigen::Vector3d aT = a1L * gL.tangent.x() + a2L * gL.tangent.y();
            Eigen::Vector3d t = aT.normalized();

            double c = bL.dot(bR);
            double s = (bL.cross(bR)).dot(t);
            double theta = std::atan2(-s, c);

            // Pure torsional spring without penalty barriers
            double U_base = 0.5 * Kt * std::pow(theta - gpTheta0(theta0, theta0_gp, i), 2);
            E_hinge += gL.weight * U_base;
        }
        return E_hinge;
    }

    inline void evaluateElasticFold(
        const std::vector<GaussPointData>& gpL, const std::vector<GaussPointData>& gpR,
        const std::vector<Node>& nodes, double theta0, double Kt,
        Eigen::VectorXd& F_int, std::vector<T>* K_triplets,
        bool exact_tangent = false,
        const std::vector<double>* theta0_gp = nullptr,
        const std::vector<char>* active = nullptr)
    {
        for (size_t i = 0; i < gpL.size(); ++i) {
            if (!gpActive(active, i)) continue;
            const auto& gL = gpL[i]; const auto& gR = gpR[i];
            auto kinL = Kinematics::compute(gL, nodes);
            auto kinR = Kinematics::compute(gR, nodes);

            const double tx = gL.tangent.x(), ty = gL.tangent.y();

            Eigen::Matrix<double, 12, 1> F;
            F.segment<3>(0) = kinL.F0.col(0); F.segment<3>(3) = kinL.F0.col(1);
            F.segment<3>(6) = kinR.F0.col(0); F.segment<3>(9) = kinR.F0.col(1);

            Eigen::Matrix<double, 12, 1> gradF;
            const double theta = foldAngleAndGrad(F, tx, ty, &gradF);

            // Pure torsional spring internal moment and stiffness
            double moment_mag = Kt * (theta - gpTheta0(theta0, theta0_gp, i));
            double Kt_eff = Kt;

            double W = gL.weight;
            double W_moment = W * moment_mag;
            double W_Kt = W * Kt_eff;

            std::vector<int> all_ids;
            std::vector<double> all_grad;
            std::vector<Eigen::Matrix<double, 12, 1>> B;   // nodal dof -> F-space map

            auto extract_grad = [&](int block, const GaussPointData& gp) {
                for (size_t j = 0; j < gp.neighbor_node_ids.size(); ++j) {
                    int id = gp.neighbor_node_ids[j];
                    double dX = gp.dPsi[j].x(), dY = gp.dPsi[j].y();
                    for (int d = 0; d < 3; ++d) {
                        all_ids.push_back(id * 3 + d);
                        all_grad.push_back(gradF(block + d) * dX + gradF(block + d + 3) * dY);
                        if (exact_tangent && K_triplets != nullptr) {
                            Eigen::Matrix<double, 12, 1> b = Eigen::Matrix<double, 12, 1>::Zero();
                            b(block + d)     = dX;
                            b(block + d + 3) = dY;
                            B.push_back(b);
                        }
                    }
                }
            };
            extract_grad(0, gL); extract_grad(6, gR);

            // Exact linearization needs the dihedral-angle Hessian d2(theta)/du2.
            // theta depends on only 12 numbers, so the Hessian is obtained by
            // central-differencing the analytic gradient above - accurate to ~1e-8
            // with no error-prone hand derivation. The true Hessian is symmetric
            // (second derivative of a scalar); the tiny asymmetry left by the
            // difference stencil is numerical noise, so it is averaged out to keep
            // K exactly symmetric for MUMPS (sym=2 reads only the upper triangle).
            std::vector<Eigen::Matrix<double, 12, 1>> HB;
            if (exact_tangent && K_triplets != nullptr && std::abs(W_moment) > 0.0) {
                Eigen::Matrix<double, 12, 12> H;
                const double hstep = 1e-6 * std::max(1.0, F.cwiseAbs().maxCoeff());
                for (int k = 0; k < 12; ++k) {
                    Eigen::Matrix<double, 12, 1> Fp = F, Fm = F, gp_, gm_;
                    Fp(k) += hstep; Fm(k) -= hstep;
                    foldAngleAndGrad(Fp, tx, ty, &gp_);
                    foldAngleAndGrad(Fm, tx, ty, &gm_);
                    H.col(k) = (gp_ - gm_) / (2.0 * hstep);
                }
                H = 0.5 * (H + H.transpose()).eval();

                HB.resize(B.size());
                for (size_t m = 0; m < B.size(); ++m) HB[m] = H * B[m];
            }

            const bool use_hess = !HB.empty();

            for (size_t k = 0; k < all_ids.size(); ++k) {
                F_int(all_ids[k]) += all_grad[k] * W_moment;
                if (K_triplets != nullptr) {
                    for (size_t m = 0; m < all_ids.size(); ++m) {
                        double val = all_grad[k] * all_grad[m] * W_Kt;
                        if (use_hess) val += W_moment * B[k].dot(HB[m]);
                        // K is stored block-upper; the fold block is symmetric in
                        // (k,m), so the mirror is redundant and would fall outside
                        // the pattern.
                        if (all_ids[k] / 3 <= all_ids[m] / 3) {
                            K_triplets->push_back(T(all_ids[k], all_ids[m], val));
                        }
                    }
                }
            }
        }
    }
}
