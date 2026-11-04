#pragma once
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "PlateData.h"
#include "Assembly.h"
#include "Constraints.h"
#include "curvami/Enrichment.h"

class Meshless {
public:
    std::vector<Node> nodes;
    IntegrationDomain faces;
    std::vector<IntegrationDomain> edges;
    std::vector<IntegrationDomain> points;
    std::vector<Fold> folds;

    LaminateProperties material;
    Assembly::NonLinearAssembler face_assembler;

    Eigen::VectorXd external_force_cache;
    bool external_force_cache_valid = false;

    double rho;
    int order;

    Meshless(int reproducibility_order) : order(reproducibility_order) {}

    void addFold(int edge_idx, double Kt, double Ang0, bool rigid, double sign = 1.0) {
        Fold f;
        f.Kt = Kt;
        f.Ang0 = Ang0;
        f.flagRig = rigid;
        f.fold_sign = sign;

        const double s = 1e-7;

        f.left = edges[edge_idx];
        f.right = edges[edge_idx];

        for (auto& gp : f.left.gps) gp.X_ref += s * gp.normal;
        for (auto& gp : f.right.gps) gp.X_ref -= s * gp.normal;

        folds.push_back(f);
    }

    // Include the (theta - theta0) * d2(theta)/du2 term in the hinge tangent.
    // Off by default: the rank-one term alone is what published results used.
    bool fold_exact_tangent = false;

    void initAssembly() {
        face_assembler.initSparsityPattern(static_cast<int>(nodes.size()), faces.gps);
    }

    void updateKinematics(const Eigen::VectorXd& x) {
        for (size_t i = 0; i < nodes.size(); ++i) nodes[i].u = x.segment<3>(i * 3);
    }

    Eigen::VectorXd computeExternalForce(const Eigen::VectorXd& x) {
        (void)x;
        const int ndofs = static_cast<int>(nodes.size()) * 3;
        if (external_force_cache_valid && external_force_cache.size() == ndofs) return external_force_cache;

        Eigen::VectorXd F_ext = Eigen::VectorXd::Zero(ndofs);

        auto accumulate_force = [&](const IntegrationDomain& dom) {
            if (dom.bc.force.squaredNorm() <= 0.0) return;
            const Eigen::Vector3d weighted_force = dom.bc.force;
            for (const auto& gp : dom.gps) {
                const double W = gp.weight;
                for (size_t i = 0; i < gp.neighbor_node_ids.size(); ++i) {
                    F_ext.segment<3>(gp.neighbor_node_ids[i] * 3).noalias() += (gp.Psi[i] * W) * weighted_force;
                }
            }
        };

        // Edge moment: F_ext(w_I) += weight * moment * (grad Psi_I . n_out). The
        // stored edge normal is the tangent turned by +90 deg, inward or outward
        // depending on how the mesh lists the edge; the neighbour nodes all lie
        // on the inner side, which fixes the outward sense.
        auto accumulate_moment = [&](const IntegrationDomain& dom) {
            if (dom.bc.moment == 0.0) return;
            for (const auto& gp : dom.gps) {
                double side = 0.0;
                for (int id : gp.neighbor_node_ids) side += (nodes[id].X_ref - gp.X_ref).dot(gp.normal);
                const Eigen::Vector2d n_out = (side > 0.0) ? Eigen::Vector2d(-gp.normal) : gp.normal;
                for (size_t i = 0; i < gp.neighbor_node_ids.size(); ++i)
                    F_ext(gp.neighbor_node_ids[i] * 3 + 2) += gp.weight * dom.bc.moment * gp.dPsi[i].dot(n_out);
            }
        };

        for (const auto& p : points) accumulate_force(p);
        for (const auto& e : edges)  { accumulate_force(e); accumulate_moment(e); }
        for (const auto& f : folds) { accumulate_force(f.left); accumulate_force(f.right); }

        external_force_cache = F_ext;
        external_force_cache_valid = true;
        return external_force_cache;
    }

    Eigen::VectorXd assembleSystemInternalOnly(const Eigen::VectorXd& x, bool flagRig_current) {
        updateKinematics(x);
        Eigen::VectorXd F_int = Eigen::VectorXd::Zero(nodes.size() * 3);
        face_assembler.updateStiffness(faces.gps, nodes, material, F_int);
        
        if (!flagRig_current) {
            for (const auto& f : folds) Constraints::evaluateElasticFold(f.left.gps, f.right.gps, nodes, f.Ang0, f.Kt, F_int, nullptr, fold_exact_tangent, &f.gp_theta0, &f.gp_active);
        }
        return F_int;
    }

    void assembleSystem(Eigen::SparseMatrix<double>& K, Eigen::VectorXd& F_int, const Eigen::VectorXd& x, bool flagRig_current) {
        updateKinematics(x);
        face_assembler.updateStiffness(faces.gps, nodes, material, F_int);
        K = face_assembler.K;
        
        if (!flagRig_current) {
            std::vector<Eigen::Triplet<double>> K_fold_trips;
            for (const auto& f : folds) Constraints::evaluateElasticFold(f.left.gps, f.right.gps, nodes, f.Ang0, f.Kt, F_int, &K_fold_trips, fold_exact_tangent, &f.gp_theta0, &f.gp_active);
            if (!K_fold_trips.empty()) {
                Eigen::SparseMatrix<double> K_fold(K.rows(), K.cols());
                K_fold.setFromTriplets(K_fold_trips.begin(), K_fold_trips.end());
                K += K_fold;
            }
        }
    }

    void evaluateConstraints(const Eigen::VectorXd& x, Eigen::VectorXd& c_all, Eigen::SparseMatrix<double>& J_all, bool flagRig_current,
                             Eigen::VectorXd* dcdtheta_all = nullptr,
                             const std::vector<double>* W_folds = nullptr,
                             std::vector<std::vector<int>>* fold_angle_rows = nullptr)
    {
        updateKinematics(x);
        std::vector<double> c_vec;
        std::vector<Eigen::Triplet<double>> J_triplets;
        std::vector<double> dcdtheta_vec;

        size_t estimated_rows = 0;
        size_t estimated_nnz = 0;
        auto estimate_dirichlet = [&](const IntegrationDomain& dom) {
            const int active = static_cast<int>(dom.bc.active_u) + static_cast<int>(dom.bc.active_v)
                             + static_cast<int>(dom.bc.active_w) + static_cast<int>(dom.bc.active_rot);
            if (active == 0) return;
            estimated_rows += static_cast<size_t>(active) * dom.gps.size();
            for (const auto& gp : dom.gps) estimated_nnz += static_cast<size_t>(active) * gp.neighbor_node_ids.size();
        };
        for (const auto& p : points) estimate_dirichlet(p);
        for (const auto& e : edges)  estimate_dirichlet(e);
        for (const auto& f : folds) {
            estimated_rows += static_cast<size_t>(flagRig_current ? 4 : 3) * f.left.gps.size();
            for (const auto& gp : f.left.gps) estimated_nnz += 3 * gp.neighbor_node_ids.size();
            for (const auto& gp : f.right.gps) estimated_nnz += 3 * gp.neighbor_node_ids.size();
            if (flagRig_current) {
                for (const auto& gp : f.left.gps) estimated_nnz += 3 * gp.neighbor_node_ids.size();
                for (const auto& gp : f.right.gps) estimated_nnz += 3 * gp.neighbor_node_ids.size();
            }
        }
        c_vec.reserve(estimated_rows);
        J_triplets.reserve(estimated_nnz);
        if (dcdtheta_all) dcdtheta_vec.reserve(estimated_rows);

        int row_ptr = 0;

        for (const auto& p : points)
            if (p.bc.active_u || p.bc.active_v || p.bc.active_w || p.bc.active_rot)
                Constraints::evaluateDirichlet(p.gps, nodes, p.bc, row_ptr, c_vec, J_triplets, dcdtheta_all ? &dcdtheta_vec : nullptr);

        for (const auto& e : edges)
            if (e.bc.active_u || e.bc.active_v || e.bc.active_w || e.bc.active_rot)
                Constraints::evaluateDirichlet(e.gps, nodes, e.bc, row_ptr, c_vec, J_triplets, dcdtheta_all ? &dcdtheta_vec : nullptr);

        if (fold_angle_rows) fold_angle_rows->resize(folds.size());

        for (size_t f_idx = 0; f_idx < folds.size(); ++f_idx) {
            const auto& f = folds[f_idx];
            if (fold_angle_rows) (*fold_angle_rows)[f_idx].clear();

            double weight = (W_folds && f_idx < W_folds->size()) ? (*W_folds)[f_idx] : 1.0;
            double dtheta_dlambda = f.fold_sign * (f.angle_target - f.angle_start) * weight;

            Constraints::evaluateFold(f.left.gps, f.right.gps, nodes, f.Ang0, flagRig_current,
                                      row_ptr, c_vec, J_triplets,
                                      dcdtheta_all ? &dcdtheta_vec : nullptr,
                                      dtheta_dlambda,
                                      fold_angle_rows ? &(*fold_angle_rows)[f_idx] : nullptr,
                                      &f.gp_theta0, &f.gp_active, &f.tie);
        }

        c_all = Eigen::VectorXd::Map(c_vec.data(), c_vec.size());
        J_all.resize(row_ptr, nodes.size() * 3);
        J_all.setFromTriplets(J_triplets.begin(), J_triplets.end());

        if (dcdtheta_all) *dcdtheta_all = Eigen::VectorXd::Map(dcdtheta_vec.data(), dcdtheta_vec.size());
    }
    
    double computeTotalHingeEnergy(const Eigen::VectorXd& x) {
        updateKinematics(x);
        double E_hin = 0.0;
        for (const auto& f : folds) E_hin += Constraints::computeHingeEnergy(f.left.gps, f.right.gps, nodes, f.Ang0, f.Kt, &f.gp_theta0, &f.gp_active);
        return E_hin;
    }

    void computeStrainEnergies(const Eigen::VectorXd& x, double& E_mem, double& E_ben) {
        updateKinematics(x);
        E_mem = 0.0; E_ben = 0.0;
        
        for (const auto& gp : faces.gps) {
            auto kin = Kinematics::compute(gp, nodes);
            Eigen::Vector3d e0, e1;
            e0 << kin.E0(0,0), kin.E0(1,1), 2.0 * kin.E0(0,1);
            e1 << kin.E1_el(0,0), kin.E1_el(1,1), 2.0 * kin.E1_el(0,1);
            
            Constitutive::Resultants res;
            Constitutive::evaluate(kin, material, res);
            
            E_mem += 0.5 * gp.weight * e0.dot(res.N);
            E_ben += 0.5 * gp.weight * e1.dot(res.M);
        }
    }
};
