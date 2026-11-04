#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <algorithm>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "PlateData.h"
#include "Kinematics.h"
#include "Constitutive.h"

namespace Assembly {

    typedef Eigen::Triplet<double> T;

    class NonLinearAssembler {
    private:
        int num_dofs;
        // Stores the 1D CSR index for every interaction per Gauss point
        std::vector<std::vector<int>> gp_csr_mapping;

        // Reused Newton-iteration scratch. This avoids repeated allocation of
        // max_threads full-size vectors at every stiffness assembly.
        std::vector<Eigen::VectorXd> scratch_forces;
        std::vector<std::vector<double>> scratch_K_vals;

        // Per-thread per-Gauss-point workspace. Without this, every Gauss point
        // heap-allocates its neighbour arrays (~100 neighbours x several arrays,
        // 12960 points per assembly); reusing the buffers keeps the capacity
        // after the first point and removes the allocator from the hot loop.
        struct GpScratch {
            std::vector<Eigen::Matrix3d> Bm, Bb, P, Q, Db;
            std::vector<Eigen::Vector2d> N2dPsi, C0invdPsi;
            std::vector<double> Total_u;

            void ensure(size_t n) {
                if (Bm.size() >= n) return;
                Bm.resize(n); Bb.resize(n); P.resize(n); Q.resize(n); Db.resize(n);
                N2dPsi.resize(n); C0invdPsi.resize(n); Total_u.resize(n);
            }
        };
        std::vector<GpScratch> scratch_gp;

    public:
        Eigen::SparseMatrix<double> K;

        // Read-only view of one Gauss point's CSR map: for each local pair (i <= j)
        // of its neighbours, in that loop order, the position of node min(ni, nj)
        // among the block rows of node max(ni, nj)'s columns (see
        // initSparsityPattern). Lets another node-pair matrix with K's pattern be
        // filled in place (the progressive-creasing re-fit).
        const std::vector<int>& csrMapping(size_t gp_idx) const { return gp_csr_mapping[gp_idx]; }

        // ====================================================================
        // PHASE 1: SYMBOLIC INITIALIZATION (Call exactly ONCE)
        // ====================================================================
        void initSparsityPattern(int num_nodes, const std::vector<GaussPointData>& gps) {
            num_dofs = num_nodes * 3;
            
            // 1. Build unique nodal connectivity (Adjacency List)
            std::vector<std::vector<int>> col_nodes(num_nodes);
            // K is symmetric (verified to 1.1e-17 relative, including the fold and
            // hinge terms), so only the block-upper half is stored: node blocks with
            // ni <= nj. Consumers read it through selfadjointView<Upper>. This halves
            // the value array, the per-thread scratch, the CSR mapping table and the
            // number of scattered writes in the assembly.
            for (const auto& gp : gps) {
                for (int ni : gp.neighbor_node_ids) {
                    for (int nj : gp.neighbor_node_ids) {
                        if (ni <= nj) col_nodes[nj].push_back(ni);
                    }
                }
            }
            
            int nnz_estimate = 0;
            for (int j = 0; j < num_nodes; ++j) {
                std::sort(col_nodes[j].begin(), col_nodes[j].end());
                col_nodes[j].erase(std::unique(col_nodes[j].begin(), col_nodes[j].end()), col_nodes[j].end());
                nnz_estimate += col_nodes[j].size() * 9;
            }

            // 2. Generate exactly ONE Triplet per structural non-zero (Already Sorted!)
            std::vector<T> dummy_triplets;
            dummy_triplets.reserve(nnz_estimate);
            for (int j = 0; j < num_nodes; ++j) {
                for (int c = 0; c < 3; ++c) {
                    for (int i : col_nodes[j]) {
                        for (int r = 0; r < 3; ++r) {
                            dummy_triplets.push_back(T(i * 3 + r, j * 3 + c, 0.0));
                        }
                    }
                }
            }

            // Lock the Sparsity Pattern in memory instantly
            K.resize(num_dofs, num_dofs);
            K.setFromTriplets(dummy_triplets.begin(), dummy_triplets.end());

            // 3. MULTI-THREADED CSR MAPPING (OpenMP)
            //
            // Column node_j*3+c holds the rows of col_nodes[node_j], each expanded
            // into 3 consecutive entries, in ascending order. So a block's CSR
            // location is fixed by the position of node_i inside col_nodes[node_j]:
            //
            //     index(i,j,r,c) = outerIndexPtr[node_j*3 + c] + 3*pos + r
            //
            // Storing that single position per block instead of all 9 explicit
            // indices cuts this table by 9x (it is the largest allocation in the
            // program: sum(n^2)*9 ints, ~4.5 GB on OneCurve).
            gp_csr_mapping.resize(gps.size());

            int num_points = gps.size();
            bool mapping_ok = true;

            #pragma omp parallel for
            for (int gp_idx = 0; gp_idx < num_points; ++gp_idx) {
                const auto& gp = gps[gp_idx];
                const size_t n = gp.neighbor_node_ids.size();
                auto& map = gp_csr_mapping[gp_idx];
                map.resize(n * (n + 1) / 2);

                // One entry per local pair (i <= j), consumed in the same order by the
                // assembly. Each records where the canonical block (min, max) of the
                // two node indices lives; note local order does not imply node order,
                // since the neighbour list is not sorted.
                size_t mo = 0;
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = i; j < n; ++j) {
                        const int ni = gp.neighbor_node_ids[i];
                        const int nj = gp.neighbor_node_ids[j];
                        const int lo = std::min(ni, nj), hi = std::max(ni, nj);
                        const auto& ch = col_nodes[hi];
                        auto it = std::lower_bound(ch.begin(), ch.end(), lo);
                        if (it == ch.end() || *it != lo) {
                            map[mo++] = -1;
                            mapping_ok = false;   // benign race: only ever set false
                        } else {
                            map[mo++] = static_cast<int>(it - ch.begin());
                        }
                    }
                }
            }

            // Previously a miss silently produced index -1, which the assembly loop
            // then used as an array subscript. Fail loudly here instead.
            if (!mapping_ok) {
                throw std::runtime_error(
                    "Assembly: Gauss-point neighbour missing from the sparsity pattern; "
                    "CSR mapping is inconsistent.");
            }
        }

        // ====================================================================
        // PHASE 2: NUMERIC ASSEMBLY (Call EVERY Newton Iteration)
        // ====================================================================
        void updateStiffness(
            const std::vector<GaussPointData>& gps,
            const std::vector<Node>& nodes,
            const LaminateProperties& mat,
            Eigen::VectorXd& F_int)
        {
            // Apple Silicon Thread Configuration
            int max_threads = 1;
            #ifdef _OPENMP
                max_threads = omp_get_max_threads();
            #endif

            const int nnz = static_cast<int>(K.nonZeros());

            if (static_cast<int>(scratch_forces.size()) != max_threads) {
                scratch_forces.assign(max_threads, Eigen::VectorXd::Zero(num_dofs));
            }
            if (static_cast<int>(scratch_K_vals.size()) != max_threads) {
                scratch_K_vals.assign(max_threads, std::vector<double>(nnz, 0.0));
            }
            if (static_cast<int>(scratch_gp.size()) != max_threads) {
                scratch_gp.resize(max_threads);
            }

            // Zeroing max_threads x nnz doubles is hundreds of MB; done serially it
            // is a pure Amdahl tax with every worker idle. Each t touches only its
            // own slot, so this is safe to run in parallel.
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < max_threads; ++t) {
                if (scratch_forces[t].size() != num_dofs) scratch_forces[t].resize(num_dofs);
                scratch_forces[t].setZero();
                if (static_cast<int>(scratch_K_vals[t].size()) != nnz) scratch_K_vals[t].resize(nnz);
                std::fill(scratch_K_vals[t].begin(), scratch_K_vals[t].end(), 0.0);
            }

            auto& local_forces = scratch_forces;
            auto& local_K_vals = scratch_K_vals;

            int num_points = gps.size();

            // OpenMP will dynamically assign GPS chunks to available threads
            #pragma omp parallel for schedule(guided, 8)
            for (int gp_idx = 0; gp_idx < num_points; ++gp_idx) {
                
                int t_id = 0;
                #ifdef _OPENMP
                    t_id = omp_get_thread_num();
                #endif

                const auto& gp = gps[gp_idx];
                if (gp.neighbor_node_ids.empty()) continue;

                auto kin = Kinematics::compute(gp, nodes);
                Constitutive::Resultants res;
                Constitutive::evaluate(kin, mat, res);

                size_t n_neighbors = gp.neighbor_node_ids.size();
                double W = gp.weight;

                Eigen::Matrix<double, 3, 2> V = kin.F0 * kin.C0_inv;
                Eigen::Matrix2d N2x2;
                N2x2 << res.N(0), res.N(2), res.N(2), res.N(1);
                
                Eigen::Vector3d v_curv = kin.r0XX * res.M(0) + 2.0 * kin.r0XY * res.M(2) + kin.r0YY * res.M(1);
                Eigen::Vector2d p_XY = kin.C0_inv * kin.F0.transpose() * v_curv;
                Eigen::Matrix3d bbT = kin.b * kin.b.transpose();
                double b_dot_vcurv = kin.b.dot(v_curv);

                GpScratch& sc = scratch_gp[t_id];
                sc.ensure(n_neighbors);
                auto& Bm = sc.Bm; auto& Bb = sc.Bb; auto& Db = sc.Db;
                auto& P  = sc.P;  auto& Q  = sc.Q;
                auto& N2dPsi = sc.N2dPsi; auto& C0invdPsi = sc.C0invdPsi;
                auto& Total_u = sc.Total_u;

                for (size_t i = 0; i < n_neighbors; ++i) {
                    int node_i = gp.neighbor_node_ids[i];

                    Bm[i].row(0) = kin.F0.col(0).transpose() * gp.dPsi[i].x();
                    Bm[i].row(1) = kin.F0.col(1).transpose() * gp.dPsi[i].y();
                    Bm[i].row(2) = kin.F0.col(0).transpose() * gp.dPsi[i].y() + kin.F0.col(1).transpose() * gp.dPsi[i].x();

                    Eigen::Vector3d gI = V * gp.dPsi[i];
                    double S_XX = kin.r0XX.dot(gI);
                    double S_YY = kin.r0YY.dot(gI);
                    double S_XY = kin.r0XY.dot(gI);

                    Bb[i].row(0) = (-gp.ddPsi[i](0,0) + S_XX) * kin.b.transpose();
                    Bb[i].row(1) = (-gp.ddPsi[i](1,1) + S_YY) * kin.b.transpose();
                    Bb[i].row(2) = 2.0 * (-gp.ddPsi[i](0,1) + S_XY) * kin.b.transpose();

                    Eigen::Vector3d f_local = (Bm[i].transpose() * res.N + Bb[i].transpose() * res.M) * W;
                    local_forces[t_id].segment<3>(node_i * 3) += f_local;

                    // Fold the constitutive products into the two combinations the
                    // inner loop actually consumes, so the 3x3 additions happen
                    // once per neighbour instead of once per (i,j) pair.
                    P[i] = mat.A * Bm[i] + mat.B * Bb[i];
                    Q[i] = mat.B * Bm[i] + mat.D * Bb[i];

                    Db[i] = -V * gp.dPsi[i] * kin.b.transpose();

                    // The inner loop reads these as [j]; computing them per
                    // neighbour here replaces n^2 matrix-vector products with n.
                    N2dPsi[i]    = N2x2 * gp.dPsi[i];
                    C0invdPsi[i] = kin.C0_inv * gp.dPsi[i];

                    double HM_u_I = gp.ddPsi[i](0,0) * res.M(0) + 2.0 * gp.ddPsi[i](0,1) * res.M(2) + gp.ddPsi[i](1,1) * res.M(1);
                    double Gp_I = gp.dPsi[i].dot(p_XY);
                    Total_u[i] = Gp_I - HM_u_I;
                }

                // DIRECT MEMORY INJECTION
                //
                // The element block is symmetric under neighbour exchange:
                // K_IJ(j,i) == K_IJ(i,j)^T (verified numerically to 1.1e-16
                // relative). Only the j >= i half is computed; the mirror block is
                // written as the transpose, halving the block arithmetic.
                const auto& csr_map = gp_csr_mapping[gp_idx];
                const int* outer = K.outerIndexPtr();
                double* K_local = local_K_vals[t_id].data();

                size_t map_off = 0;

                for (size_t i = 0; i < n_neighbors; ++i) {
                    const Eigen::Matrix3d BmiT = Bm[i].transpose();
                    const Eigen::Matrix3d BbiT = Bb[i].transpose();
                    const Eigen::Matrix3d DbiT = Db[i].transpose();
                    const Eigen::Vector2d dPsi_i = gp.dPsi[i];
                    const double Tu_i = Total_u[i];

                    for (size_t j = i; j < n_neighbors; ++j) {
                        Eigen::Matrix3d K_mat = BmiT * P[j] + BbiT * Q[j];

                        double geo_m_scalar = dPsi_i.dot(N2dPsi[j]);
                        Eigen::Matrix3d K_geo_m = Eigen::Matrix3d::Identity() * geo_m_scalar;

                        Eigen::Matrix3d T12 = Tu_i * Db[j] + Total_u[j] * DbiT;

                        double T3_scalar = b_dot_vcurv * dPsi_i.dot(C0invdPsi[j]);
                        Eigen::Matrix3d T3 = T3_scalar * bbT;

                        Eigen::Matrix3d K_IJ = (K_mat + K_geo_m + T12 + T3) * W;

                        // Only the canonical block (min node, max node) is stored, so
                        // there is exactly one write per pair. When the local order is
                        // opposite to the node order the transpose goes in instead.
                        const int ni = gp.neighbor_node_ids[i];
                        const int nj = gp.neighbor_node_ids[j];
                        const int pos3 = 3 * csr_map[map_off++];
                        const bool flip = (ni > nj);
                        const int  hi3  = (flip ? ni : nj) * 3;

                        for (int c = 0; c < 3; ++c) {
                            double* dst = K_local + outer[hi3 + c] + pos3;
                            if (!flip) {
                                dst[0] += K_IJ(0, c);
                                dst[1] += K_IJ(1, c);
                                dst[2] += K_IJ(2, c);
                            } else {
                                dst[0] += K_IJ(c, 0);
                                dst[1] += K_IJ(c, 1);
                                dst[2] += K_IJ(c, 2);
                            }
                        }
                    }
                }
            }

            // --- MERGE PHASE: Collapse the threads directly into Eigen's array ---
            F_int = Eigen::VectorXd::Zero(num_dofs);
            for (int t = 0; t < max_threads; ++t) {
                F_int += local_forces[t];
            }

            // Reducing across threads index-by-index (rather than one full pass per
            // thread) touches each K entry once and parallelizes cleanly. The inner
            // accumulation order is unchanged, so results are bit-identical.
            double* K_val_ptr = K.valuePtr();
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < nnz; ++idx) {
                double s = 0.0;
                for (int t = 0; t < max_threads; ++t) {
                    s += local_K_vals[t][idx];
                }
                K_val_ptr[idx] = s;
            }
        }
    };
}
