#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <string>
#include <omp.h>
#include "curvami/MeshfreeData.h"
#include "curvami/Enrichment.h"

namespace GlobalRKPM {

    inline double w(double s) { if (s >= 1.0) return 0.0; double s2 = s*s; return 1.0-6.0*s2+8.0*s2*s-3.0*s2*s2; }
    inline double dw(double s) { if (s>=1.0||s<1e-14) return 0.0; return -12.0*s+24.0*s*s-12.0*s*s*s; }
    inline double d2w(double s) { if (s>=1.0) return 0.0; return -12.0+48.0*s-36.0*s*s; }

    inline void solve(std::vector<GaussPointData>& gps, const std::vector<Node>& nodes,
                      const std::vector<size_t>& ig, const std::vector<size_t>& js, const std::vector<double>& s,
                      double rho, int order, const std::vector<Enrichment::FoldLine>& creases)
    {
        assert(order >= 1 && "RKPM Order must be at least 1");

        const size_t nnz = ig.size();
        const size_t ng  = gps.size();
        const size_t np  = static_cast<size_t>((order + 1) * (order + 2) / 2);

        // Symmetric packed moment-index map.
        std::vector<std::vector<size_t>> K(np, std::vector<size_t>(np));
        size_t nm = 0;
        for (size_t i = 0; i < np; ++i) {
            for (size_t j = i; j < np; ++j) {
                K[i][j] = K[j][i] = nm++;
            }
        }

        // Group support entries by Gauss point once. This removes the old final O(ng*nnz)
        // scan and lets the moment matrices be assembled without atomics.
        std::vector<std::vector<size_t>> entries_by_gp(ng);
        for (size_t i = 0; i < nnz; ++i) {
            entries_by_gp[ig[i]].push_back(i);
        }

        // Keep shape-function arrays consistent if solve() is called more than once.
        #pragma omp parallel for schedule(static)
        for (long long gg = 0; gg < static_cast<long long>(ng); ++gg) {
            auto& gp = gps[static_cast<size_t>(gg)];
            gp.Psi.clear();
            gp.dPsi.clear();
            gp.ddPsi.clear();
            gp.Psi.reserve(entries_by_gp[static_cast<size_t>(gg)].size());
            gp.dPsi.reserve(entries_by_gp[static_cast<size_t>(gg)].size());
            gp.ddPsi.reserve(entries_by_gp[static_cast<size_t>(gg)].size());
        }

        Eigen::VectorXd PSI(nnz), dPx(nnz), dPy(nnz), d2Pxx(nnz), d2Pyy(nnz), d2Pxy(nnz), CSI(nnz), ETA(nnz);

        #pragma omp parallel for schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(nnz); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            const double xi  = (gps[ig[i]].X_ref.x() - nodes[js[i]].X_ref.x()) / rho;
            const double eta = (gps[ig[i]].X_ref.y() - nodes[js[i]].X_ref.y()) / rho;
            CSI(i) = xi;
            ETA(i) = eta;

            const double si = s[i];
            const double wi = w(si);
            const double d  = dw(si);
            const double dd = d2w(si);
            const double dx = (si == 0.0) ? 0.0 : xi / si;
            const double dy = (si == 0.0) ? 0.0 : eta / si;

            PSI(i) = wi;
            dPx(i) = d * dx;
            dPy(i) = d * dy;

            const double si3 = si * si * si;
            const double d2x = (si == 0.0) ? dd : (xi * xi) / si3;
            const double d2y = (si == 0.0) ? dd : (eta * eta) / si3;
            d2Pxx(i) = dd * dx * dx + d * d2x;
            d2Pyy(i) = dd * dy * dy + d * d2y;
            d2Pxy(i) = (si == 0.0) ? 0.0 : dd * dx * dy + d * (-(xi * eta) / si3);
        }

        // Apply crease visibility. This remains per interaction, but without shared writes.
        #pragma omp parallel for schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(nnz); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            for (const auto& crease : creases) {
                auto iG = Enrichment::evaluateIndicator(gps[ig[i]].X_ref, crease);
                auto iN = Enrichment::evaluateIndicator(nodes[js[i]].X_ref, crease);
                const double sn = (iN.f >= 0.0) ? 1.0 : -1.0;
                const double W  = 0.5 * (1.0 + sn * iG.f);
                const Eigen::Vector2d dW = 0.5 * sn * iG.grad_f * rho;
                const Eigen::Matrix2d ddW = 0.5 * sn * iG.hess_f * rho * rho;

                const double oW = PSI(i);
                const double dPx_old = dPx(i);
                const double dPy_old = dPy(i);
                const double d2Pxx_old = d2Pxx(i);
                const double d2Pyy_old = d2Pyy(i);
                const double d2Pxy_old = d2Pxy(i);

                PSI(i) *= W;
                dPx(i) = dPx_old * W + oW * dW.x();
                dPy(i) = dPy_old * W + oW * dW.y();
                d2Pxx(i) = d2Pxx_old * W + 2.0 * dPx_old * dW.x() + oW * ddW(0,0);
                d2Pyy(i) = d2Pyy_old * W + 2.0 * dPy_old * dW.y() + oW * ddW(1,1);
                d2Pxy(i) = d2Pxy_old * W + dPx_old * dW.y() + dPy_old * dW.x() + oW * ddW(0,1);
            }
        }

        // Polynomial basis and derivatives. Avoid std::pow in the innermost loop.
        Eigen::MatrixXd P       = Eigen::MatrixXd::Zero(np, nnz);
        Eigen::MatrixXd dPdX    = Eigen::MatrixXd::Zero(np, nnz);
        Eigen::MatrixXd dPdY    = Eigen::MatrixXd::Zero(np, nnz);
        Eigen::MatrixXd d2PdX2  = Eigen::MatrixXd::Zero(np, nnz);
        Eigen::MatrixXd d2PdY2  = Eigen::MatrixXd::Zero(np, nnz);
        Eigen::MatrixXd d2PdXDY = Eigen::MatrixXd::Zero(np, nnz);

        #pragma omp parallel for schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(nnz); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            const double x = CSI(i);
            const double y = ETA(i);

            std::vector<double> xp(static_cast<size_t>(order) + 1), yp(static_cast<size_t>(order) + 1);
            xp[0] = 1.0;
            yp[0] = 1.0;
            for (int p = 1; p <= order; ++p) {
                xp[static_cast<size_t>(p)] = xp[static_cast<size_t>(p - 1)] * x;
                yp[static_cast<size_t>(p)] = yp[static_cast<size_t>(p - 1)] * y;
            }

            size_t idx = 0;
            for (int d = 0; d <= order; ++d) {
                for (int p_y = 0; p_y <= d; ++p_y) {
                    const int p_x = d - p_y;
                    const size_t sx = static_cast<size_t>(p_x);
                    const size_t sy = static_cast<size_t>(p_y);

                    P(idx, i) = xp[sx] * yp[sy];
                    if (p_x > 0) dPdX(idx, i) = static_cast<double>(p_x) * xp[sx - 1] * yp[sy];
                    if (p_y > 0) dPdY(idx, i) = static_cast<double>(p_y) * xp[sx] * yp[sy - 1];
                    if (p_x > 1) d2PdX2(idx, i) = static_cast<double>(p_x * (p_x - 1)) * xp[sx - 2] * yp[sy];
                    if (p_y > 1) d2PdY2(idx, i) = static_cast<double>(p_y * (p_y - 1)) * xp[sx] * yp[sy - 2];
                    if (p_x > 0 && p_y > 0) d2PdXDY(idx, i) = static_cast<double>(p_x * p_y) * xp[sx - 1] * yp[sy - 1];
                    ++idx;
                }
            }
        }

        Eigen::MatrixXd M       = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd dMx     = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd dMy     = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd d2Mxx   = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd d2Myy   = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd d2Mxy   = Eigen::MatrixXd::Zero(nm, ng);

        // Assemble by Gauss point: no atomic operations, better cache locality.
        #pragma omp parallel for schedule(static)
        for (long long gg = 0; gg < static_cast<long long>(ng); ++gg) {
            const size_t g = static_cast<size_t>(gg);
            for (const size_t i : entries_by_gp[g]) {
                for (size_t ip = 0; ip < np; ++ip) {
                    for (size_t jp = ip; jp < np; ++jp) {
                        const size_t k = K[ip][jp];
                        const double pi   = P(ip,i);
                        const double pj   = P(jp,i);
                        const double dpix = dPdX(ip,i);
                        const double dpjx = dPdX(jp,i);
                        const double dpiy = dPdY(ip,i);
                        const double dpjy = dPdY(jp,i);

                        M(k,g)     += pi * pj * PSI(i);
                        dMx(k,g)   += pi * pj * dPx(i) + (dpix * pj + pi * dpjx) * PSI(i);
                        dMy(k,g)   += pi * pj * dPy(i) + (dpiy * pj + pi * dpjy) * PSI(i);
                        d2Mxx(k,g) += d2Pxx(i) * pi * pj + 2.0 * dPx(i) * (dpix * pj + pi * dpjx)
                                      + PSI(i) * (d2PdX2(ip,i) * pj + 2.0 * dpix * dpjx + pi * d2PdX2(jp,i));
                        d2Myy(k,g) += d2Pyy(i) * pi * pj + 2.0 * dPy(i) * (dpiy * pj + pi * dpjy)
                                      + PSI(i) * (d2PdY2(ip,i) * pj + 2.0 * dpiy * dpjy + pi * d2PdY2(jp,i));
                        d2Mxy(k,g) += d2Pxy(i) * pi * pj + dPx(i) * (dpiy * pj + pi * dpjy)
                                      + dPy(i) * (dpix * pj + pi * dpjx)
                                      + PSI(i) * (d2PdXDY(ip,i) * pj + dpix * dpjy + dpiy * dpjx + pi * d2PdXDY(jp,i));
                    }
                }
            }
        }

        Eigen::MatrixXd Minv      = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd dMinvx    = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd dMinvy    = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd d2Minvxx  = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd d2Minvyy  = Eigen::MatrixXd::Zero(nm, ng);
        Eigen::MatrixXd d2Minvxy  = Eigen::MatrixXd::Zero(nm, ng);

        // A degenerate support makes the moment matrix singular and m3.inverse()
        // then yields inf/NaN that propagates silently into the shape functions.
        // Record per-Gauss-point rather than throwing inside the parallel region.
        std::vector<char> gp_bad(ng, 0);

        #pragma omp parallel for schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(ng); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            Eigen::Matrix3d m3, dmx3, dmy3, d2mxx3, d2myy3, d2mxy3;
            auto fill3 = [&](const Eigen::MatrixXd& Src, Eigen::Matrix3d& Dst) {
                Dst << Src(K[0][0],i), Src(K[0][1],i), Src(K[0][2],i),
                       Src(K[0][1],i), Src(K[1][1],i), Src(K[1][2],i),
                       Src(K[0][2],i), Src(K[1][2],i), Src(K[2][2],i);
            };
            fill3(M, m3); fill3(dMx, dmx3); fill3(dMy, dmy3); fill3(d2Mxx, d2mxx3); fill3(d2Myy, d2myy3); fill3(d2Mxy, d2mxy3);

            const Eigen::Matrix3d inv   = m3.inverse();

            // Direct residual test: self-scaling and unambiguous for 3x3.
            if (!inv.allFinite() ||
                (m3 * inv - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-8) {
                gp_bad[i] = 1;
            }

            const Eigen::Matrix3d invx  = -inv * dmx3 * inv;
            const Eigen::Matrix3d invy  = -inv * dmy3 * inv;
            const Eigen::Matrix3d invxx = inv * (2.0 * dmx3 * inv * dmx3 - d2mxx3) * inv;
            const Eigen::Matrix3d invyy = inv * (2.0 * dmy3 * inv * dmy3 - d2myy3) * inv;
            const Eigen::Matrix3d invxy = inv * (dmx3 * inv * dmy3 + dmy3 * inv * dmx3 - d2mxy3) * inv;

            for (int r = 0; r < 3; ++r) {
                for (int c = r; c < 3; ++c) {
                    const size_t k = K[static_cast<size_t>(r)][static_cast<size_t>(c)];
                    Minv(k,i) = inv(r,c);
                    dMinvx(k,i) = invx(r,c);
                    dMinvy(k,i) = invy(r,c);
                    d2Minvxx(k,i) = invxx(r,c);
                    d2Minvyy(k,i) = invyy(r,c);
                    d2Minvxy(k,i) = invxy(r,c);
                }
            }
        }

        {
            size_t nbad = 0, first = 0;
            for (size_t i = 0; i < ng; ++i)
                if (gp_bad[i]) { if (!nbad) first = i; ++nbad; }
            if (nbad) {
                throw std::runtime_error(
                    "RKPM: singular moment matrix at " + std::to_string(nbad) + " of "
                    + std::to_string(ng) + " Gauss points (first is index "
                    + std::to_string(first) + " with "
                    + std::to_string(gps[first].neighbor_node_ids.size())
                    + " neighbours). The support is rank-deficient for the requested "
                      "basis order - typically too few visible nodes, which happens "
                      "where a crease cuts most of a support away. Increase the support "
                      "radius, refine the mesh there, or lower the order.");
            }
        }

        // Sequential block inversion in basis dimension; each Gauss point is independent.
        for (size_t iter = 3; iter < np; ++iter) {
            #pragma omp parallel for schedule(static)
            for (long long ii = 0; ii < static_cast<long long>(ng); ++ii) {
                const size_t i = static_cast<size_t>(ii);
                Eigen::VectorXd b(iter), dbx(iter), dby(iter), d2bxx(iter), d2byy(iter), d2bxy(iter);
                for (size_t j = 0; j < iter; ++j) {
                    b(j)     = M(K[j][iter], i);
                    dbx(j)   = dMx(K[j][iter], i);
                    dby(j)   = dMy(K[j][iter], i);
                    d2bxx(j) = d2Mxx(K[j][iter], i);
                    d2byy(j) = d2Myy(K[j][iter], i);
                    d2bxy(j) = d2Mxy(K[j][iter], i);
                }

                Eigen::MatrixXd aI(iter, iter), daIx(iter, iter), daIy(iter, iter), d2Axx(iter, iter), d2Ayy(iter, iter), d2Axy(iter, iter);
                for (size_t r = 0; r < iter; ++r) {
                    for (size_t c = 0; c < iter; ++c) {
                        const size_t k = K[r][c];
                        aI(r,c) = Minv(k,i);
                        daIx(r,c) = dMinvx(k,i);
                        daIy(r,c) = dMinvy(k,i);
                        d2Axx(r,c) = d2Minvxx(k,i);
                        d2Ayy(r,c) = d2Minvyy(k,i);
                        d2Axy(r,c) = d2Minvxy(k,i);
                    }
                }

                const Eigen::VectorXd v   = aI * b;
                const Eigen::VectorXd vx  = daIx * b + aI * dbx;
                const Eigen::VectorXd vy  = daIy * b + aI * dby;
                const Eigen::VectorXd vxx = d2Axx * b + 2.0 * daIx * dbx + aI * d2bxx;
                const Eigen::VectorXd vyy = d2Ayy * b + 2.0 * daIy * dby + aI * d2byy;
                const Eigen::VectorXd vxy = d2Axy * b + daIx * dby + daIy * dbx + aI * d2bxy;

                const size_t kii = K[iter][iter];
                const double q   = M(kii,i)     - b.dot(v);
                const double qx  = dMx(kii,i)   - (dbx.dot(v) + b.dot(vx));
                const double qy  = dMy(kii,i)   - (dby.dot(v) + b.dot(vy));
                const double qxx = d2Mxx(kii,i) - (d2bxx.dot(v) + 2.0 * dbx.dot(vx) + b.dot(vxx));
                const double qyy = d2Myy(kii,i) - (d2byy.dot(v) + 2.0 * dby.dot(vy) + b.dot(vyy));
                const double qxy = d2Mxy(kii,i) - (d2bxy.dot(v) + dbx.dot(vy) + dby.dot(vx) + b.dot(vxy));

                // q is the Schur complement; if it collapses, invQ is inf and NaN
                // spreads through the whole inverse.
                if (!(std::abs(q) > 1e-12 * std::max(std::abs(M(kii,i)), 1.0))) {
                    gp_bad[i] = 2;
                }

                const double invQ  = 1.0 / q;
                const double invQ2 = invQ * invQ;
                const double invQ3 = invQ2 * invQ;

                for (size_t r = 0; r < iter; ++r) {
                    for (size_t c = r; c < iter; ++c) {
                        const size_t k = K[r][c];
                        Minv(k,i) += v(r) * v(c) * invQ;
                        dMinvx(k,i) += (vx(r) * v(c) + v(r) * vx(c)) * invQ - v(r) * v(c) * qx * invQ2;
                        dMinvy(k,i) += (vy(r) * v(c) + v(r) * vy(c)) * invQ - v(r) * v(c) * qy * invQ2;
                        d2Minvxx(k,i) += (vxx(r) * v(c) + 2.0 * vx(r) * vx(c) + v(r) * vxx(c)) * invQ
                                          - (vx(r) * v(c) + v(r) * vx(c)) * qx * invQ2 * 2.0
                                          + v(r) * v(c) * (2.0 * qx * qx * invQ3 - qxx * invQ2);
                        d2Minvyy(k,i) += (vyy(r) * v(c) + 2.0 * vy(r) * vy(c) + v(r) * vyy(c)) * invQ
                                          - (vy(r) * v(c) + v(r) * vy(c)) * qy * invQ2 * 2.0
                                          + v(r) * v(c) * (2.0 * qy * qy * invQ3 - qyy * invQ2);
                        d2Minvxy(k,i) += (vxy(r) * v(c) + vx(r) * vy(c) + vy(r) * vx(c) + v(r) * vxy(c)) * invQ
                                          - (vx(r) * v(c) + v(r) * vx(c)) * qy * invQ2
                                          - (vy(r) * v(c) + v(r) * vy(c)) * qx * invQ2
                                          + v(r) * v(c) * (2.0 * qx * qy * invQ3 - qxy * invQ2);
                    }
                }
                for (size_t j = 0; j < iter; ++j) {
                    const size_t k = K[j][iter];
                    Minv(k,i) = -v(j) * invQ;
                    dMinvx(k,i) = -vx(j) * invQ + v(j) * qx * invQ2;
                    dMinvy(k,i) = -vy(j) * invQ + v(j) * qy * invQ2;
                    d2Minvxx(k,i) = -vxx(j) * invQ + 2.0 * vx(j) * qx * invQ2 + v(j) * (qxx * invQ2 - 2.0 * qx * qx * invQ3);
                    d2Minvyy(k,i) = -vyy(j) * invQ + 2.0 * vy(j) * qy * invQ2 + v(j) * (qyy * invQ2 - 2.0 * qy * qy * invQ3);
                    d2Minvxy(k,i) = -vxy(j) * invQ + vx(j) * qy * invQ2 + vy(j) * qx * invQ2 + v(j) * (qxy * invQ2 - 2.0 * qx * qy * invQ3);
                }
                Minv(kii,i) = invQ;
                dMinvx(kii,i) = -qx * invQ2;
                dMinvy(kii,i) = -qy * invQ2;
                d2Minvxx(kii,i) = 2.0 * qx * qx * invQ3 - qxx * invQ2;
                d2Minvyy(kii,i) = 2.0 * qy * qy * invQ3 - qyy * invQ2;
                d2Minvxy(kii,i) = 2.0 * qx * qy * invQ3 - qxy * invQ2;
            }
        }

        {
            size_t nbad = 0, first = 0;
            for (size_t i = 0; i < ng; ++i)
                if (gp_bad[i] == 2) { if (!nbad) first = i; ++nbad; }
            if (nbad) {
                throw std::runtime_error(
                    "RKPM: moment-matrix pivot collapsed during block inversion at "
                    + std::to_string(nbad) + " of " + std::to_string(ng)
                    + " Gauss points (first is index " + std::to_string(first)
                    + " with " + std::to_string(gps[first].neighbor_node_ids.size())
                    + " neighbours). Same cause and remedies as a singular moment matrix.");
            }
        }

        // Final shape-function assembly by Gauss point. Same ordering as the FRNN entries,
        // but without scanning every nnz for every Gauss point.
        #pragma omp parallel for schedule(static)
        for (long long gg = 0; gg < static_cast<long long>(ng); ++gg) {
            const size_t g = static_cast<size_t>(gg);
            auto& gp = gps[g];

            for (const size_t i : entries_by_gp[g]) {
                double C = 0.0, dCx = 0.0, dCy = 0.0, d2Cxx = 0.0, d2Cyy = 0.0, d2Cxy = 0.0;
                for (size_t jp = 0; jp < np; ++jp) {
                    const size_t k = K[0][jp];
                    const double v   = Minv(k,g);
                    const double vx  = dMinvx(k,g);
                    const double vy  = dMinvy(k,g);
                    const double vxx = d2Minvxx(k,g);
                    const double vyy = d2Minvyy(k,g);
                    const double vxy = d2Minvxy(k,g);

                    C     += v * P(jp,i);
                    dCx   += vx * P(jp,i) + v * dPdX(jp,i);
                    dCy   += vy * P(jp,i) + v * dPdY(jp,i);
                    d2Cxx += vxx * P(jp,i) + 2.0 * vx * dPdX(jp,i) + v * d2PdX2(jp,i);
                    d2Cyy += vyy * P(jp,i) + 2.0 * vy * dPdY(jp,i) + v * d2PdY2(jp,i);
                    d2Cxy += vxy * P(jp,i) + vx * dPdY(jp,i) + vy * dPdX(jp,i) + v * d2PdXDY(jp,i);
                }

                gp.Psi.push_back(C * PSI(i));
                gp.dPsi.push_back(Eigen::Vector2d((dCx * PSI(i) + C * dPx(i)) / rho,
                                                   (dCy * PSI(i) + C * dPy(i)) / rho));
                Eigen::Matrix2d dd;
                dd(0,0) = (d2Cxx * PSI(i) + 2.0 * dCx * dPx(i) + C * d2Pxx(i)) / (rho * rho);
                dd(1,1) = (d2Cyy * PSI(i) + 2.0 * dCy * dPy(i) + C * d2Pyy(i)) / (rho * rho);
                dd(0,1) = dd(1,0) = (d2Cxy * PSI(i) + dCx * dPy(i) + dCy * dPx(i) + C * d2Pxy(i)) / (rho * rho);
                gp.ddPsi.push_back(dd);
            }
        }
    }
}
