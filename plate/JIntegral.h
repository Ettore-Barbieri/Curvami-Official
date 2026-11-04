#pragma once

#include <Eigen/Dense>
#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "Meshless.h"
#include "Kinematics.h"
#include "Constitutive.h"
#include "Constraints.h"

// =========================================================
// ENERGY RELEASE RATE OF A CRACK (domain integral)
// =========================================================
// Crack_Brief_for_Code.md, Task A2. Post-processing on a converged state; it
// changes nothing in the solution. For a ring rho1 < rho < rho2 around the tip,
//
//   G_vec = (1/h) * sum_g w_g * B_g * grad q(X_g),
//   B     = C0 N + 2 E1 M - W_h I                       (2x2, not symmetric)
//
// with, at each face Gauss point g (weight w_g = reference area):
//   C0  = F0^T F0            mid-surface metric
//   E1                       bending strain (kin.E1)
//   N, M                     [[v0, v2], [v2, v1]] from the Voigt resultants
//   W_h = 1/2 (N.e0 + M.e1)  area strain energy, Voigt strains with engineering
//                            shear, exactly as Constitutive::evaluate
//   q                        radial weight: 1 for rho <= rho1, 0 for rho >= rho2;
//                            between, with t = (rho - rho1)/(rho2 - rho1):
//                            cubic 1 - 3t^2 + 2t^3 (default) or linear 1 - t
//   grad q = (dq/drho) (X - X_tip)/rho
// h is the plate thickness. C0 N and E1 M are plain matrix products, in that order.
//
// Reported per ring: G_vec in the global frame; its components in the crack frame,
// G1 = d . G_vec (along the crack, the energy release rate for straight advance)
// and G2 = d_perp . G_vec with d_perp = d rotated by +90 deg; |G_vec|; its angle
// from d. Also abs_sum = (1/h) sum_g w_g |B_g grad q|: the size of what cancels.
// With no tip inside the ring, G_vec is zero up to the discretisation error;
// G / abs_sum says how well.
//
// Quadrature of grad q. Exactly, the integral of grad q over the ring is zero, so
// adding any constant tensor to B changes nothing. Numerically sum_g w_g grad q is
// not zero (the ring cuts through the background cells), and a large uniform part
// of B - the far-field stress - leaks through it. The "_c" values use
//   G_c = (1/h) sum_g w_g (B_g - B_mean) grad q,   B_mean = sum w_g B_g / sum w_g
// over the ring's Gauss points: the same integral, without that leak. grad_q_sum =
// |sum_g w_g grad q| is reported as the measure of the quadrature error.
//
// The first-order term (Stage B1). C0 N = F0^T (F0 N), and F0 = [I; 0] + H with H the
// displacement gradient, so B holds the stress resultant itself: B = N + O(strain^2).
// The sum of (F0 N) grad q is the equilibrium residual tested with a translation
// weighted by q; it vanishes in the exact integral, and G is what is left, second
// order in the strain. With the analytic grad q that sum is only as small as the
// quadrature error, which is first order in the strain: at small load it swamps G.
// Two more values are therefore reported, the main one unchanged:
//   "_h"  the same B, with q taken as the RKPM interpolant of its nodal values,
//         q_h = sum_I q(X_I) Psi_I, grad q_h = sum_I q(X_I) grad Psi_I. q_h is a valid
//         weight function too, and for it the first-order sum IS the solver's nodal
//         residual, zero to the Newton tolerance. Its ring is wider than [rho1, rho2]
//         by one kernel support.
//   "_d"  analytic grad q, with H^T (F0 N) in place of C0 N: the first-order term
//         dropped by hand. Exact for a membrane state only (with bending, the
//         dropped integral is not zero); a cross-check of "_h" in Stage B1.
//
// A FOLD IN THE RING (Crack_Brief_for_Code.md, C0; notes Eqs. 36, 40-42). Computed
// only when the plate has folds; everything above is then the "plate part". At each
// crease Gauss point k (line weight w_k, reference tangent T, reference normal n):
//   side +, side -   the fold's "left" and "right" points, X_k +/- 1e-7 n, with the
//                    one-sided shape functions the hinge itself uses; n = gp.normal
//                    points from - to +
//   theta, psi       fold angle exactly as the hinge computes it; psi = 1/2 Kt
//                    (theta - theta0)^2 (0 for a rigid fold: no hinge energy)
//   B+, B-           the tensor B from the one-sided kinematics
//   f_n = -n.[B]n,  f_T = -T.[B]n + d(psi)/ds,   [B] = B+ - B-
//                    d(psi)/ds by central differences between neighbouring crease
//                    points (one-sided at the two ends of the crease)
// and per ring, all along d:
//   line term   (1/h) sum_k w_k psi_k (d.T)(T.grad q)
//   G_D       = plate part - line term
//   G_fold    = (1/h) sum_k w_k (d.n) [ -(q+ n.B+n - q- n.B-n) ]
//   G_tip     = G_D - G_fold
//   G_fold_T  = (1/h) sum_k w_k (d.T) [ -(q+ T.B+n - q- T.B-n) + q d(psi)/ds ]
//               the tangential share, which the theory says is zero; reported, not
//               subtracted
// q on the crease. The "_h" values use the interpolated q of the main output, taken
// from each side with that side's shape functions: q+- = sum_I q(X_I) Psi_I(+-),
// grad q+- likewise. q+ and q- differ slightly (q_h is not continuous across the
// cut), so each one-sided B is weighted by its own q: this is what integrating the
// plate part by parts on each side gives. Where one value is needed (line term,
// d(psi)/ds term) it is the mean of the two sides. The "_a" values use the analytic
// q(X_k) and grad q(X_k) with the analytic-q plate part, as extra columns.
// The formulas hold for a straight crease with uniform Kt and theta0.
//
// Band form of the fold's share (the fallback of the notes, Step 5c; only when
// "fold_band" > 0 is given). f_n above needs the jump of one-sided values at the
// crease. The same share is the release rate of the map V2 = chi q (d.n) n, which
// moves the crease sideways and vanishes away from it:
//   G_fold_band = (1/h) sum_g w_g (n . B_g grad g),   g = chi(eta) q (d.n)
// over the face Gauss points, with eta the signed distance to the crease line and
// chi = 1 - 3t^2 + 2t^3, t = |eta| / band, zero beyond the band. V2 is normal to the
// crease, so it has no line term. "_a": analytic q and grad q; "_h": g interpolated
// from its nodal values, as q is in the main output. Each fold is taken as the
// infinite straight line through its first crease point.
// Plateau band (brief, C0b; "fold_band_plateau" = eta0 < band, default 0): chi = 1
// for |eta| <= eta0, then the same cubic with t = (|eta| - eta0) / (band - eta0).
// chi' is then zero next to the crease, where the values of B are poorest.
//
// The interpolated form at a fold ("_r"). Form "_h" has no first-order term because
// sum_I q_I d . F_int,I is the nodal residual, zero at free nodes. At a fold the
// nodes carry the hinge forces and the multipliers lambda of the displacement tie,
// and q_h is not continuous across the cut, so the tie does the work
// lambda . (q+ - q-) d: first order in the load, times the interpolation error of q.
// It is removed exactly. With F_int the plate's own nodal internal force,
//   resid  = (1/h) sum_I q_I d . F_int,I          (zero with no fold)
//   hinge  = (1/h) sum_k w_k Kt (theta - theta0) dtheta_k[q_h d]   (hinge virtual work)
// nodal equilibrium gives  resid + hinge = -(1/h) sum_k lambda_k . (q+ - q-) d, the
// spurious work, and
//   plate_r = plate_h - resid - hinge,   G_D_r = plate_r - line term.
// The band form is corrected the same way with g n in place of q d ("Gfold_band_r").
//
// Mixed form ("_m", brief E1). Only C0 N has a first-order part that must cancel
// through nodal equilibrium, so only C0 N needs the interpolated weight; the other two
// terms of B take the analytic weight, which is continuous across a crease:
//   G_m       = (1/h) sum_g w_g [ C0 N grad q_h + (2 E1 M - W_h I) grad q ]
//   G_D_m     = d . G_m - resid - hinge - line term        (the tie correction as above)
//   Gfold_band_m = (1/h) sum_g w_g [ n . C0 N grad g_h + n . (2 E1 M - W_h I) grad g ]
//                  - resid_band - hinge_band,   Gtip_band_m = G_D_m - Gfold_band_m
// Subtracted form ("_s", brief E2). Analytic q everywhere (continuous across a
// crease), and the first-order part removed explicitly at each Gauss point instead of
// through nodal equilibrium:
//   G_s    = (1/h) sum_g w_g [ B grad q - vw ] - (1/h) VW_h          (per direction)
//   G_D_s  = d . G_s - line term (analytic q)
// vw is the plate's internal virtual work density for the shift du = q e (e = d, or
// d_perp for the second component), membrane and bending, with the solver's own
// strain-displacement operator (Assembly.h) fed with q, grad q, grad grad q in place
// of a node's shape function: vw = N . de0 + M . de1. VW_h is the hinge's virtual work
// for the same shift, the same grad q on both sides. No tie correction: a continuous
// shift does no work on the tie (dq_a_max = largest |q+ - q-| between the two crease
// points, which sit 1e-7 either side of the line; the multipliers themselves are not
// available to this post-processing). Band form: the same with the shift g n,
//   Gfold_band_s = (1/h) sum_g w_g [ n . B grad g - vw(g n) ] - (1/h) VW_h(g n).
// Diagnostic parts: vw1_s, hw1_s (ring, along d), vw_band_s, hw_band_s; and vw1_h, the
// same vw summed with the INTERPOLATED q, which is the nodal residual sum_I q_I d .
// F_int,I (equal to resid_h at a fold, zero with no fold): it checks vw against the
// solver. The linear q is not C1: its second derivative has a jump line that vw does
// not see, so "_s" is meant for the cubic q.
//
// Natural curvature ("Gband", brief E7; notes Eq. 56). With a prescribed natural
// curvature kappa_bar(X) (Kinematics::naturalCurvature) the energy depends on X
// explicitly and
//   G = (1/h) sum_g w_g d . (C0 N + 2 E1 M - W_h I) grad q
//       - (1/h) sum_g w_g q M_ab d_c d(kappa_bar_ab)/dX_c,
// E1 geometric in the first sum, N, M, W_h elastic. The second sum needs q itself and
// runs over the whole disc rho < rho2 (and the kernel support beyond, for q_h):
//   Gband_h  the second sum along d with the interpolated q (Gband2_h: along d_perp)
//   Gband_a  the same with the analytic q (diagnostic)
//   G1b_h = G1_h + Gband_h,  G2b_h = G2_h + Gband2_h.
// All zero, and G1b_h = G1_h, when there is no natural curvature.
//
// "_mc": the same with the analytic-weight terms mean-subtracted, B2 - mean(B2) with
// B2 = 2 E1 M - W_h I and the mean over the Gauss points where the analytic gradient
// is not zero (the ring; for the band form, the band inside the ring's disc).
namespace JIntegral {

    struct Ring { double rho1, rho2; };

    struct Settings {
        Eigen::Vector2d tip = Eigen::Vector2d::Zero();
        Eigen::Vector2d dir = Eigen::Vector2d::UnitX();   // direction of advance d (unit)
        std::vector<Ring> rings;
        bool linear_q = false;
        double fold_band = 0.0;       // half-width of the band form of G_fold; 0: not computed
        double fold_band_plateau = 0.0;   // half-width of the part of the band where chi = 1
        std::vector<int> steps;       // empty: every exported step
        std::string csv;              // one row per ring and step
        std::string tag = "G";        // h5 dataset suffix: /step_NNNNN_<tag>
        bool csv_started = false;
    };

    struct RingResult {
        Eigen::Vector2d G = Eigen::Vector2d::Zero();
        double G1 = 0.0, G2 = 0.0, absG = 0.0, angle_deg = 0.0, abs_sum = 0.0;
        int ngp = 0;
        // mean-subtracted form (see the header) and the quadrature diagnostic
        Eigen::Vector2d Gc = Eigen::Vector2d::Zero();
        double G1c = 0.0, G2c = 0.0, absGc = 0.0, grad_q_sum = 0.0;
        Eigen::Matrix2d Bsum = Eigen::Matrix2d::Zero();
        Eigen::Vector2d gsum = Eigen::Vector2d::Zero();
        double wsum = 0.0;
        // interpolated-q form and displacement-gradient form (see the header)
        Eigen::Vector2d Gh = Eigen::Vector2d::Zero(), Gd = Eigen::Vector2d::Zero();
        double G1h = 0.0, G2h = 0.0, G1d = 0.0, G2d = 0.0;
        int ngp_h = 0;
        // fold terms (see the header); all along d, zero when there is no fold
        double abs_sum_h = 0.0;                       // (1/h) sum w |d . B grad q_h|
        double line_h = 0.0, GD_h = 0.0, Gfold_h = 0.0, Gtip_h = 0.0, GfoldT_h = 0.0, abs_fold_h = 0.0;
        double line_a = 0.0, GD_a = 0.0, Gfold_a = 0.0, Gtip_a = 0.0;
        int n_crease = 0;                             // crease points the ring touches
        double dq_max = 0.0;                          // max |q+ - q-| on the crease (interpolated q)
        // band form of the fold's share and what is left of G_D after it
        double Gfold_band_h = 0.0, Gfold_band_a = 0.0, Gtip_band_h = 0.0, Gtip_band_a = 0.0;
        // interpolated form with the work of the tie removed (see the header)
        double resid_h = 0.0, hinge_h = 0.0, GD_r = 0.0;
        double resid_band = 0.0, hinge_band = 0.0, Gfold_band_r = 0.0, Gtip_band_r = 0.0;
        // mixed form (see the header)
        Eigen::Vector2d Gm = Eigen::Vector2d::Zero(), Gmc = Eigen::Vector2d::Zero();
        Eigen::Matrix2d B2sum = Eigen::Matrix2d::Zero();
        double G1m = 0.0, G2m = 0.0, G1mc = 0.0, G2mc = 0.0, GD_m = 0.0, GD_mc = 0.0;
        double Gfold_band_m = 0.0, Gtip_band_m = 0.0, Gfold_band_mc = 0.0, Gtip_band_mc = 0.0;
        // subtracted form (see the header)
        double vw1 = 0.0, vw2 = 0.0, hw1 = 0.0, hw2 = 0.0, vw1_h = 0.0;
        double G1s = 0.0, G2s = 0.0, GD_s = 0.0;
        double vw_band = 0.0, hw_band = 0.0, Gfold_band_s = 0.0, Gtip_band_s = 0.0, dq_a_max = 0.0;
        // natural-curvature term (see the header)
        double Gband_h = 0.0, Gband2_h = 0.0, Gband_a = 0.0, G1b_h = 0.0, G2b_h = 0.0;
    };

    // Internal virtual work density N . de0 + M . de1 of the plate for the displacement
    // variation du = f(X) e, e a fixed vector: g = grad f, Hs = grad grad f. The strain
    // variations are those of Assembly.h (rows Bm, Bb) with (g, Hs) for (dPsi, ddPsi).
    inline double virtualWork(const Kinematics::LocalKinematics& kin, const Constitutive::Resultants& res,
                              const Eigen::Vector3d& e, const Eigen::Vector2d& g, const Eigen::Matrix2d& Hs) {
        const double p1 = kin.F0.col(0).dot(e), p2 = kin.F0.col(1).dot(e), be = kin.b.dot(e);
        const Eigen::Vector3d gI = kin.F0 * (kin.C0_inv * g);
        const Eigen::Vector3d de0(p1 * g.x(), p2 * g.y(), p1 * g.y() + p2 * g.x());
        const Eigen::Vector3d de1((-Hs(0, 0) + kin.r0XX.dot(gI)) * be, (-Hs(1, 1) + kin.r0YY.dot(gI)) * be,
                                  2.0 * (-Hs(0, 1) + kin.r0XY.dot(gI)) * be);
        return res.N.dot(de0) + res.M.dot(de1);
    }

    // Analytic q of a ring at a point: value, gradient and second gradient.
    inline double qAnalytic(const Eigen::Vector2d& rvec, const Ring& r, bool linear_q,
                            Eigen::Vector2d& grad, Eigen::Matrix2d& hess) {
        grad.setZero(); hess.setZero();
        const double rho = rvec.norm();
        if (rho <= r.rho1) return 1.0;
        if (rho >= r.rho2) return 0.0;
        const double width = r.rho2 - r.rho1, t = (rho - r.rho1) / width;
        const double dq = (linear_q ? -1.0 : (-6.0 * t + 6.0 * t * t)) / width;
        const double d2q = linear_q ? 0.0 : (-6.0 + 12.0 * t) / (width * width);
        const Eigen::Vector2d er = rvec / rho;
        grad = dq * er;
        hess = d2q * er * er.transpose() + (dq / rho) * (Eigen::Matrix2d::Identity() - er * er.transpose());
        return linear_q ? 1.0 - t : 1.0 - 3.0 * t * t + 2.0 * t * t * t;
    }

    // One crease Gauss point of a fold: ring-independent quantities.
    struct CreasePoint {
        int fold = 0, i = 0;
        Eigen::Vector2d X = Eigen::Vector2d::Zero(), T = Eigen::Vector2d::Zero(), n = Eigen::Vector2d::Zero();
        double s = 0.0, w = 0.0, theta = 0.0, theta0 = 0.0, psi = 0.0, dpsi_ds = 0.0;
        Eigen::Matrix2d Bp = Eigen::Matrix2d::Zero(), Bm = Eigen::Matrix2d::Zero();
        double fn = 0.0, fT = 0.0;
        // one-sided resultants, to judge the one-sided values: the membrane traction
        // N n must be continuous across the crease, and the bending moment n.M n on
        // each side must balance the hinge moment Kt (theta - theta0)
        double Nnn_p = 0.0, Nnn_m = 0.0, NTn_p = 0.0, NTn_m = 0.0, Mnn_p = 0.0, Mnn_m = 0.0, m_hinge = 0.0;
    };

    inline double qValue(double rho, const Ring& r, bool linear_q) {
        if (rho <= r.rho1) return 1.0;
        if (rho >= r.rho2) return 0.0;
        const double t = (rho - r.rho1) / (r.rho2 - r.rho1);
        return linear_q ? 1.0 - t : 1.0 - 3.0 * t * t + 2.0 * t * t * t;
    }

    // The tensor B at one face Gauss point; W_h returned too.
    inline Eigen::Matrix2d eshelbyB(const GaussPointData& gp, const std::vector<Node>& nodes,
                                    const LaminateProperties& mat, double* W_h = nullptr,
                                    Eigen::Matrix2d* B_d = nullptr, Eigen::Matrix2d* C0N = nullptr,
                                    Kinematics::LocalKinematics* kin_out = nullptr,
                                    Constitutive::Resultants* res_out = nullptr) {
        const auto kin = Kinematics::compute(gp, nodes);
        Constitutive::Resultants res;
        Constitutive::evaluate(kin, mat, res);
        if (kin_out) *kin_out = kin;
        if (res_out) *res_out = res;
        const Eigen::Matrix2d C0 = kin.F0.transpose() * kin.F0;
        Eigen::Matrix2d N, M;
        N << res.N(0), res.N(2), res.N(2), res.N(1);
        M << res.M(0), res.M(2), res.M(2), res.M(1);
        Eigen::Vector3d e0, e1;
        e0 << kin.E0(0, 0), kin.E0(1, 1), 2.0 * kin.E0(0, 1);
        // W_h, like N and M, from the elastic bending strain; the E1 of the 2 E1 M term
        // below stays the geometric one (notes, Eq. 56)
        e1 << kin.E1_el(0, 0), kin.E1_el(1, 1), 2.0 * kin.E1_el(0, 1);
        const double W = 0.5 * (res.N.dot(e0) + res.M.dot(e1));
        if (W_h) *W_h = W;
        if (B_d) {
            Eigen::Matrix<double, 3, 2> Hd = kin.F0;
            Hd(0, 0) -= 1.0;
            Hd(1, 1) -= 1.0;
            *B_d = Hd.transpose() * (kin.F0 * N) + 2.0 * kin.E1 * M - W * Eigen::Matrix2d::Identity();
        }
        if (C0N) *C0N = C0 * N;
        return C0 * N + 2.0 * kin.E1 * M - W * Eigen::Matrix2d::Identity();
    }

    // Crease points of every fold, from the nodal displacements already in plate.nodes.
    inline std::vector<CreasePoint> creasePoints(const Meshless& plate) {
        std::vector<CreasePoint> out;
        for (size_t f = 0; f < plate.folds.size(); ++f) {
            const Fold& fold = plate.folds[f];
            const size_t first = out.size();
            for (size_t i = 0; i < fold.left.gps.size(); ++i) {
                if (!Constraints::gpActive(&fold.gp_active, i)) continue;
                const GaussPointData& gL = fold.left.gps[i];
                const GaussPointData& gR = fold.right.gps[i];
                const auto kinL = Kinematics::compute(gL, plate.nodes);
                const auto kinR = Kinematics::compute(gR, plate.nodes);
                Eigen::Matrix<double, 12, 1> F;
                F.segment<3>(0) = kinL.F0.col(0); F.segment<3>(3) = kinL.F0.col(1);
                F.segment<3>(6) = kinR.F0.col(0); F.segment<3>(9) = kinR.F0.col(1);
                CreasePoint c;
                c.fold = static_cast<int>(f);
                c.i = static_cast<int>(i);
                c.T = gL.tangent;
                c.n = gL.normal;
                c.X = 0.5 * (gL.X_ref + gR.X_ref);
                c.w = gL.weight;
                c.theta = Constraints::foldAngleAndGrad(F, c.T.x(), c.T.y(), nullptr);
                c.theta0 = Constraints::gpTheta0(fold.Ang0, &fold.gp_theta0, i);
                c.psi = fold.flagRig ? 0.0 : 0.5 * fold.Kt * (c.theta - c.theta0) * (c.theta - c.theta0);
                c.Bp = eshelbyB(gL, plate.nodes, plate.material);
                c.Bm = eshelbyB(gR, plate.nodes, plate.material);
                c.m_hinge = fold.flagRig ? 0.0 : fold.Kt * (c.theta - c.theta0);
                Constitutive::Resultants rL, rR;
                Constitutive::evaluate(kinL, plate.material, rL);
                Constitutive::evaluate(kinR, plate.material, rR);
                auto sym = [](const Eigen::Vector3d& v) { Eigen::Matrix2d m; m << v(0), v(2), v(2), v(1); return m; };
                c.Nnn_p = c.n.dot(sym(rL.N) * c.n); c.Nnn_m = c.n.dot(sym(rR.N) * c.n);
                c.NTn_p = c.T.dot(sym(rL.N) * c.n); c.NTn_m = c.T.dot(sym(rR.N) * c.n);
                c.Mnn_p = c.n.dot(sym(rL.M) * c.n); c.Mnn_m = c.n.dot(sym(rR.M) * c.n);
                out.push_back(c);
            }
            if (out.size() == first) continue;
            // arc length along the fold's own direction, then d(psi)/ds in that order
            const Eigen::Vector2d T0 = out[first].T;
            for (size_t j = first; j < out.size(); ++j) out[j].s = T0.dot(out[j].X);
            std::vector<size_t> ord;
            for (size_t j = first; j < out.size(); ++j) ord.push_back(j);
            std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return out[a].s < out[b].s; });
            for (size_t m = 0; m < ord.size(); ++m) {
                const size_t a = ord[m > 0 ? m - 1 : m], b = ord[m + 1 < ord.size() ? m + 1 : m];
                CreasePoint& c = out[ord[m]];
                const double ds = out[b].s - out[a].s;
                const double dpsi = ds > 0.0 ? (out[b].psi - out[a].psi) / ds : 0.0;
                c.dpsi_ds = dpsi * (c.T.dot(T0) < 0.0 ? -1.0 : 1.0);   // along the point's own T
                const Eigen::Matrix2d jump = c.Bp - c.Bm;
                c.fn = -c.n.dot(jump * c.n);
                c.fT = -c.T.dot(jump * c.n) + c.dpsi_ds;
            }
        }
        return out;
    }

    // Virtual work of the hinges, sum_k w_k Kt (theta - theta0) dtheta_k, for the
    // in-plane virtual displacement with nodal values g_I * dir.
    inline double hingeVirtualWork(const Meshless& plate, const std::vector<double>& g, const Eigen::Vector2d& dir) {
        const Eigen::Vector3d d3(dir.x(), dir.y(), 0.0);
        double work = 0.0;
        for (const Fold& fold : plate.folds) {
            if (fold.flagRig) continue;
            for (size_t i = 0; i < fold.left.gps.size(); ++i) {
                if (!Constraints::gpActive(&fold.gp_active, i)) continue;
                const GaussPointData& gL = fold.left.gps[i];
                const GaussPointData& gR = fold.right.gps[i];
                Eigen::Vector2d ggL = Eigen::Vector2d::Zero(), ggR = Eigen::Vector2d::Zero();
                for (size_t j = 0; j < gL.neighbor_node_ids.size(); ++j) ggL += g[gL.neighbor_node_ids[j]] * gL.dPsi[j];
                for (size_t j = 0; j < gR.neighbor_node_ids.size(); ++j) ggR += g[gR.neighbor_node_ids[j]] * gR.dPsi[j];
                if (ggL.squaredNorm() <= 0.0 && ggR.squaredNorm() <= 0.0) continue;
                const auto kinL = Kinematics::compute(gL, plate.nodes);
                const auto kinR = Kinematics::compute(gR, plate.nodes);
                Eigen::Matrix<double, 12, 1> F, gradF, dF;
                F.segment<3>(0) = kinL.F0.col(0); F.segment<3>(3) = kinL.F0.col(1);
                F.segment<3>(6) = kinR.F0.col(0); F.segment<3>(9) = kinR.F0.col(1);
                const double theta = Constraints::foldAngleAndGrad(F, gL.tangent.x(), gL.tangent.y(), &gradF);
                dF.segment<3>(0) = d3 * ggL.x(); dF.segment<3>(3) = d3 * ggL.y();
                dF.segment<3>(6) = d3 * ggR.x(); dF.segment<3>(9) = d3 * ggR.y();
                const double m = fold.Kt * (theta - Constraints::gpTheta0(fold.Ang0, &fold.gp_theta0, i));
                work += gL.weight * m * gradF.dot(dF);
            }
        }
        return work;
    }

    // The same for an analytic shift f(X) dir: grad f, given by `gradAt`, is taken at
    // the crease point and used on both sides.
    template <typename GradAt>
    inline double hingeVirtualWorkAnalytic(const Meshless& plate, const Eigen::Vector2d& dir, GradAt gradAt) {
        const Eigen::Vector3d d3(dir.x(), dir.y(), 0.0);
        double work = 0.0;
        for (const Fold& fold : plate.folds) {
            if (fold.flagRig) continue;
            for (size_t i = 0; i < fold.left.gps.size(); ++i) {
                if (!Constraints::gpActive(&fold.gp_active, i)) continue;
                const GaussPointData& gL = fold.left.gps[i];
                const GaussPointData& gR = fold.right.gps[i];
                const Eigen::Vector2d gg = gradAt(Eigen::Vector2d(0.5 * (gL.X_ref + gR.X_ref)));
                if (gg.squaredNorm() <= 0.0) continue;
                const auto kinL = Kinematics::compute(gL, plate.nodes);
                const auto kinR = Kinematics::compute(gR, plate.nodes);
                Eigen::Matrix<double, 12, 1> F, gradF, dF;
                F.segment<3>(0) = kinL.F0.col(0); F.segment<3>(3) = kinL.F0.col(1);
                F.segment<3>(6) = kinR.F0.col(0); F.segment<3>(9) = kinR.F0.col(1);
                const double theta = Constraints::foldAngleAndGrad(F, gL.tangent.x(), gL.tangent.y(), &gradF);
                dF.segment<3>(0) = d3 * gg.x(); dF.segment<3>(3) = d3 * gg.y();
                dF.segment<3>(6) = d3 * gg.x(); dF.segment<3>(9) = d3 * gg.y();
                const double m = fold.Kt * (theta - Constraints::gpTheta0(fold.Ang0, &fold.gp_theta0, i));
                work += gL.weight * m * gradF.dot(dF);
            }
        }
        return work;
    }

    inline std::vector<RingResult> compute(Meshless& plate, const Eigen::VectorXd& x, const Settings& s,
                                           std::vector<CreasePoint>* crease_out = nullptr) {
        plate.updateKinematics(x);
        const double h = plate.material.total_thickness();
        const Eigen::Vector2d d = s.dir.normalized();
        const Eigen::Vector2d dperp(-d.y(), d.x());
        const Eigen::Vector3d d3(d.x(), d.y(), 0.0), dperp3(dperp.x(), dperp.y(), 0.0);

        double rmin = 1e300, rmax = 0.0;
        for (const auto& r : s.rings) { rmin = std::min(rmin, r.rho1); rmax = std::max(rmax, r.rho2); }

        std::vector<RingResult> out(s.rings.size());
        // nodal values of q, one row per ring, for the interpolated form
        std::vector<std::vector<double>> qn(s.rings.size(), std::vector<double>(plate.nodes.size()));
        for (size_t k = 0; k < s.rings.size(); ++k)
            for (size_t I = 0; I < plate.nodes.size(); ++I)
                qn[k][I] = qValue((plate.nodes[I].X_ref - s.tip).norm(), s.rings[k], s.linear_q);

        for (const auto& gp : plate.faces.gps) {
            const Eigen::Vector2d rvec = gp.X_ref - s.tip;
            const double rho = rvec.norm();
            // interpolated q: grad q_h is not zero wherever the neighbours' q differ
            bool haveB = false;
            Eigen::Matrix2d B, Bd, CN;
            Kinematics::LocalKinematics kin;
            Constitutive::Resultants res;
            for (size_t k = 0; k < s.rings.size(); ++k) {
                Eigen::Vector2d gh = Eigen::Vector2d::Zero();
                Eigen::Matrix2d Hh = Eigen::Matrix2d::Zero();
                double qmin = 2.0, qmax = -1.0;
                for (size_t i = 0; i < gp.neighbor_node_ids.size(); ++i) {
                    const double qI = qn[k][gp.neighbor_node_ids[i]];
                    qmin = std::min(qmin, qI);
                    qmax = std::max(qmax, qI);
                    gh += qI * gp.dPsi[i];
                    Hh += qI * gp.ddPsi[i];
                }
                if (qmax <= qmin) continue;      // q constant over the support: no contribution
                if (!haveB) { B = eshelbyB(gp, plate.nodes, plate.material, nullptr, &Bd, &CN, &kin, &res); haveB = true; }
                out[k].Gh += gp.weight * (B * gh);
                out[k].Gm += gp.weight * (CN * gh);
                out[k].vw1_h += gp.weight * virtualWork(kin, res, d3, gh, Hh);
                out[k].abs_sum_h += gp.weight * std::abs(d.dot(B * gh));
                out[k].ngp_h += 1;
            }
            if (rho <= rmin || rho >= rmax) continue;
            if (!haveB) { B = eshelbyB(gp, plate.nodes, plate.material, nullptr, &Bd, &CN, &kin, &res); haveB = true; }
            for (size_t k = 0; k < s.rings.size(); ++k) {
                const Ring& r = s.rings[k];
                if (rho <= r.rho1 || rho >= r.rho2) continue;
                const double width = r.rho2 - r.rho1;
                const double t = (rho - r.rho1) / width;
                const double dq = (s.linear_q ? -1.0 : (-6.0 * t + 6.0 * t * t)) / width;
                const Eigen::Vector2d gq = dq * rvec / rho;
                {
                    Eigen::Vector2d ga; Eigen::Matrix2d Ha;
                    qAnalytic(rvec, r, s.linear_q, ga, Ha);
                    out[k].vw1 += gp.weight * virtualWork(kin, res, d3, ga, Ha);
                    out[k].vw2 += gp.weight * virtualWork(kin, res, dperp3, ga, Ha);
                }
                const Eigen::Vector2d c = gp.weight * (B * gq);
                out[k].Bsum += gp.weight * B;
                out[k].B2sum += gp.weight * (B - CN);
                out[k].Gm += gp.weight * ((B - CN) * gq);
                out[k].gsum += gp.weight * gq;
                out[k].wsum += gp.weight;
                out[k].G += c;
                out[k].Gd += gp.weight * (Bd * gq);
                out[k].abs_sum += c.norm();
                out[k].ngp += 1;
            }
        }
        // natural-curvature term: every Gauss point where q is not zero
        const Kinematics::NaturalCurvature& nat = Kinematics::naturalCurvature();
        if (nat.active) {
            for (const auto& gp : plate.faces.gps) {
                const Eigen::Matrix2d dk1 = nat.dkappa(gp.X_ref, d), dk2 = nat.dkappa(gp.X_ref, dperp);
                if (dk1.cwiseAbs().maxCoeff() + dk2.cwiseAbs().maxCoeff() < 1e-14 * nat.theta0 / (nat.width * nat.width)) continue;
                const double rho = (gp.X_ref - s.tip).norm();
                bool have = false;
                Eigen::Matrix2d M = Eigen::Matrix2d::Zero();
                for (size_t k = 0; k < s.rings.size(); ++k) {
                    double qh = 0.0;
                    for (size_t i = 0; i < gp.neighbor_node_ids.size(); ++i) qh += qn[k][gp.neighbor_node_ids[i]] * gp.Psi[i];
                    const double qa = qValue(rho, s.rings[k], s.linear_q);
                    if (qh == 0.0 && qa == 0.0) continue;
                    if (!have) {
                        const auto kin = Kinematics::compute(gp, plate.nodes);
                        Constitutive::Resultants res;
                        Constitutive::evaluate(kin, plate.material, res);
                        M << res.M(0), res.M(2), res.M(2), res.M(1);
                        have = true;
                    }
                    const double m1 = M.cwiseProduct(dk1).sum(), m2 = M.cwiseProduct(dk2).sum();
                    out[k].Gband_h  -= gp.weight * qh * m1 / h;
                    out[k].Gband2_h -= gp.weight * qh * m2 / h;
                    out[k].Gband_a  -= gp.weight * qa * m1 / h;
                }
            }
        }

        for (auto& r : out) {
            if (r.wsum > 0.0) r.Gc = (r.G - (r.Bsum / r.wsum) * r.gsum) / h;
            r.G1c = d.dot(r.Gc);
            r.G2c = dperp.dot(r.Gc);
            r.absGc = r.Gc.norm();
            r.grad_q_sum = r.gsum.norm();
            r.Gmc = r.Gm;
            if (r.wsum > 0.0) r.Gmc -= (r.B2sum / r.wsum) * r.gsum;
            r.Gm /= h; r.Gmc /= h;
            r.G1m = d.dot(r.Gm); r.G2m = dperp.dot(r.Gm);
            r.G1mc = d.dot(r.Gmc); r.G2mc = dperp.dot(r.Gmc);
            r.GD_m = r.G1m; r.GD_mc = r.G1mc;      // no fold: nothing to remove
            r.Gh /= h;
            r.Gd /= h;
            r.G1h = d.dot(r.Gh); r.G2h = dperp.dot(r.Gh);
            r.G1d = d.dot(r.Gd); r.G2d = dperp.dot(r.Gd);
            r.G /= h;
            r.abs_sum /= h;
            r.G1 = d.dot(r.G);
            r.G2 = dperp.dot(r.G);
            r.absG = r.G.norm();
            r.angle_deg = std::atan2(r.G2, r.G1) * 180.0 / M_PI;
            r.abs_sum_h /= h;
            r.vw1 /= h; r.vw2 /= h; r.vw1_h /= h;
            r.G1b_h = r.G1h + r.Gband_h;
            r.G2b_h = r.G2h + r.Gband2_h;
            r.G1s = r.G1 - r.vw1; r.G2s = r.G2 - r.vw2;     // no fold: no hinge, no line term
            r.GD_s = r.G1s;
        }

        // fold terms (see the header)
        if (!plate.folds.empty()) {
            const std::vector<CreasePoint> cps = creasePoints(plate);
            for (size_t k = 0; k < s.rings.size(); ++k) {
                const Ring& ring = s.rings[k];
                RingResult& r = out[k];
                for (const auto& c : cps) {
                    const GaussPointData& gL = plate.folds[c.fold].left.gps[c.i];
                    const GaussPointData& gR = plate.folds[c.fold].right.gps[c.i];
                    double qp = 0.0, qm = 0.0, qabs = 0.0;
                    Eigen::Vector2d gqp = Eigen::Vector2d::Zero(), gqm = Eigen::Vector2d::Zero();
                    for (size_t i = 0; i < gL.neighbor_node_ids.size(); ++i) {
                        const double qI = qn[k][gL.neighbor_node_ids[i]];
                        qp += qI * gL.Psi[i]; gqp += qI * gL.dPsi[i]; qabs += qI;
                    }
                    for (size_t i = 0; i < gR.neighbor_node_ids.size(); ++i) {
                        const double qI = qn[k][gR.neighbor_node_ids[i]];
                        qm += qI * gR.Psi[i]; gqm += qI * gR.dPsi[i]; qabs += qI;
                    }
                    const double dn = d.dot(c.n), dT = d.dot(c.T);
                    if (qabs > 0.0) {
                        const Eigen::Vector2d Bpn = c.Bp * c.n, Bmn = c.Bm * c.n;
                        const double fold_n = c.w * dn * (-(qp * c.n.dot(Bpn) - qm * c.n.dot(Bmn)));
                        r.line_h += c.w * c.psi * dT * 0.5 * c.T.dot(gqp + gqm);
                        r.Gfold_h += fold_n;
                        r.abs_fold_h += std::abs(fold_n);
                        r.GfoldT_h += c.w * dT * (-(qp * c.T.dot(Bpn) - qm * c.T.dot(Bmn)) + 0.5 * (qp + qm) * c.dpsi_ds);
                        r.n_crease += 1;
                        r.dq_max = std::max(r.dq_max, std::abs(qp - qm));
                    }
                    // analytic q
                    const Eigen::Vector2d rvec = c.X - s.tip;
                    const double rho = rvec.norm();
                    r.Gfold_a += c.w * qValue(rho, ring, s.linear_q) * dn * c.fn;
                    if (rho > ring.rho1 && rho < ring.rho2) {
                        const double width = ring.rho2 - ring.rho1;
                        const double t = (rho - ring.rho1) / width;
                        const double dq = (s.linear_q ? -1.0 : (-6.0 * t + 6.0 * t * t)) / width;
                        r.line_a += c.w * c.psi * dT * dq * c.T.dot(rvec) / rho;
                    }
                }
                r.line_h /= h; r.Gfold_h /= h; r.abs_fold_h /= h; r.GfoldT_h /= h;
                r.line_a /= h; r.Gfold_a /= h;
                r.GD_h = r.G1h - r.line_h;
                r.Gtip_h = r.GD_h - r.Gfold_h;
                r.GD_a = r.G1 - r.line_a;
                r.Gtip_a = r.GD_a - r.Gfold_a;
            }

            // the plate's own nodal internal force, for the "_r" forms (see the header)
            Eigen::VectorXd F_int = Eigen::VectorXd::Zero(plate.nodes.size() * 3);
            plate.face_assembler.updateStiffness(plate.faces.gps, plate.nodes, plate.material, F_int);
            for (size_t k = 0; k < s.rings.size(); ++k) {
                double w = 0.0;
                for (size_t I = 0; I < plate.nodes.size(); ++I)
                    if (qn[k][I] != 0.0) w += qn[k][I] * (d.x() * F_int(3 * I) + d.y() * F_int(3 * I + 1));
                out[k].resid_h = w / h;
                out[k].hinge_h = hingeVirtualWork(plate, qn[k], d) / h;
                out[k].GD_r = out[k].G1h - out[k].resid_h - out[k].hinge_h - out[k].line_h;
                out[k].GD_m = out[k].G1m - out[k].resid_h - out[k].hinge_h - out[k].line_h;
                {
                    const Ring ring = s.rings[k];
                    auto gradQ = [&](const Eigen::Vector2d& X) {
                        Eigen::Vector2d g; Eigen::Matrix2d Hq;
                        qAnalytic(X - s.tip, ring, s.linear_q, g, Hq);
                        return g;
                    };
                    out[k].hw1 = hingeVirtualWorkAnalytic(plate, d, gradQ) / h;
                    out[k].hw2 = hingeVirtualWorkAnalytic(plate, dperp, gradQ) / h;
                    out[k].G1s = out[k].G1 - out[k].vw1 - out[k].hw1;
                    out[k].G2s = out[k].G2 - out[k].vw2 - out[k].hw2;
                    out[k].GD_s = out[k].G1s - out[k].line_a;
                    for (const auto& c : cps) {
                        const GaussPointData& gL = plate.folds[c.fold].left.gps[c.i];
                        const GaussPointData& gR = plate.folds[c.fold].right.gps[c.i];
                        out[k].dq_a_max = std::max(out[k].dq_a_max,
                            std::abs(qValue((gL.X_ref - s.tip).norm(), ring, s.linear_q) - qValue((gR.X_ref - s.tip).norm(), ring, s.linear_q)));
                    }
                }
                out[k].GD_mc = out[k].G1mc - out[k].resid_h - out[k].hinge_h - out[k].line_h;
            }

            // band form of the fold's share (see the header)
            if (s.fold_band > 0.0) {
                const double band = s.fold_band;
                for (size_t f = 0; f < plate.folds.size(); ++f) {
                    if (plate.folds[f].left.gps.empty()) continue;
                    const GaussPointData& g0 = plate.folds[f].left.gps[0];
                    const Eigen::Vector2d n = g0.normal;
                    const Eigen::Vector2d X0 = g0.X_ref - 1e-7 * n;      // on the crease line
                    const double dn = d.dot(n);
                    const Eigen::Vector3d n3(n.x(), n.y(), 0.0);
                    const double eta0 = s.fold_band_plateau;
                    // t = 0 on the plateau |eta| <= eta0, 1 at the edge of the band
                    auto tOf = [&](double eta) { return std::max(0.0, (std::abs(eta) - eta0) / (band - eta0)); };
                    auto chi = [&](double eta) {
                        const double t = tOf(eta);
                        return t >= 1.0 ? 0.0 : 1.0 - 3.0 * t * t + 2.0 * t * t * t;
                    };
                    // nodal values of g = chi q (d.n), one row per ring
                    std::vector<std::vector<double>> gnode(s.rings.size(), std::vector<double>(plate.nodes.size(), 0.0));
                    for (size_t I = 0; I < plate.nodes.size(); ++I) {
                        const double c = chi(n.dot(plate.nodes[I].X_ref - X0));
                        if (c <= 0.0) continue;
                        for (size_t k = 0; k < s.rings.size(); ++k) gnode[k][I] = c * qn[k][I] * dn;
                    }
                    for (size_t k = 0; k < s.rings.size(); ++k) {
                        double w = 0.0;
                        for (size_t I = 0; I < plate.nodes.size(); ++I)
                            if (gnode[k][I] != 0.0) w += gnode[k][I] * (n.x() * F_int(3 * I) + n.y() * F_int(3 * I + 1));
                        out[k].resid_band += w / h;
                        out[k].hinge_band += hingeVirtualWork(plate, gnode[k], n) / h;
                    }
                    // for "_mc": weighted sums of n . B2, grad g and the weights, per ring
                    std::vector<Eigen::Vector2d> nB2sum(s.rings.size(), Eigen::Vector2d::Zero()), gasum(s.rings.size(), Eigen::Vector2d::Zero());
                    std::vector<double> wasum(s.rings.size(), 0.0), mixed(s.rings.size(), 0.0);
                    for (const auto& gp : plate.faces.gps) {
                        const double eta = n.dot(gp.X_ref - X0);
                        if (std::abs(eta) >= band + plate.rho) continue;
                        bool haveB = false;
                        Eigen::Matrix2d B, CN;
                        Kinematics::LocalKinematics kin;
                        Constitutive::Resultants res;
                        const Eigen::Vector2d rvec = gp.X_ref - s.tip;
                        const double rho = rvec.norm();
                        const double t = tOf(eta);
                        const double c = chi(eta);
                        const double dc = t >= 1.0 ? 0.0 : (-6.0 * t + 6.0 * t * t) / (band - eta0) * (eta < 0.0 ? -1.0 : 1.0);
                        const double d2c = (t >= 1.0 || t <= 0.0) ? 0.0 : (-6.0 + 12.0 * t) / ((band - eta0) * (band - eta0));
                        for (size_t k = 0; k < s.rings.size(); ++k) {
                            const Ring& ring = s.rings[k];
                            Eigen::Vector2d gh = Eigen::Vector2d::Zero();
                            double gabs = 0.0;
                            for (size_t i = 0; i < gp.neighbor_node_ids.size(); ++i) {
                                const double gI = gnode[k][gp.neighbor_node_ids[i]];
                                gh += gI * gp.dPsi[i];
                                gabs += std::abs(gI);
                            }
                            const double q = qValue(rho, ring, s.linear_q);
                            Eigen::Vector2d ga = dn * dc * q * n;
                            if (c > 0.0 && rho > ring.rho1 && rho < ring.rho2) {
                                const double width = ring.rho2 - ring.rho1;
                                const double tq = (rho - ring.rho1) / width;
                                const double dq = (s.linear_q ? -1.0 : (-6.0 * tq + 6.0 * tq * tq)) / width;
                                ga += dn * c * dq * rvec / rho;
                            }
                            if (gabs <= 0.0 && ga.squaredNorm() <= 0.0) continue;
                            if (!haveB) { B = eshelbyB(gp, plate.nodes, plate.material, nullptr, nullptr, &CN, &kin, &res); haveB = true; }
                            if (ga.squaredNorm() > 0.0 || (c > 0.0 && rho > ring.rho1 && rho < ring.rho2)) {
                                // second gradient of g = chi q (d.n), for the virtual work of the shift g n
                                Eigen::Vector2d gq; Eigen::Matrix2d Hq;
                                const double qa = qAnalytic(rvec, ring, s.linear_q, gq, Hq);
                                const Eigen::Matrix2d Ha = dn * (d2c * qa * n * n.transpose()
                                                                 + dc * (n * gq.transpose() + gq * n.transpose()) + c * Hq);
                                out[k].vw_band += gp.weight * virtualWork(kin, res, n3, ga, Ha) / h;
                            }
                            const Eigen::Vector2d Btn = B.transpose() * n;       // n . B
                            out[k].Gfold_band_h += gp.weight * Btn.dot(gh) / h;
                            out[k].Gfold_band_a += gp.weight * Btn.dot(ga) / h;
                            const Eigen::Vector2d CNtn = CN.transpose() * n, B2tn = Btn - CNtn;
                            mixed[k] += gp.weight * (CNtn.dot(gh) + B2tn.dot(ga)) / h;
                            if (ga.squaredNorm() > 0.0) {
                                nB2sum[k] += gp.weight * B2tn;
                                gasum[k] += gp.weight * ga;
                                wasum[k] += gp.weight;
                            }
                        }
                    }
                    for (size_t k = 0; k < s.rings.size(); ++k) {
                        // hinge work of the shift g n: on the crease chi = 1 and chi' = 0
                        const Ring ring = s.rings[k];
                        out[k].hw_band += hingeVirtualWorkAnalytic(plate, n, [&](const Eigen::Vector2d& X) {
                            Eigen::Vector2d g; Eigen::Matrix2d Hq;
                            qAnalytic(X - s.tip, ring, s.linear_q, g, Hq);
                            return Eigen::Vector2d(dn * g);
                        }) / h;
                        out[k].Gfold_band_m += mixed[k];
                        out[k].Gfold_band_mc += mixed[k] - (wasum[k] > 0.0 ? (nB2sum[k] / wasum[k]).dot(gasum[k]) / h : 0.0);
                    }
                }
                for (auto& r : out) {
                    r.Gtip_band_h = r.GD_h - r.Gfold_band_h;
                    r.Gtip_band_a = r.GD_a - r.Gfold_band_a;
                    r.Gfold_band_r = r.Gfold_band_h - r.resid_band - r.hinge_band;
                    r.Gtip_band_r = r.GD_r - r.Gfold_band_r;
                    r.Gfold_band_m -= r.resid_band + r.hinge_band;
                    r.Gfold_band_mc -= r.resid_band + r.hinge_band;
                    r.Gtip_band_m = r.GD_m - r.Gfold_band_m;
                    r.Gtip_band_mc = r.GD_mc - r.Gfold_band_mc;
                    r.Gfold_band_s = r.Gfold_band_a - r.vw_band - r.hw_band;
                    r.Gtip_band_s = r.GD_s - r.Gfold_band_s;
                }
            }
            if (crease_out) *crease_out = cps;
        }
        return out;
    }

    // Compute at one exported step and write: a CSV row per ring, and the dataset
    // /step_NNNNN_<tag> next to the step in the .h5 (rows = rings).
    inline void report(Meshless& plate, const Eigen::VectorXd& x, int step, Settings& s,
                       const std::string& h5_path, const char* git_commit) {
        if (s.rings.empty()) return;
        if (!s.steps.empty() && std::find(s.steps.begin(), s.steps.end(), step) == s.steps.end()) return;
        std::vector<CreasePoint> cps;
        const auto res = compute(plate, x, s, &cps);

        printf("\n >> G (domain integral) at step %d, tip (%.6g, %.6g), d = (%.4f, %.4f), q %s\n",
               step, s.tip.x(), s.tip.y(), s.dir.x(), s.dir.y(), s.linear_q ? "linear" : "cubic");
        for (size_t k = 0; k < res.size(); ++k)
            printf("    ring %zu [%.4g, %.4g]: G1 = %.8e  G2 = %.8e  |G| = %.8e  angle %.3f deg  (%d GPs, |G|/abs_sum = %.2e)"
                   " | mean-subtracted: G1 = %.8e  G2 = %.8e | interpolated q: G1 = %.8e  G2 = %.8e (%d GPs)"
                   " | displacement-gradient form: G1 = %.8e  G2 = %.8e\n",
                   k, s.rings[k].rho1, s.rings[k].rho2, res[k].G1, res[k].G2, res[k].absG, res[k].angle_deg,
                   res[k].ngp, res[k].abs_sum > 0.0 ? res[k].absG / res[k].abs_sum : 0.0, res[k].G1c, res[k].G2c,
                   res[k].G1h, res[k].G2h, res[k].ngp_h, res[k].G1d, res[k].G2d);

        // Fold terms: <output>_<tag>_fold.csv, one row per ring and step, and the crease
        // profile <output>_<tag>_crease.csv, one row per crease point and step.
        const bool have_fold = !cps.empty();
        double fn_max = 0.0, fT_max = 0.0;
        for (const auto& c : cps) { fn_max = std::max(fn_max, std::abs(c.fn)); fT_max = std::max(fT_max, std::abs(c.fT)); }
        if (have_fold) {
            printf("    fold terms (along d, interpolated q; %zu crease points, max|f_T|/max|f_n| = %.3e / %.3e = %.3e)\n",
                   cps.size(), fT_max, fn_max, fn_max > 0.0 ? fT_max / fn_max : 0.0);
            for (size_t k = 0; k < res.size(); ++k)
                printf("    ring %zu [%.4g, %.4g]: plate = %.8e  line = %.8e  G_D = %.8e  G_fold = %.8e  G_D - G_fold = %.8e"
                       "  (G_fold_T = %.3e, %d crease points) | analytic q: G_D = %.8e  G_fold = %.8e  G_D - G_fold = %.8e\n",
                       k, s.rings[k].rho1, s.rings[k].rho2, res[k].G1h, res[k].line_h, res[k].GD_h, res[k].Gfold_h,
                       res[k].Gtip_h, res[k].GfoldT_h, res[k].n_crease, res[k].GD_a, res[k].Gfold_a, res[k].Gtip_a);
            for (size_t k = 0; k < res.size(); ++k)
                printf("    ring %zu, tie work removed: resid = %.8e  hinge = %.8e  G_D_r = %.8e  (max |q+ - q-| = %.2e)\n",
                       k, res[k].resid_h, res[k].hinge_h, res[k].GD_r, res[k].dq_max);
            if (s.fold_band > 0.0)
                for (size_t k = 0; k < res.size(); ++k)
                    printf("    ring %zu, band %.4g (plateau %.4g): G_fold = %.8e  G_D - G_fold = %.8e | tie work removed: G_fold = %.8e  G_D - G_fold = %.8e"
                           " | analytic q: G_fold = %.8e  G_D - G_fold = %.8e\n",
                           k, s.fold_band, s.fold_band_plateau, res[k].Gfold_band_h, res[k].Gtip_band_h, res[k].Gfold_band_r, res[k].Gtip_band_r,
                           res[k].Gfold_band_a, res[k].Gtip_band_a);
        }
        if (have_fold && s.csv.size() > 4) {
            const std::string stem = s.csv.substr(0, s.csv.size() - 4);
            const char* mode = s.csv_started ? "a" : "w";
            if (FILE* f = std::fopen((stem + "_fold.csv").c_str(), mode)) {
                if (!s.csv_started) {
                    std::fprintf(f, "# fold terms of the energy release rate (along d), git_commit=%s, tip_x=%.12g, tip_y=%.12g, d_x=%.12g, d_y=%.12g, q=%s, h=%.12g, fold_band=%.12g, fold_band_plateau=%.12g\n",
                                 git_commit, s.tip.x(), s.tip.y(), s.dir.x(), s.dir.y(),
                                 s.linear_q ? "linear" : "cubic", plate.material.total_thickness(), s.fold_band,
                                 s.fold_band_plateau);
                    std::fprintf(f, "step,ring,rho1,rho2,plate_h,line_h,GD_h,Gfold_h,Gtip_h,GfoldT_h,abs_plate_h,abs_fold_h,n_crease,plate_a,line_a,GD_a,Gfold_a,Gtip_a,fn_max,fT_max,dq_max,Gfold_band_h,Gtip_band_h,Gfold_band_a,Gtip_band_a,resid_h,hinge_h,GD_r,resid_band,hinge_band,Gfold_band_r,Gtip_band_r,GD_m,Gfold_band_m,Gtip_band_m,GD_mc,Gfold_band_mc,Gtip_band_mc,GD_s,Gfold_band_s,Gtip_band_s,hw1_s,vw_band_s,hw_band_s,dq_a_max\n");
                }
                for (size_t k = 0; k < res.size(); ++k)
                    std::fprintf(f, "%d,%zu,%.10g,%.10g,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                                 step, k, s.rings[k].rho1, s.rings[k].rho2, res[k].G1h, res[k].line_h, res[k].GD_h,
                                 res[k].Gfold_h, res[k].Gtip_h, res[k].GfoldT_h, res[k].abs_sum_h, res[k].abs_fold_h,
                                 res[k].n_crease, res[k].G1, res[k].line_a, res[k].GD_a, res[k].Gfold_a, res[k].Gtip_a,
                                 fn_max, fT_max, res[k].dq_max, res[k].Gfold_band_h, res[k].Gtip_band_h,
                                 res[k].Gfold_band_a, res[k].Gtip_band_a, res[k].resid_h, res[k].hinge_h, res[k].GD_r,
                                 res[k].resid_band, res[k].hinge_band, res[k].Gfold_band_r, res[k].Gtip_band_r,
                                 res[k].GD_m, res[k].Gfold_band_m, res[k].Gtip_band_m,
                                 res[k].GD_mc, res[k].Gfold_band_mc, res[k].Gtip_band_mc,
                                 res[k].GD_s, res[k].Gfold_band_s, res[k].Gtip_band_s, res[k].hw1,
                                 res[k].vw_band, res[k].hw_band, res[k].dq_a_max);
                std::fclose(f);
            }
            if (FILE* f = std::fopen((stem + "_crease.csv").c_str(), mode)) {
                if (!s.csv_started) {
                    std::fprintf(f, "# crease points: fold angle, hinge energy and configurational force, git_commit=%s\n", git_commit);
                    std::fprintf(f, "step,fold,i,X,Y,s,T_x,T_y,n_x,n_y,w,theta,theta0,psi,dpsi_ds,f_n,f_T,nBn_plus,nBn_minus,TBn_plus,TBn_minus,Nnn_plus,Nnn_minus,NTn_plus,NTn_minus,Mnn_plus,Mnn_minus,m_hinge\n");
                }
                for (const auto& c : cps)
                    std::fprintf(f, "%d,%d,%d,%.12g,%.12g,%.12g,%.9g,%.9g,%.9g,%.9g,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                                 step, c.fold, c.i, c.X.x(), c.X.y(), c.s, c.T.x(), c.T.y(), c.n.x(), c.n.y(), c.w,
                                 c.theta, c.theta0, c.psi, c.dpsi_ds, c.fn, c.fT,
                                 c.n.dot(c.Bp * c.n), c.n.dot(c.Bm * c.n), c.T.dot(c.Bp * c.n), c.T.dot(c.Bm * c.n),
                                 c.Nnn_p, c.Nnn_m, c.NTn_p, c.NTn_m, c.Mnn_p, c.Mnn_m, c.m_hinge);
                std::fclose(f);
            }
        }

        if (!s.csv.empty()) {
            FILE* f = std::fopen(s.csv.c_str(), s.csv_started ? "a" : "w");
            if (f) {
                if (!s.csv_started) {
                    std::fprintf(f, "# energy release rate, domain integral, git_commit=%s, tip_x=%.12g, tip_y=%.12g, d_x=%.12g, d_y=%.12g, q=%s, h=%.12g\n",
                                 git_commit, s.tip.x(), s.tip.y(), s.dir.x(), s.dir.y(),
                                 s.linear_q ? "linear" : "cubic", plate.material.total_thickness());
                    std::fprintf(f, "step,ring,rho1,rho2,G_x,G_y,G1,G2,G_abs,angle_deg,n_gp,abs_sum,G1_c,G2_c,G_abs_c,grad_q_sum,G1_h,G2_h,n_gp_h,G1_d,G2_d,G1_m,G2_m,G1_mc,G2_mc,G1_s,G2_s,vw1_s,vw1_h,Gband_h,Gband_a,G1b_h,Gband2_h,G2b_h\n");
                    s.csv_started = true;
                }
                for (size_t k = 0; k < res.size(); ++k)
                    std::fprintf(f, "%d,%zu,%.10g,%.10g,%.12e,%.12e,%.12e,%.12e,%.12e,%.8g,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                                 step, k, s.rings[k].rho1, s.rings[k].rho2, res[k].G.x(), res[k].G.y(),
                                 res[k].G1, res[k].G2, res[k].absG, res[k].angle_deg, res[k].ngp, res[k].abs_sum,
                                 res[k].G1c, res[k].G2c, res[k].absGc, res[k].grad_q_sum,
                                 res[k].G1h, res[k].G2h, res[k].ngp_h, res[k].G1d, res[k].G2d,
                                 res[k].G1m, res[k].G2m, res[k].G1mc, res[k].G2mc,
                                 res[k].G1s, res[k].G2s, res[k].vw1, res[k].vw1_h,
                                 res[k].Gband_h, res[k].Gband_a, res[k].G1b_h, res[k].Gband2_h, res[k].G2b_h);
                std::fclose(f);
            }
        }

        try {
            using namespace H5;
            Exception::dontPrint();
            H5File file(h5_path, H5F_ACC_RDWR);
            const int ncol = 32;
            std::vector<double> data;
            for (size_t k = 0; k < res.size(); ++k)
                data.insert(data.end(), {s.rings[k].rho1, s.rings[k].rho2, res[k].G.x(), res[k].G.y(), res[k].G1,
                                         res[k].G2, res[k].absG, res[k].angle_deg,
                                         static_cast<double>(res[k].ngp), res[k].abs_sum,
                                         res[k].G1c, res[k].G2c, res[k].absGc, res[k].grad_q_sum,
                                         res[k].G1h, res[k].G2h, static_cast<double>(res[k].ngp_h),
                                         res[k].G1d, res[k].G2d, res[k].G1m, res[k].G2m, res[k].G1mc, res[k].G2mc,
                                         res[k].G1s, res[k].G2s, res[k].vw1, res[k].vw1_h,
                                         res[k].Gband_h, res[k].Gband_a, res[k].G1b_h, res[k].Gband2_h, res[k].G2b_h});
            hsize_t dims[2] = { res.size(), static_cast<hsize_t>(ncol) };
            DataSpace space(2, dims);
            char name[96];
            std::snprintf(name, sizeof(name), "/step_%05d_%s", step, s.tag.c_str());
            DataSet ds = file.createDataSet(name, PredType::NATIVE_DOUBLE, space);
            ds.write(data.data(), PredType::NATIVE_DOUBLE);
            const char* cols = "rho1,rho2,G_x,G_y,G1,G2,G_abs,angle_deg,n_gp,abs_sum,G1_c,G2_c,G_abs_c,grad_q_sum,G1_h,G2_h,n_gp_h,G1_d,G2_d,G1_m,G2_m,G1_mc,G2_mc,G1_s,G2_s,vw1_s,vw1_h,Gband_h,Gband_a,G1b_h,Gband2_h,G2b_h";
            StrType stype(PredType::C_S1, std::strlen(cols) + 1);
            DataSpace scalar(H5S_SCALAR);
            ds.createAttribute("columns", stype, scalar).write(stype, cols);
            const double meta[5] = { s.tip.x(), s.tip.y(), s.dir.x(), s.dir.y(), s.linear_q ? 1.0 : 0.0 };
            hsize_t md[1] = { 5 };
            ds.createAttribute("tip_x,tip_y,d_x,d_y,linear_q", PredType::NATIVE_DOUBLE, DataSpace(1, md))
              .write(PredType::NATIVE_DOUBLE, meta);
            StrType gtype(PredType::C_S1, std::strlen(git_commit) + 1);
            ds.createAttribute("git_commit", gtype, scalar).write(gtype, git_commit);
            if (have_fold) {
                const int nf = 41;
                std::vector<double> fd;
                for (size_t k = 0; k < res.size(); ++k)
                    fd.insert(fd.end(), {s.rings[k].rho1, s.rings[k].rho2, res[k].G1h, res[k].line_h, res[k].GD_h,
                                         res[k].Gfold_h, res[k].Gtip_h, res[k].GfoldT_h, res[k].abs_sum_h,
                                         res[k].abs_fold_h, static_cast<double>(res[k].n_crease),
                                         res[k].G1, res[k].line_a, res[k].GD_a, res[k].Gfold_a, res[k].Gtip_a,
                                         res[k].dq_max, res[k].Gfold_band_h, res[k].Gtip_band_h,
                                         res[k].Gfold_band_a, res[k].Gtip_band_a, res[k].resid_h, res[k].hinge_h,
                                         res[k].GD_r, res[k].resid_band, res[k].hinge_band, res[k].Gfold_band_r,
                                         res[k].Gtip_band_r, res[k].GD_m, res[k].Gfold_band_m, res[k].Gtip_band_m,
                                         res[k].GD_mc, res[k].Gfold_band_mc, res[k].Gtip_band_mc,
                                         res[k].GD_s, res[k].Gfold_band_s, res[k].Gtip_band_s, res[k].hw1,
                                         res[k].vw_band, res[k].hw_band, res[k].dq_a_max});
                hsize_t fdims[2] = { res.size(), static_cast<hsize_t>(nf) };
                std::snprintf(name, sizeof(name), "/step_%05d_%s_fold", step, s.tag.c_str());
                DataSet fs = file.createDataSet(name, PredType::NATIVE_DOUBLE, DataSpace(2, fdims));
                fs.write(fd.data(), PredType::NATIVE_DOUBLE);
                const char* fcols = "rho1,rho2,plate_h,line_h,GD_h,Gfold_h,Gtip_h,GfoldT_h,abs_plate_h,abs_fold_h,n_crease,plate_a,line_a,GD_a,Gfold_a,Gtip_a,dq_max,Gfold_band_h,Gtip_band_h,Gfold_band_a,Gtip_band_a,resid_h,hinge_h,GD_r,resid_band,hinge_band,Gfold_band_r,Gtip_band_r,GD_m,Gfold_band_m,Gtip_band_m,GD_mc,Gfold_band_mc,Gtip_band_mc,GD_s,Gfold_band_s,Gtip_band_s,hw1_s,vw_band_s,hw_band_s,dq_a_max";
                StrType ftype(PredType::C_S1, std::strlen(fcols) + 1);
                fs.createAttribute("columns", ftype, scalar).write(ftype, fcols);
                const double fmax[4] = { fn_max, fT_max, s.fold_band, s.fold_band_plateau };
                hsize_t f2[1] = { 4 };
                fs.createAttribute("fn_max,fT_max,fold_band,fold_band_plateau", PredType::NATIVE_DOUBLE, DataSpace(1, f2)).write(PredType::NATIVE_DOUBLE, fmax);
                fs.createAttribute("git_commit", gtype, scalar).write(gtype, git_commit);
            }
        } catch (...) {
            printf("!!! WARNING: could not write the G dataset for step %d to %s\n", step, h5_path.c_str());
        }
    }
}
