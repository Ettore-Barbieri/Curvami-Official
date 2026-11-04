#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "curvami/MeshfreeData.h"

namespace Kinematics {

    // Natural curvature kappa_bar(X): a prescribed symmetric 2x2 field [1/m], the
    // curvature at which the sheet carries no moment (Crack_Brief_for_Code.md, E7;
    // notes JIntegral_KL.tex, Eqs. 54-56). The elastic bending strain is
    // E1_el = E1 + kappa_bar, with E1 the geometric one. It does not depend on the
    // unknowns, so it changes the residual's strain only, not the tangent's form.
    // One field for the whole plate, set once from the config ("natural_curvature")
    // before any assembly; inactive by default, and then E1_el is E1 itself.
    //   band: kappa_bar = sign * theta0 * g(X) eX (x) eX,
    //         g(X) = exp(-((X - x0) / w)^2) / (sqrt(pi) w),  integral of g = 1,
    //   a straight band along Y centred at X = x0, total turning angle theta0, w the
    //   distance at which g falls to 1/e of its peak. sign = -1 is a mountain (ridge
    //   towards +z, checked on an unloaded run), +1 a valley.
    struct NaturalCurvature {
        bool active = false;
        double x0 = 0.0, width = 1.0, theta0 = 0.0, sign = -1.0;

        double g(double X) const {
            const double t = (X - x0) / width;
            return std::exp(-t * t) / (1.7724538509055160273 * width);
        }
        Eigen::Matrix2d kappa(const Eigen::Vector2d& X) const {
            Eigen::Matrix2d k = Eigen::Matrix2d::Zero();
            if (active) k(0, 0) = sign * theta0 * g(X.x());
            return k;
        }
        // directional derivative d . grad kappa_bar
        Eigen::Matrix2d dkappa(const Eigen::Vector2d& X, const Eigen::Vector2d& d) const {
            Eigen::Matrix2d k = Eigen::Matrix2d::Zero();
            if (active) k(0, 0) = sign * theta0 * g(X.x()) * (-2.0 * (X.x() - x0) / (width * width)) * d.x();
            return k;
        }
    };

    inline NaturalCurvature& naturalCurvature() {
        static NaturalCurvature field;
        return field;
    }

    // Structure to hold the derived kinematic quantities at a single point
    struct LocalKinematics {
        Eigen::Matrix<double, 3, 2> F0;          // Mid-surface deformation gradient [3x2 block]
        Eigen::Vector3d b;                      // Enslaved unit normal vector
        Eigen::Matrix2d C0_inv;                 // Inverse of the metric tensor
        Eigen::Matrix2d E0;                     // Membrane strain (Green-Lagrange)
        Eigen::Matrix2d E1;                     // Bending strain (Curvature), geometric
        Eigen::Matrix2d E1_el;                  // Elastic bending strain: E1 + natural curvature
        Eigen::Vector3d r0XX;
        Eigen::Vector3d r0YY;
        Eigen::Vector3d r0XY;
    };

    // Computes kinematics for a single Gauss point
    inline LocalKinematics compute(const GaussPointData& gp, const std::vector<Node>& all_nodes) {
        LocalKinematics kin;

        // 1. Compute Mid-surface gradients (r0,X and r0,Y)
        // r0 = X + u. Therefore, dr0/dX = dX/dX + du/dX = [1; 0; 0] + du/dX
        Eigen::Vector3d r0X = Eigen::Vector3d::UnitX();
        Eigen::Vector3d r0Y = Eigen::Vector3d::UnitY();

        // 2nd derivatives (Hessian) for curvature
        Eigen::Vector3d r0XX = Eigen::Vector3d::Zero();
        Eigen::Vector3d r0YY = Eigen::Vector3d::Zero();
        Eigen::Vector3d r0XY = Eigen::Vector3d::Zero();

        // Sum contributions from neighboring nodes using RKPM shape function derivatives
        for (size_t i = 0; i < gp.neighbor_node_ids.size(); ++i) {
            int node_idx = gp.neighbor_node_ids[i];
            const Eigen::Vector3d& u = all_nodes[node_idx].u;

            // First derivatives
            r0X += u * gp.dPsi[i].x();
            r0Y += u * gp.dPsi[i].y();

            // Second derivatives (Required for KL Bending)
            r0XX += u * gp.ddPsi[i](0, 0);
            r0YY += u * gp.ddPsi[i](1, 1);
            r0XY += u * gp.ddPsi[i](0, 1);
        }

        // 2. Deformation Gradient F0 [cite: 26]
        kin.F0.col(0) = r0X;
        kin.F0.col(1) = r0Y;

        // 3. Metric Tensor C0 and its Inverse [cite: 42]
        Eigen::Matrix2d C0;
        C0(0, 0) = r0X.dot(r0X);
        C0(1, 1) = r0Y.dot(r0Y);
        C0(0, 1) = C0(1, 0) = r0X.dot(r0Y);
        
        kin.C0_inv = C0.inverse();

        // 4. Enslaved Unit Normal b
        Eigen::Vector3d n = r0X.cross(r0Y);
        double area_stretch = n.norm(); // The Jacobian J
        kin.b = n / area_stretch;

        // 5. Membrane Strain E0 = 0.5 * (C0 - I) [cite: 42]
        kin.E0 = 0.5 * (C0 - Eigen::Matrix2d::Identity());

        // 6. Bending Strain E1 (Negative of Second Fundamental Form) [cite: 36, 43]
        // kappa_ab = r0,ab . b
        kin.E1(0, 0) = -r0XX.dot(kin.b);
        kin.E1(1, 1) = -r0YY.dot(kin.b);
        kin.E1(0, 1) = kin.E1(1, 0) = -r0XY.dot(kin.b);
        kin.E1_el = kin.E1;
        if (naturalCurvature().active) kin.E1_el += naturalCurvature().kappa(gp.X_ref);
        kin.r0XX = r0XX;
        kin.r0YY = r0YY;
        kin.r0XY = r0XY;
        return kin;
    }
}
