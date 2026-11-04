#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "PlateData.h"
#include "Kinematics.h"

namespace Constitutive {

    // 1. Pre-compute ABD matrices using Classical Laminate Theory
    inline void computeABD(LaminateProperties& mat) {
        double h_total = mat.total_thickness();
        double z_current = -h_total / 2.0;

        mat.A.setZero(); mat.B.setZero(); mat.D.setZero();

        for (const auto& ply : mat.plies) {
            // Reduced Stiffness Matrix (Q) for orthotropic material
            double nu21 = (ply.E2 / ply.E1) * ply.nu12;
            double den = 1.0 - ply.nu12 * nu21;
            
            Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
            Q(0,0) = ply.E1 / den;
            Q(1,1) = ply.E2 / den;
            Q(0,1) = Q(1,0) = (ply.nu12 * ply.E2) / den;
            Q(2,2) = ply.G12;

            // Transform Q to Q-bar based on ply angle
            double theta = ply.angle_deg * M_PI / 180.0;
            double m = std::cos(theta);
            double n = std::sin(theta);
            
            Eigen::Matrix3d Qbar;
            double m2 = m*m; double n2 = n*n; double m4 = m2*m2; double n4 = n2*n2;
            
            Qbar(0,0) = Q(0,0)*m4 + 2.0*(Q(0,1) + 2.0*Q(2,2))*m2*n2 + Q(1,1)*n4;
            Qbar(1,1) = Q(0,0)*n4 + 2.0*(Q(0,1) + 2.0*Q(2,2))*m2*n2 + Q(1,1)*m4;
            Qbar(0,1) = Qbar(1,0) = (Q(0,0) + Q(1,1) - 4.0*Q(2,2))*m2*n2 + Q(0,1)*(m4 + n4);
            Qbar(0,2) = Qbar(2,0) = (Q(0,0) - Q(0,1) - 2.0*Q(2,2))*n*m*m2 + (Q(0,1) - Q(1,1) + 2.0*Q(2,2))*n2*n*m;
            Qbar(1,2) = Qbar(2,1) = (Q(0,0) - Q(0,1) - 2.0*Q(2,2))*m*n*n2 + (Q(0,1) - Q(1,1) + 2.0*Q(2,2))*m2*m*n;
            Qbar(2,2) = (Q(0,0) + Q(1,1) - 2.0*Q(0,1) - 2.0*Q(2,2))*m2*n2 + Q(2,2)*(m4 + n4);

            // Integrate through the thickness to build A, B, D matrices
            double z_next = z_current + ply.thickness;
            mat.A += Qbar * (z_next - z_current);
            mat.B += 0.5 * Qbar * (z_next*z_next - z_current*z_current);
            mat.D += (1.0/3.0) * Qbar * (z_next*z_next*z_next - z_current*z_current*z_current);
            
            z_current = z_next;
        }
    }

    struct Resultants {
        Eigen::Vector3d N, M;
    };

    // 2. Evaluate Generalized Stresses (Resultants)
    inline void evaluate(const Kinematics::LocalKinematics& kin,
                        const LaminateProperties& mat,
                        Resultants& res)
    {
        // Engineering strains (Voigt): [e11, e22, 2*e12] [cite: 156-158]
        Eigen::Vector3d e0, e1;
        e0 << kin.E0(0,0), kin.E0(1,1), 2.0 * kin.E0(0,1);
        // elastic bending strain: the geometric one plus the natural curvature, if any
        e1 << kin.E1_el(0,0), kin.E1_el(1,1), 2.0 * kin.E1_el(0,1);

        // Generalized constitutive relation
        res.N = mat.A * e0 + mat.B * e1;
        res.M = mat.B.transpose() * e0 + mat.D * e1;
    }
}
