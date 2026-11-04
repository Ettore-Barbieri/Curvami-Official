#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "curvami/MeshfreeData.h"

// Plate-layer data: laminate, boundary conditions, integration domains and folds.
// Split from MeshfreeData.h, which keeps the generic Node and GaussPointData.

// 3. New Structure for a Single Orthotropic Ply
struct Ply {
    double E1, E2, nu12, G12; // Material constants
    double angle_deg;         // Ply orientation in degrees
    double thickness;         // Physical thickness of this layer
};

// 4. Generalized Laminate Properties
struct LaminateProperties {
    std::vector<Ply> plies;
    
    // Pre-computed ABD matrices for Classical Laminate Theory
    // A: Membrane, B: Coupling, D: Bending
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d B = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d D = Eigen::Matrix3d::Zero();

    // Helper to get total laminate thickness
    double total_thickness() const {
        double h = 0;
        for (const auto& p : plies) h += p.thickness;
        return h;
    }
};

struct BoundaryCondition {
    bool active_u = false, active_v = false, active_w = false;
    double val_u = 0, val_v = 0, val_w = 0;

    bool active_rot = false; // C1/Normal derivative constraint
    double val_rot = 0;

    // External loads
    Eigen::Vector3d force = Eigen::Vector3d::Zero();

    // Edge bending moment per unit length (edges only), a dead load conjugate to
    // the normal slope dw/dn that "rot" constrains: it does the work
    // moment * dw/dn per unit length, n the OUTWARD edge normal in the reference
    // plane. Linearised rotation: exact for small slopes.
    double moment = 0.0;
};

struct IntegrationDomain {
    std::string name;
    std::vector<GaussPointData> gps;
    BoundaryCondition bc;
};

// How the mid-surface displacement is tied across a fold (Crack_Brief_for_Code.md, E0).
// Only the displacement is tied; the hinge is the same in every variant.
struct FoldTie {
    enum Type { Pointwise, Subset, MortarP0, MortarP1 };
    Type type = Pointwise;   // Pointwise: u+ = u- at every crease Gauss point (the default)
    int every = 1;           // Subset: the same constraint at every `every`-th crease point
    // Integrated multipliers: sum_k w_k phi_j(s_k) (u+ - u-)(X_k) = 0 for every multiplier
    // function phi_j, w_k the crease Gauss weights.
    //   MortarP0: phi_j = 1 on the j-th segment of length `length` along the crease
    //   MortarP1: phi_j = hat function at points spaced `length` (adjusted so that a
    //             whole number of intervals spans the crease)
    double length = 0.0;
};

struct Fold {
    int crease_tag = -1; // Store the tag for terminal JSON output
    FoldTie tie;
    
    IntegrationDomain left;
    IntegrationDomain right;
    double Kt;      // Torsional stiffness
    double Ang0;    // Target dihedral angle
    bool flagRig;   // If true, enforced as hard constraint
    double fold_sign = 1.0; // +1.0 for Valley, -1.0 for Mountain
    
    // Proportional Loading Targets
    double angle_start = 0.0;
    double angle_target = 0.0;
    bool is_driver = false; // ADD THIS

    // Progressive creasing: per crease Gauss point, indexed like left.gps.
    // Empty (the default) means every point is active with rest angle Ang0, which
    // is exactly the behaviour before these existed.
    std::vector<double> gp_s;       // arc length along the full crease path
    std::vector<double> gp_theta0;  // signed rest angle (fold_sign already applied)
    std::vector<char>   gp_active;  // 0: no crease here yet - no hinge, no constraint
};
