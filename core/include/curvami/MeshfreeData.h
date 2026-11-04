#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

// 1. The Node Structure (Just 3 DOFs!)
struct Node {
    int id;
    Eigen::Vector2d X_ref; // Reference mid-surface coordinates (X, Y)
    
    // We only need translations (u, v, w).
    // Rotations are mathematically enslaved to the mid-surface in KL theory.
    Eigen::Vector3d u;
};

// 2. The Gauss Point Structure
struct GaussPointData {
    int id;
    int entity_tag = -1;     // Stores the Gmsh Physical Group ID!
    
    Eigen::Vector2d X_ref;   // Location of the integration point
    double weight;           // Integration weight (dA)

    // Boundary geometry (zero-initialized; only meaningful for edge/fold GPs)
    Eigen::Vector2d normal  = Eigen::Vector2d::Zero(); // Outward unit normal  (nx, ny)
    Eigen::Vector2d tangent = Eigen::Vector2d::Zero(); // Unit tangent along edge (tx, ty)

    // FRNN Spatial Search Data
    std::vector<int> neighbor_node_ids;
    
    // RKPM Shape Functions (Pre-computed for each neighbor)
    std::vector<double> Psi;
    std::vector<Eigen::Vector2d> dPsi;  // [dPsi/dX, dPsi/dY]
    std::vector<Eigen::Matrix2d> ddPsi; // [d2Psi/dX2, d2Psi/dY2; d2Psi/dXdY, d2Psi/dYdX]
};
