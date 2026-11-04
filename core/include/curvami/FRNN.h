#pragma once
#include <vector>
#include <cmath>
#include <unordered_map>
#include <Eigen/Dense>
#include "curvami/MeshfreeData.h"

class GlobalFRNN {
private:
    double cell_size;
    int cols;
    Eigen::Vector2d min_bounds;

    inline int getCellIndex(const Eigen::Vector2d& pos) const {
        int x = static_cast<int>(std::floor((pos.x() - min_bounds.x()) / cell_size));
        int y = static_cast<int>(std::floor((pos.y() - min_bounds.y()) / cell_size));
        return y * cols + x;
    }

public:
    GlobalFRNN(double radius) : cell_size(radius) {}

    void execute(std::vector<GaussPointData>& gps, const std::vector<Node>& nodes,
                 double rho, std::vector<size_t>& ig, std::vector<size_t>& js, std::vector<double>& s)
    {
        // ---> ADD THESE 3 LINES TO PREVENT MEMORY CORRUPTION <---
                ig.clear();
                js.clear();
                s.clear();
        
        if (nodes.empty() || gps.size() == 0) return;

        min_bounds = nodes[0].X_ref;
        Eigen::Vector2d max_bounds = nodes[0].X_ref;
        for (const auto& n : nodes) {
            min_bounds = min_bounds.cwiseMin(n.X_ref);
            max_bounds = max_bounds.cwiseMax(n.X_ref);
        }
        cols = static_cast<int>(std::ceil((max_bounds.x() - min_bounds.x()) / cell_size)) + 1;

        std::unordered_map<int, std::vector<size_t>> grid;
        for (size_t i = 0; i < nodes.size(); ++i) grid[getCellIndex(nodes[i].X_ref)].push_back(i);

        // ig/js/s grow to one entry per (Gauss point, neighbour) pair - about 7.9M on
        // SquareTwist90, ~190 MB across the three - and geometric growth can leave up
        // to twice that resident. Size them up front from the expected neighbour count
        // (disc of radius rho at the mean node density) so the doubling never happens.
        const Eigen::Vector2d span = (max_bounds - min_bounds).cwiseMax(1e-30);
        const double density  = static_cast<double>(nodes.size()) / (span.x() * span.y());
        const double per_gp_f = 3.14159265358979323846 * rho * rho * density * 1.15 + 8.0;
        const size_t per_gp   = static_cast<size_t>(per_gp_f);
        const size_t est      = per_gp * gps.size();

        ig.reserve(est); js.reserve(est); s.reserve(est);

        double rho2 = rho * rho;
        for (size_t iq = 0; iq < gps.size(); ++iq) {
            gps[iq].neighbor_node_ids.clear();
            gps[iq].neighbor_node_ids.reserve(per_gp);
            int qx = static_cast<int>(std::floor((gps[iq].X_ref.x() - min_bounds.x()) / cell_size));
            int qy = static_cast<int>(std::floor((gps[iq].X_ref.y() - min_bounds.y()) / cell_size));

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    auto it = grid.find((qy + dy) * cols + (qx + dx));
                    if (it != grid.end()) {
                        for (size_t node_idx : it->second) {
                            double dist2 = (nodes[node_idx].X_ref - gps[iq].X_ref).squaredNorm();
                            if (dist2 <= rho2) {
                                ig.push_back(iq); js.push_back(node_idx); s.push_back(std::sqrt(dist2) / rho);
                                gps[iq].neighbor_node_ids.push_back(static_cast<int>(node_idx));
                            }
                        }
                    }
                }
            }
        }
    }
};
