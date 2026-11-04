#pragma once
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <limits>
#include "curvami/MeshfreeData.h"
#include "curvami/GmshReader.h"
#include "curvami/Enrichment.h"
#include "curvami/FRNN.h"

namespace MeshUtils {

    // Helper: 2D Cross Product for Convex Hull
    inline double crossProduct(const Eigen::Vector2d& O, const Eigen::Vector2d& A, const Eigen::Vector2d& B) {
        return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
    }

    // Helper: Compute Convex Hull (Monotone Chain Algorithm) - Mimics MATLAB's convhull
    inline std::vector<Eigen::Vector2d> computeConvexHull(std::vector<Eigen::Vector2d> P) {
        size_t n = P.size(), k = 0;
        if (n <= 3) return P;
        std::vector<Eigen::Vector2d> H(2 * n);
        std::sort(P.begin(), P.end(), [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
            return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
        });
        for (size_t i = 0; i < n; ++i) {
            while (k >= 2 && crossProduct(H[k - 2], H[k - 1], P[i]) <= 0) k--;
            H[k++] = P[i];
        }
        for (size_t i = n - 1, t = k + 1; i > 0; --i) {
            while (k >= t && crossProduct(H[k - 2], H[k - 1], P[i - 1]) <= 0) k--;
            H[k++] = P[i - 1];
        }
        H.resize(k - 1);
        return H;
    }

    // Helper: Check if a point lies on the segments of the convex hull (Mimics MATLAB's inpolygon boundary check)
    inline bool isOnHullBoundary(const Eigen::Vector2d& pt, const std::vector<Eigen::Vector2d>& hull, double tol = 1e-4) {
        for (size_t i = 0; i < hull.size(); ++i) {
            Eigen::Vector2d A = hull[i];
            Eigen::Vector2d B = hull[(i + 1) % hull.size()];
            Eigen::Vector2d AB = B - A;
            double len2 = AB.squaredNorm();
            if (len2 == 0) continue;
            // Project point onto the line segment AB
            double t = std::max(0.0, std::min(1.0, (pt - A).dot(AB) / len2));
            Eigen::Vector2d proj = A + t * AB;
            if ((pt - proj).norm() <= tol) return true;
        }
        return false;
    }

    inline Enrichment::FoldLine extractCurve(const std::vector<GmshReader::Element>& elements, const std::vector<Node>& nodes, const std::vector<int>& target_entity_tags) {
        Enrichment::FoldLine fold;
        std::map<int, std::vector<int>> adj;
        
        auto contains = [](const std::vector<int>& vec, int val) {
            return std::find(vec.begin(), vec.end(), val) != vec.end();
        };

        // 1. Build adjacency for the combined fold lines
        for (const auto& el : elements) {
            if (el.type == 1 && contains(target_entity_tags, el.entity_tag)) {
                int n1 = el.node_indices[0]; int n2 = el.node_indices[1];
                adj[n1].push_back(n2); adj[n2].push_back(n1);
            }
        }
        if (adj.empty()) return fold;
        
        // 2. Find start node and order the vertices
        int start_node = -1;
        for (const auto& pair : adj) { if (pair.second.size() == 1) { start_node = pair.first; break; } }
        if (start_node == -1) start_node = adj.begin()->first;
        
        int current = start_node; int prev = -1;
        std::vector<int> ordered_nodes;
        while (true) {
            ordered_nodes.push_back(current);
            int next = -1;
            for (int neighbor : adj[current]) { if (neighbor != prev) { next = neighbor; break; } }
            
            if (next == -1 || next == start_node) {
                if (next == start_node) {
                    fold.is_closed = true; // <-- DETECT LOOP CLOSURE
                }
                break;
            }
            prev = current; current = next;
        }
        // The walk follows one chain. A tag set that is disconnected, or that branches,
        // leaves nodes unvisited and they are silently dropped from the crease geometry.
        if (ordered_nodes.size() < adj.size()) {
            std::cerr << "!!! WARNING: crease tags [";
            for (int t : target_entity_tags) std::cerr << t << " ";
            std::cerr << "] contain " << adj.size() << " nodes but only "
                      << ordered_nodes.size() << " form a single chain; "
                      << (adj.size() - ordered_nodes.size())
                      << " were dropped. Disconnected or branching crease groups are not"
                         " supported - split them into separate cuts." << std::endl;
        }

        for(int idx : ordered_nodes) fold.vertices.push_back(nodes[idx].X_ref);

        if (fold.is_closed) {
            fold.vertices.push_back(fold.vertices.front()); // <-- CLOSE THE GEOMETRIC SEGMENT GAP
        }
        
        // 3. GEOMETRIC CONVEX HULL BOUNDARY DETECTION (MATLAB Match)
        std::vector<Eigen::Vector2d> all_pts;
        for (const auto& n : nodes) all_pts.push_back(n.X_ref);
        std::vector<Eigen::Vector2d> hull = computeConvexHull(all_pts);

        int first_idx = ordered_nodes.front();
        int last_idx = ordered_nodes.back();

        bool start_on_boundary = isOnHullBoundary(nodes[first_idx].X_ref, hull, 1e-4);
        bool end_on_boundary = isOnHullBoundary(nodes[last_idx].X_ref, hull, 1e-4);
        
        // --- ADDED ENHANCED DEBUG PRINT ---
        std::cout << "   -> Geometric Cut/Fold [ ";
        for (int t : target_entity_tags) std::cout << t << " ";
        std::cout << "]\n      Start (" << std::fixed << std::setprecision(3) << nodes[first_idx].X_ref.x() << ", " << nodes[first_idx].X_ref.y() << "): " << (start_on_boundary ? "TRUE" : "FALSE")
                  << "\n      End   (" << nodes[last_idx].X_ref.x() << ", " << nodes[last_idx].X_ref.y() << "): " << (end_on_boundary ? "TRUE" : "FALSE") << std::endl;
        // -------------------------

        fold.extend_start = start_on_boundary;
        fold.extend_end   = end_on_boundary;
        
        return fold;
    }

    // Median 3rd-nearest-neighbour spacing, scaled by (order + 3).
    //
    // The direct form measured every node against every other one - O(N^2), 4.6 s at
    // 8558 nodes and roughly ten minutes at 100k. A uniform grid with an expanding
    // ring search gives the identical answer: after scanning every cell within
    // Chebyshev radius r of the query cell, anything still unseen lies at least
    // r*cell away, so once the running 4th-best distance is within that bound no
    // unscanned node can beat it and the search is provably complete.
    inline double computeSupportRadius(const std::vector<Node>& nodes, int order) {
        const size_t N = nodes.size();
        if (N < 4) return 0.1;

        Eigen::Vector2d lo = nodes[0].X_ref, hi = nodes[0].X_ref;
        for (const auto& n : nodes) { lo = lo.cwiseMin(n.X_ref); hi = hi.cwiseMax(n.X_ref); }
        const Eigen::Vector2d ext = (hi - lo).cwiseMax(1e-30);

        double cell = std::sqrt(ext.x() * ext.y() / static_cast<double>(N)) * 1.5;
        if (!(cell > 0.0)) cell = ext.maxCoeff();

        const int nx = std::max(1, static_cast<int>(ext.x() / cell) + 1);
        const int ny = std::max(1, static_cast<int>(ext.y() / cell) + 1);

        auto cell_x = [&](const Eigen::Vector2d& p) {
            return std::min(nx - 1, std::max(0, static_cast<int>((p.x() - lo.x()) / cell))); };
        auto cell_y = [&](const Eigen::Vector2d& p) {
            return std::min(ny - 1, std::max(0, static_cast<int>((p.y() - lo.y()) / cell))); };

        std::vector<std::vector<int>> grid(static_cast<size_t>(nx) * static_cast<size_t>(ny));
        for (size_t i = 0; i < N; ++i) {
            grid[static_cast<size_t>(cell_y(nodes[i].X_ref)) * nx + cell_x(nodes[i].X_ref)]
                .push_back(static_cast<int>(i));
        }

        std::vector<double> h_I_raw(N);

        #pragma omp parallel for schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(N); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            const Eigen::Vector2d p = nodes[i].X_ref;
            const int cx = cell_x(p), cy = cell_y(p);

            // four smallest distances so far, ascending; the node itself contributes 0
            double best[4] = { std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity() };
            auto consider = [&](double d) {
                if (!(d < best[3])) return;
                int k = 3;
                while (k > 0 && best[k - 1] > d) { best[k] = best[k - 1]; --k; }
                best[k] = d;
            };

            for (int r = 0; ; ++r) {
                const int x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r;
                for (int gy = std::max(0, y0); gy <= std::min(ny - 1, y1); ++gy) {
                    for (int gx = std::max(0, x0); gx <= std::min(nx - 1, x1); ++gx) {
                        // only the new ring; the interior was covered by earlier r
                        if (r > 0 && gx != x0 && gx != x1 && gy != y0 && gy != y1) continue;
                        for (int j : grid[static_cast<size_t>(gy) * nx + gx]) {
                            consider((p - nodes[j].X_ref).norm());
                        }
                    }
                }
                if (best[3] <= static_cast<double>(r) * cell) break;
                if (x0 < 0 && y0 < 0 && x1 >= nx && y1 >= ny) break;   // whole grid seen
            }
            h_I_raw[i] = best[3];
        }

        std::nth_element(h_I_raw.begin(), h_I_raw.begin() + h_I_raw.size() / 2, h_I_raw.end());
        double c_scale = static_cast<double>(order + 3);
        return c_scale * h_I_raw[h_I_raw.size() / 2];
    }

}
