#pragma once
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "Meshless.h"
#include "curvami/GmshReader.h"
#include "curvami/Quadrature.h"
#include "curvami/FRNN.h"
#include "curvami/GlobalRKPM.h"

namespace SetupHelper {

    inline void loadMeshAndQuadrature(const std::string& filename, Meshless& plate,
                                      std::vector<GmshReader::Element>& elements,
                                      std::map<int, int>& edge_tag_to_idx,
                                      int TAG_CREASE = 5,
                                      std::vector<Node>* raw_nodes_out = nullptr)
    {
        std::vector<Node> raw_nodes;
        GmshReader::readMesh(filename, raw_nodes, elements);

        if (raw_nodes_out) {
            *raw_nodes_out = raw_nodes;
        }
        
        std::vector<GaussPointData> face_gps, edge_gps, point_gps;
        Quadrature::generate(raw_nodes, elements, plate.nodes,
                             face_gps, edge_gps, point_gps,
                             1, 3);
        plate.faces.gps = face_gps;

        std::map<int, IntegrationDomain> point_map, edge_map;
        for(auto& gp : point_gps) point_map[gp.entity_tag].gps.push_back(gp);
        for(auto& gp : edge_gps)  edge_map[gp.entity_tag].gps.push_back(gp);
        std::cout << "\n >> Using Quadrature::generate(..., line_degree = 1) for all line domains.\n";
        std::cout << "\n--- Discovered Gmsh Physical Tags ---\n";
        for(auto& kv : point_map) std::cout << "Point Tag Found: " << kv.first << " (" << kv.second.gps.size() << " GPs)\n";
        for(auto& kv : edge_map)  std::cout << "Edge Tag Found:  " << kv.first << " (" << kv.second.gps.size() << " GPs)\n";
        std::cout << "-------------------------------------\n\n";

        for(auto& kv : point_map) plate.points.push_back(kv.second);
        for(auto& kv : edge_map) {
            edge_tag_to_idx[kv.first] = static_cast<int>(plate.edges.size());
            plate.edges.push_back(kv.second);
        }
    }

    // A face Gauss point that lands EXACTLY on a crease gets no enrichment at all:
    // evaluateIndicator returns is_valid = false and f = 0 there, so that point
    // stitches the two facets together instead of separating them. It happens
    // whenever a straight crease runs parallel to the node rows through the middle
    // of a cell - a 3x3 rule puts its centre row exactly on the cut, 76 of 9594
    // points on the straight control - and the run then bends the facets instead
    // of folding them, silently (E_ben/E_hin = 5.9 where it should be 0).
    //
    // The integrand is piecewise across the cut, so the correct treatment of such a
    // point is half its weight on each side. Split it the same way addFold already
    // offsets the crease's own left/right points, with the same 1e-7.
    inline int splitGaussPointsOnCreases(Meshless& plate,
                                         const std::vector<Enrichment::FoldLine>& creases,
                                         double eps = 1e-9, double s = 1e-7) {
        std::vector<GaussPointData> extra;
        int split = 0;

        for (auto& gp : plate.faces.gps) {
            double best_d = std::numeric_limits<double>::infinity();
            Eigen::Vector2d best_n = Eigen::Vector2d::Zero();

            for (const auto& crease : creases) {
                const size_t N = crease.vertices.size();
                for (size_t i = 0; i + 1 < N; ++i) {
                    const Eigen::Vector2d A = crease.vertices[i];
                    const Eigen::Vector2d B = crease.vertices[i + 1];
                    const Eigen::Vector2d v = B - A;
                    const double v2 = v.squaredNorm();
                    if (v2 == 0.0) continue;
                    const double t = std::max(0.0, std::min(1.0, (gp.X_ref - A).dot(v) / v2));
                    const Eigen::Vector2d C = A + t * v;
                    const double d = (gp.X_ref - C).norm();
                    if (d < best_d) {
                        best_d = d;
                        best_n = Eigen::Vector2d(-v.y(), v.x()) / std::sqrt(v2);
                    }
                }
            }

            if (best_d < eps && best_n.squaredNorm() > 0.0) {
                GaussPointData twin = gp;
                gp.weight *= 0.5;
                twin.weight *= 0.5;
                gp.X_ref   += s * best_n;
                twin.X_ref -= s * best_n;
                extra.push_back(twin);
                ++split;
            }
        }

        plate.faces.gps.insert(plate.faces.gps.end(), extra.begin(), extra.end());
        if (split > 0) {
            std::cout << "\n >>> " << split << " face Gauss point(s) lay on a crease and were"
                      << " split in two, half weight either side (they would otherwise carry no"
                      << " enrichment and tie the facets together)." << std::endl;
        }
        return split;
    }

    // RKPM shape functions for every integration domain, cut by `creases`. The
    // neighbour lists come from the support radius alone, so they - and the
    // stiffness sparsity pattern - do not depend on the creases; progressive
    // creasing calls this again at every step with the crease cut short.
    inline void recomputeShapeFunctions(Meshless& plate, const std::vector<Enrichment::FoldLine>& creases) {
        std::cout << "Computing RKPM Shape Functions... " << std::flush;
        
        GlobalFRNN frnn(plate.rho);
        std::vector<size_t> ig, js;
        std::vector<double> s;
        
        frnn.execute(plate.faces.gps, plate.nodes, plate.rho, ig, js, s);
        GlobalRKPM::solve(plate.faces.gps, plate.nodes, ig, js, s, plate.rho, plate.order, creases);
        
        for(auto& pt_dom : plate.points) {
            frnn.execute(pt_dom.gps, plate.nodes, plate.rho, ig, js, s);
            GlobalRKPM::solve(pt_dom.gps, plate.nodes, ig, js, s, plate.rho, plate.order, creases);
        }
        for(auto& edge_dom : plate.edges) {
            frnn.execute(edge_dom.gps, plate.nodes, plate.rho, ig, js, s);
            GlobalRKPM::solve(edge_dom.gps, plate.nodes, ig, js, s, plate.rho, plate.order, creases);
        }
        for(auto& f : plate.folds) {
            frnn.execute(f.left.gps, plate.nodes, plate.rho, ig, js, s);
            GlobalRKPM::solve(f.left.gps, plate.nodes, ig, js, s, plate.rho, plate.order, creases);
            
            frnn.execute(f.right.gps, plate.nodes, plate.rho, ig, js, s);
            GlobalRKPM::solve(f.right.gps, plate.nodes, ig, js, s, plate.rho, plate.order, creases);
        }
        std::cout << "Done!\n" << std::endl;
    }

    inline void computeAllShapeFunctions(Meshless& plate, const std::vector<Enrichment::FoldLine>& creases) {
        splitGaussPointsOnCreases(plate, creases);
        recomputeShapeFunctions(plate, creases);
    }
}
