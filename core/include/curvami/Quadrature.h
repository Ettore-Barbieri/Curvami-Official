#pragma once
#include <vector>
#include <cmath>
#include <set>
#include <iostream>
#include <Eigen/Dense>
#include "curvami/MeshfreeData.h"
#include "curvami/GmshReader.h"

namespace Quadrature {

    struct TriRule { double u, v, w, weight; };

    inline void makeLineRule(int line_degree,
                             std::vector<double>& l_pts,
                             std::vector<double>& l_wts)
    {
        if (line_degree <= 1) {
            l_pts = {0.0};
            l_wts = {2.0};
        } else {
            l_pts = {-0.77459666924148337704, 0.0, 0.77459666924148337704};
            l_wts = {0.55555555555555555556,
                     0.88888888888888888889,
                     0.55555555555555555556};
        }
    }

    inline void makeQuadTensorRule(int face_degree,
                                   std::vector<double>& q_pts,
                                   std::vector<double>& q_wts)
    {
        if (face_degree <= 1) {
            q_pts = {0.0};
            q_wts = {2.0};
        } else if (face_degree == 2) {
            q_pts = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
            q_wts = {1.0, 1.0};
        } else {
            // MATLAB deg_integration = 3 for quads -> 3x3 tensor-product rule.
            q_pts = {-std::sqrt(3.0 / 5.0), 0.0, std::sqrt(3.0 / 5.0)};
            q_wts = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
        }
    }

    inline std::vector<TriRule> makeTriRule(int face_degree)
    {
        if (face_degree <= 1) {
            return {{1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0}};
        }

        return {
            {2.0/3.0, 1.0/6.0, 1.0/6.0, 1.0/3.0},
            {1.0/6.0, 2.0/3.0, 1.0/6.0, 1.0/3.0},
            {1.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0}
        };
    }

    inline void generate(
        const std::vector<Node>& raw_nodes,
        const std::vector<GmshReader::Element>& elements,
        std::vector<Node>& active_nodes,
        std::vector<GaussPointData>& face_gps,
        std::vector<GaussPointData>& edge_gps,
        std::vector<GaussPointData>& point_gps,
        int line_degree = 3,
        int face_degree = 2)
    {
        face_gps.clear();
        edge_gps.clear();
        point_gps.clear();
        active_nodes.clear();

        std::set<int> active_node_indices;
        int gp_id_counter = 0;

        std::vector<double> l_pts, l_wts;
        makeLineRule(line_degree, l_pts, l_wts);

        std::vector<double> q_pts, q_wts;
        makeQuadTensorRule(face_degree, q_pts, q_wts);

        std::vector<TriRule> t_pts = makeTriRule(face_degree);

        for (const auto& el : elements) {
            // --- 0D POINTS (Gmsh Type 15) ---
            if (el.type == 15 && el.node_indices.size() == 1) {
                GaussPointData gp;
                gp.id = gp_id_counter++;
                gp.X_ref = raw_nodes[el.node_indices[0]].X_ref;
                gp.weight = 1.0;
                gp.entity_tag = el.entity_tag;
                point_gps.push_back(gp);

                // MATLAB sets obj.Nodes from face elements only. Do not insert
                // point-only nodes into active_nodes.
            }

            // --- 1D LINES (Gmsh Type 1) ---
            else if (el.type == 1 && el.node_indices.size() == 2) {
                Eigen::Vector2d p1 = raw_nodes[el.node_indices[0]].X_ref;
                Eigen::Vector2d p2 = raw_nodes[el.node_indices[1]].X_ref;

                Eigen::Vector2d t = p2 - p1;
                const double L = t.norm();
                if (!(L > 0.0)) continue;

                t /= L;
                Eigen::Vector2d n(-t.y(), t.x());
                const double detJ = L / 2.0;

                for (size_t i = 0; i < l_pts.size(); ++i) {
                    const double xi = l_pts[i];

                    GaussPointData gp;
                    gp.id = gp_id_counter++;
                    gp.X_ref = ((1.0 - xi) / 2.0) * p1
                             + ((1.0 + xi) / 2.0) * p2;
                    gp.weight = detJ * l_wts[i];
                    gp.tangent = t;
                    gp.normal = n;
                    gp.entity_tag = el.entity_tag;
                    edge_gps.push_back(gp);
                }

                // MATLAB uses the face-node cloud for shape functions on edges.
                // Do not insert line-only nodes into active_nodes.
            }

            // --- 2D TRIANGLES (Gmsh Type 2) ---
            else if (el.type == 2 && el.node_indices.size() == 3) {
                Eigen::Vector2d p1 = raw_nodes[el.node_indices[0]].X_ref;
                Eigen::Vector2d p2 = raw_nodes[el.node_indices[1]].X_ref;
                Eigen::Vector2d p3 = raw_nodes[el.node_indices[2]].X_ref;

                const double area = 0.5 * std::abs(
                    p1.x() * (p2.y() - p3.y())
                  + p2.x() * (p3.y() - p1.y())
                  + p3.x() * (p1.y() - p2.y())
                );

                for (const auto& pt : t_pts) {
                    GaussPointData gp;
                    gp.id = gp_id_counter++;
                    gp.weight = area * pt.weight;
                    gp.X_ref = pt.u * p1 + pt.v * p2 + pt.w * p3;
                    gp.entity_tag = el.entity_tag;
                    face_gps.push_back(gp);
                }

                for (int idx : el.node_indices) active_node_indices.insert(idx);
            }

            // --- 2D QUADS (Gmsh Type 3) ---
            else if (el.type == 3 && el.node_indices.size() == 4) {
                Eigen::Vector2d p[4];

                for (int i = 0; i < 4; ++i) {
                    p[i] = raw_nodes[el.node_indices[i]].X_ref;
                    active_node_indices.insert(el.node_indices[i]);
                }

                for (size_t ix = 0; ix < q_pts.size(); ++ix) {
                    const double xi = q_pts[ix];
                    const double wx = q_wts[ix];

                    for (size_t iy = 0; iy < q_pts.size(); ++iy) {
                        const double eta = q_pts[iy];
                        const double wy = q_wts[iy];

                        const double N1 = 0.25 * (1.0 - xi) * (1.0 - eta);
                        const double N2 = 0.25 * (1.0 + xi) * (1.0 - eta);
                        const double N3 = 0.25 * (1.0 + xi) * (1.0 + eta);
                        const double N4 = 0.25 * (1.0 - xi) * (1.0 + eta);

                        const double J11 =
                            -0.25 * (1.0 - eta) * p[0].x()
                            +0.25 * (1.0 - eta) * p[1].x()
                            +0.25 * (1.0 + eta) * p[2].x()
                            -0.25 * (1.0 + eta) * p[3].x();

                        const double J12 =
                            -0.25 * (1.0 - eta) * p[0].y()
                            +0.25 * (1.0 - eta) * p[1].y()
                            +0.25 * (1.0 + eta) * p[2].y()
                            -0.25 * (1.0 + eta) * p[3].y();

                        const double J21 =
                            -0.25 * (1.0 - xi) * p[0].x()
                            -0.25 * (1.0 + xi) * p[1].x()
                            +0.25 * (1.0 + xi) * p[2].x()
                            +0.25 * (1.0 - xi) * p[3].x();

                        const double J22 =
                            -0.25 * (1.0 - xi) * p[0].y()
                            -0.25 * (1.0 + xi) * p[1].y()
                            +0.25 * (1.0 + xi) * p[2].y()
                            +0.25 * (1.0 - xi) * p[3].y();

                        // Take |detJ|, as the triangle branch already does for its
                        // area. A quad whose nodes are listed clockwise gives a
                        // negative Jacobian and would otherwise contribute negative
                        // integration weights: the facet stiffness would enter with
                        // the opposite sign to the hinge and boundary terms, and the
                        // strain energy would come out negative. Circle.msh has all
                        // 5760 of its quad Gauss points in that state.
                        const double detJ = std::abs(J11 * J22 - J12 * J21);

                        GaussPointData gp;
                        gp.id = gp_id_counter++;
                        gp.X_ref = N1 * p[0] + N2 * p[1] + N3 * p[2] + N4 * p[3];
                        gp.weight = detJ * wx * wy;
                        gp.entity_tag = el.entity_tag;
                        face_gps.push_back(gp);
                    }
                }
            }
        }

        int new_id = 0;
        for (int raw_idx : active_node_indices) {
            Node n = raw_nodes[raw_idx];
            n.id = new_id++;
            active_nodes.push_back(n);
        }

        std::cout << "Quadrature: " << face_gps.size() << " Face, "
                  << edge_gps.size() << " Edge, "
                  << point_gps.size() << " Point GPs spawned."
                  << " (face_degree = " << face_degree
                  << ", line_degree = " << line_degree << ")"
                  << std::endl;
    }
}

