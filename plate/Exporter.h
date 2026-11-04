#pragma once

// The build stamp (CURVAMI_GIT_COMMIT), generated at every build by
// cmake/BuildStamp.cmake.
#if __has_include("curvami_build_stamp.h")
#include "curvami_build_stamp.h"
#endif
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <H5Cpp.h>
#include <fstream>
#include <set>
#include "Meshless.h"
#include "Kinematics.h"

namespace Exporter {

    // Counts steps that failed to reach disk, so a run cannot finish looking
    // successful while silently having lost output.
    inline int failed_writes = 0;

    // Appends the current state to a compressed HDF5 file.
    // Mandatory columns: [X, Y, u, v, w, rotX_deg, rotY_deg]
    // Optional columns are appended dynamically based on 'fields'.
    inline bool writeBinaryStep(Meshless& plate, const Eigen::VectorXd& x, int step,
                                const std::vector<double>& angles,
                                const std::vector<int>& is_driver, // <--- NEW ARGUMENT
                                double disp, double W_ext, double E_mem, double E_ben, double E_hin,
                                const std::vector<std::string>& fields, const std::string& filename = "results_history.h5") {
        using namespace H5;
        
        plate.updateKinematics(x);
        int num_gps = static_cast<int>(plate.faces.gps.size());
        
        // Base columns are X, Y, u, v, w, rotX, rotY (7 columns)
        int num_cols = 7 + fields.size();

        // 1. Pre-allocate a flat buffer for the HDF5 writer
        std::vector<double> step_data(num_gps * num_cols);
        int idx = 0;
        
        // Conversion factor from radians to degrees
        const double rad2deg = 180.0 / M_PI;

        // 2. Compute the exact kinematics at the Gauss Points
        for(const auto& gp : plate.faces.gps) {
            double u = 0, v = 0, w = 0;
            for(size_t j = 0; j < gp.neighbor_node_ids.size(); ++j) {
                int id = gp.neighbor_node_ids[j];
                u += plate.nodes[id].u.x() * gp.Psi[j];
                v += plate.nodes[id].u.y() * gp.Psi[j];
                w += plate.nodes[id].u.z() * gp.Psi[j];
            }
            
            auto kin = Kinematics::compute(gp, plate.nodes);
            
            // --- MANDATORY COLUMNS (1 to 7) ---
            step_data[idx++] = gp.X_ref.x();
            step_data[idx++] = gp.X_ref.y();
            step_data[idx++] = u;
            step_data[idx++] = v;
            step_data[idx++] = w;
            step_data[idx++] = std::atan2(-kin.b.y(), kin.b.z()) * rad2deg; // rotX (Degrees)
            step_data[idx++] = std::atan2(kin.b.x(), kin.b.z()) * rad2deg;  // rotY (Degrees)
            
            // --- OPTIONAL COLUMNS (8+) ---
            for (const auto& field : fields) {
                // Membrane Strains (E0)
                if      (field == "E_XX") step_data[idx++] = kin.E0(0, 0);
                else if (field == "E_YY") step_data[idx++] = kin.E0(1, 1);
                else if (field == "E_XY") step_data[idx++] = kin.E0(0, 1);
                
                // Bending Curvatures (E1)
                else if (field == "K_XX") step_data[idx++] = kin.E1(0, 0);
                else if (field == "K_YY") step_data[idx++] = kin.E1(1, 1);
                else if (field == "K_XY") step_data[idx++] = kin.E1(0, 1);
                
                // An unrecognised name used to write a silent column of zeros, which
                // is indistinguishable from a genuinely zero field.
                else {
                    static std::set<std::string> warned;
                    if (warned.insert(field).second) {
                        std::cerr << "!!! WARNING: unknown output field \"" << field
                                  << "\" - writing zeros. Valid names: E_XX, E_YY, E_XY,"
                                     " K_XX, K_YY, K_XY." << std::endl;
                    }
                    step_data[idx++] = 0.0;
                }
            }
        }

        // --- GLOBAL PLANARITY OF THE DEFORMED MID-SURFACE -------------------
        // A dihedral angle of 180 deg is a LOCAL measure at the crease. For a
        // CURVED crease the facets cannot close globally -- reflection across a
        // curved line is not a planar isometry -- so a run can report 180 deg at
        // the crease and still show a visible gap between the faces. These
        // scalars measure the surface rather than the crease.
        //
        // Best-fit plane by PCA of the deformed Gauss-point cloud. The smallest
        // eigenvalue of the covariance is the out-of-plane variance, so its
        // square root is the RMS distance from the best-fit plane. For a truly
        // flat-folded state both numbers go to zero; for a curved crease they
        // plateau at the flattening defect.
        double planar_rms = 0.0, planar_max = 0.0, ref_diag = 0.0;
        if (num_gps > 2) {
            std::vector<Eigen::Vector3d> P;
            P.reserve(static_cast<size_t>(num_gps));
            Eigen::Vector3d mean = Eigen::Vector3d::Zero();
            double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
            for (int i = 0; i < num_gps; ++i) {
                const double* r = &step_data[static_cast<size_t>(i) * num_cols];
                P.emplace_back(r[0] + r[2], r[1] + r[3], r[4]);  // (X+u, Y+v, w)
                mean += P.back();
                xmin = std::min(xmin, r[0]); xmax = std::max(xmax, r[0]);
                ymin = std::min(ymin, r[1]); ymax = std::max(ymax, r[1]);
            }
            mean /= static_cast<double>(num_gps);

            Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
            for (const auto& p : P) { Eigen::Vector3d d = p - mean; C += d * d.transpose(); }
            C /= static_cast<double>(num_gps);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
            const Eigen::Vector3d n = es.eigenvectors().col(0);   // smallest eigenvalue
            planar_rms = std::sqrt(std::max(0.0, es.eigenvalues()(0)));
            for (const auto& p : P)
                planar_max = std::max(planar_max, std::abs((p - mean).dot(n)));
            ref_diag = std::hypot(xmax - xmin, ymax - ymin);
        }

        // --- PER-GAUSS-POINT DIHEDRAL ALONG EACH CREASE ---------------------
        // The `angles` attribute is a length-weighted MEAN. It cannot be compared
        // against a pointwise curvature: on CurvedPleat the local dihedral varies
        // ~15 deg along one crease at large folds, and reconstructing it from the
        // exported surface needs facet normals fitted to ~3 Gauss-point rows,
        // which under-rotate where the facet bends sharply (7 deg of drift at
        // theta ~ 150). The solver already has the exact value from the enriched
        // crease kinematics, so export it.
        //
        // Columns: [fold_index, crease_tag, X_ref, Y_ref, s_arc, theta_deg, weight]
        // s_arc is measured along the crease in the FLAT pattern: the Gauss points
        // are chained nearest-neighbour from one end (they carry no ordering of
        // their own) and s accumulates the quadrature weights, which partition the
        // crease length exactly.
        const int crease_cols = 7;
        std::vector<double> crease_data;
        for (size_t fi = 0; fi < plate.folds.size(); ++fi) {
            const auto& fold = plate.folds[fi];
            const size_t n = fold.left.gps.size();
            if (n == 0 || fold.right.gps.size() != n) continue;

            std::vector<double> theta_deg(n);
            for (size_t i = 0; i < n; ++i) {
                const auto& gL = fold.left.gps[i];
                const auto& gR = fold.right.gps[i];
                auto kinL = Kinematics::compute(gL, plate.nodes);
                auto kinR = Kinematics::compute(gR, plate.nodes);
                Eigen::Matrix<double, 12, 1> F;
                F.segment<3>(0) = kinL.F0.col(0); F.segment<3>(3) = kinL.F0.col(1);
                F.segment<3>(6) = kinR.F0.col(0); F.segment<3>(9) = kinR.F0.col(1);
                // Same function the hinge residual uses, so the exported angle is
                // the one the mechanics acts on, not a re-derivation of it.
                theta_deg[i] = Constraints::foldAngleAndGrad(
                                   F, gL.tangent.x(), gL.tangent.y(), nullptr) * rad2deg;
            }

            // Chain the Gauss points along the crease, starting from an endpoint
            // (one of the two mutually farthest points).
            size_t start = 0;
            double best = -1.0;
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j) {
                    double d = (fold.left.gps[i].X_ref - fold.left.gps[j].X_ref).squaredNorm();
                    if (d > best) { best = d; start = i; }
                }
            std::vector<char> used(n, 0);
            std::vector<size_t> order;
            order.reserve(n);
            size_t cur = start;
            used[cur] = 1; order.push_back(cur);
            for (size_t k = 1; k < n; ++k) {
                double dmin = 1e300; size_t next = n;
                for (size_t j = 0; j < n; ++j) {
                    if (used[j]) continue;
                    double d = (fold.left.gps[cur].X_ref - fold.left.gps[j].X_ref).squaredNorm();
                    if (d < dmin) { dmin = d; next = j; }
                }
                if (next == n) break;
                used[next] = 1; order.push_back(next); cur = next;
            }

            double s_acc = 0.0;
            for (size_t k = 0; k < order.size(); ++k) {
                const size_t i = order[k];
                const auto& gL = fold.left.gps[i];
                const double w = gL.weight;
                crease_data.push_back(static_cast<double>(fi));
                crease_data.push_back(static_cast<double>(fold.crease_tag));
                crease_data.push_back(gL.X_ref.x());
                crease_data.push_back(gL.X_ref.y());
                crease_data.push_back(s_acc + 0.5 * w);   // midpoint of this weight
                crease_data.push_back(theta_deg[i]);
                crease_data.push_back(w);
                s_acc += w;
            }
        }

        // 3. Write to the HDF5 File
        try {
            H5File file;
            Exception::dontPrint();

            // Only create-and-truncate when the file genuinely does not exist yet.
            // The old code truncated on ANY open failure, so a transient problem
            // (lock, permissions, network filesystem) destroyed every step written
            // so far. If it exists but will not open, let the exception propagate.
            bool exists = false;
            { std::ifstream probe(filename, std::ios::binary); exists = probe.good(); }

            file = exists ? H5File(filename, H5F_ACC_RDWR)
                          : H5File(filename, H5F_ACC_TRUNC);

            hsize_t dims[2] = { static_cast<hsize_t>(num_gps), static_cast<hsize_t>(num_cols) };
            DataSpace dataspace(2, dims);

            DSetCreatPropList plist;
            hsize_t chunk_dims[2] = { std::min(dims[0], static_cast<hsize_t>(1000)), static_cast<hsize_t>(num_cols) };
            plist.setChunk(2, chunk_dims);
            plist.setDeflate(6);

            char dset_name[64];
            snprintf(dset_name, sizeof(dset_name), "/step_%05d", step);

            DataSet dataset = file.createDataSet(dset_name, PredType::NATIVE_DOUBLE, dataspace, plist);
            dataset.write(step_data.data(), PredType::NATIVE_DOUBLE);

            // Per-crease dihedral, as its own dataset next to the step. Additive:
            // nothing above changes, and readers that don't know about it see the
            // same step_NNNNN dataset and attributes as before.
            if (!crease_data.empty()) {
                hsize_t cdims[2] = { crease_data.size() / crease_cols,
                                     static_cast<hsize_t>(crease_cols) };
                DataSpace cspace(2, cdims);
                DSetCreatPropList cplist;
                hsize_t cchunk[2] = { std::min(cdims[0], static_cast<hsize_t>(1000)),
                                      static_cast<hsize_t>(crease_cols) };
                cplist.setChunk(2, cchunk);
                cplist.setDeflate(6);

                char cname[80];
                snprintf(cname, sizeof(cname), "/step_%05d_creases", step);
                DataSet cds = file.createDataSet(cname, PredType::NATIVE_DOUBLE, cspace, cplist);
                cds.write(crease_data.data(), PredType::NATIVE_DOUBLE);

                const char* cols = "fold_index,crease_tag,X_ref,Y_ref,s_arc,theta_deg,weight";
                StrType stype(PredType::C_S1, strlen(cols) + 1);
                DataSpace cattr(H5S_SCALAR);
                Attribute acols = cds.createAttribute("columns", stype, cattr);
                acols.write(stype, cols);
            }

            // Provenance. A stale .h5 from a pre-fix binary once produced a
            // convincing false diagnosis (the straight control "failure"), and a
            // skip-if-exists run loop makes that easy to repeat silently. Stamp
            // every step with the commit the solver was built from; the analysis
            // scripts compare it against the current build and complain.
#ifndef CURVAMI_GIT_COMMIT
#define CURVAMI_GIT_COMMIT "unknown"
#endif
            {
                const char* stamp = CURVAMI_GIT_COMMIT;
                StrType stamp_type(PredType::C_S1, strlen(stamp) + 1);
                DataSpace stamp_space(H5S_SCALAR);
                Attribute a_stamp = dataset.createAttribute("git_commit", stamp_type, stamp_space);
                a_stamp.write(stamp_type, stamp);
            }

            // =========================================================
            // WRITE SCALAR & ARRAY ATTRIBUTES (Metadata)
            // =========================================================
            
            // 1. Write the vector of fold angles as an Array Attribute
            std::vector<double> safe_angles = angles;
            if (safe_angles.empty()) safe_angles.push_back(0.0); // Safeguard for HDF5 size constraint
            
            hsize_t num_angles = safe_angles.size();
            ArrayType array_type(PredType::NATIVE_DOUBLE, 1, &num_angles);
            DataSpace angle_space(H5S_SCALAR);
            Attribute attr_angles = dataset.createAttribute("angles", array_type, angle_space);
            attr_angles.write(array_type, safe_angles.data());

            // 2. Write the vector of Active/Passive flags (NEW)
            std::vector<int> safe_driver = is_driver;
            if (safe_driver.empty()) safe_driver.push_back(0);
            hsize_t num_drivers = safe_driver.size();
            ArrayType driver_type(PredType::NATIVE_INT, 1, &num_drivers);
            Attribute attr_driver = dataset.createAttribute("is_driver", driver_type, angle_space);
            attr_driver.write(driver_type, safe_driver.data());

            // 3. Write the rest of the standard scalar attributes
            DataSpace attr_space(H5S_SCALAR);

            Attribute attr_disp = dataset.createAttribute("disp", PredType::NATIVE_DOUBLE, attr_space);
            attr_disp.write(PredType::NATIVE_DOUBLE, &disp);

            Attribute attr_wext = dataset.createAttribute("W_ext", PredType::NATIVE_DOUBLE, attr_space);
            attr_wext.write(PredType::NATIVE_DOUBLE, &W_ext);

            Attribute attr_emem = dataset.createAttribute("E_mem", PredType::NATIVE_DOUBLE, attr_space);
            attr_emem.write(PredType::NATIVE_DOUBLE, &E_mem);

            Attribute attr_eben = dataset.createAttribute("E_ben", PredType::NATIVE_DOUBLE, attr_space);
            attr_eben.write(PredType::NATIVE_DOUBLE, &E_ben);

            Attribute attr_ehin = dataset.createAttribute("E_hin", PredType::NATIVE_DOUBLE, attr_space);
            attr_ehin.write(PredType::NATIVE_DOUBLE, &E_hin);

            // 4. Global flatness of the deformed surface (see computation above).
            //    planar_max / ref_diag is the dimensionless flattening defect.
            Attribute attr_prms = dataset.createAttribute("planar_rms", PredType::NATIVE_DOUBLE, attr_space);
            attr_prms.write(PredType::NATIVE_DOUBLE, &planar_rms);

            Attribute attr_pmax = dataset.createAttribute("planar_max", PredType::NATIVE_DOUBLE, attr_space);
            attr_pmax.write(PredType::NATIVE_DOUBLE, &planar_max);

            Attribute attr_rdiag = dataset.createAttribute("ref_diag", PredType::NATIVE_DOUBLE, attr_space);
            attr_rdiag.write(PredType::NATIVE_DOUBLE, &ref_diag);

        } catch(const Exception& error) {
            std::cerr << "!!! HDF5 write FAILED at step " << step << " -> " << filename
                      << "\n!!! This step was NOT saved.\n";
            error.printErrorStack();
            ++failed_writes;
            return false;
        }
        return true;
    }
}
