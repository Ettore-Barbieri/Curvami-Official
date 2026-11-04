#include <iostream>
#include <chrono>
#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <random>
#include <exception>

#include <nlohmann/json.hpp>

#include "Meshless.h"
#include "NonLinearSolver.h"
#include "Exporter.h"
#include "SetupHelper.h"
#include "ProgressiveCreasing.h"
#include "StressSampler.h"
#include "JIntegral.h"
#include "Reactions.h"
#include "curvami/MeshUtils.h"

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

static int run_solver(int argc, char* argv[]) {
    cout << "=======================================================" << endl;
    cout << "   Curvami C++: Generalized Simulation Engine          " << endl;
    cout << "=======================================================" << endl;

    if (argc < 2) {
        cerr << "!!! ERROR: No configuration file provided." << endl;
        cerr << "Usage: ./solver <config_file.json>" << endl;
        return 1;
    }

    string config_path = argv[1];
    ifstream f(config_path);
    if (!f.is_open()) {
        cerr << "!!! ERROR: Could not open config file: " << config_path << endl;
        return 1;
    }
    json config = json::parse(f);
    cout << "Loaded Configuration: " << config_path << "\n" << endl;

    auto total_start = high_resolution_clock::now();

    // =========================================================================
    // 1. Initialization (From JSON)
    // =========================================================================
    Meshless plate(3);
    double E  = config["material"]["E"];
    double nu = config["material"]["nu"];
    double h  = config["material"]["h"];
    
    Ply p1; p1.E1 = E; p1.E2 = E; p1.nu12 = nu; p1.G12 = E / (2.0 * (1.0 + nu));
    p1.thickness = h; p1.angle_deg = 0.0;
    plate.material.plies.push_back(p1);
    Constitutive::computeABD(plate.material);

    // =========================================================================
    // 2. Load Mesh & Quadrature
    // =========================================================================
    auto start_mesh = high_resolution_clock::now();
    vector<GmshReader::Element> elements;
    std::vector<Node> raw_nodes;
    std::map<int, int> edge_tag_to_idx;

    string mesh_file = config["mesh_file"];
    
    SetupHelper::loadMeshAndQuadrature(mesh_file, plate, elements, edge_tag_to_idx, -1, &raw_nodes);
    
    auto end_mesh = high_resolution_clock::now();
    cout << "Mesh Setup Time: " << duration_cast<milliseconds>(end_mesh - start_mesh).count() << " ms" << endl;

    plate.rho = MeshUtils::computeSupportRadius(plate.nodes, 3);

    // Natural curvature (plate/Kinematics.h): a band along Y that is stress-free when bent.
    //   "natural_curvature": {"type": "band", "x0": X, "width": w, "angle_deg": theta0,
    //                         "sense": "mountain" | "valley"}
    Kinematics::naturalCurvature() = Kinematics::NaturalCurvature();
    if (config.contains("natural_curvature")) {
        const json& nc = config["natural_curvature"];
        if (nc.value("type", std::string("band")) != "band") throw std::runtime_error("natural_curvature.type must be band");
        Kinematics::NaturalCurvature& f = Kinematics::naturalCurvature();
        f.active = true;
        f.x0 = nc.at("x0").get<double>();
        f.width = nc.at("width").get<double>();
        f.theta0 = nc.at("angle_deg").get<double>() * M_PI / 180.0;
        const std::string sense = nc.value("sense", std::string("mountain"));
        if (sense != "mountain" && sense != "valley") throw std::runtime_error("natural_curvature.sense must be mountain or valley");
        f.sign = (sense == "mountain") ? -1.0 : 1.0;
        if (!(f.width > 0.0)) throw std::runtime_error("natural_curvature.width must be > 0");
        cout << " >> Natural curvature: band at X = " << f.x0 << ", width " << f.width << ", angle "
             << nc.at("angle_deg").get<double>() << " deg, " << sense << endl;
    }

    // =========================================================================
    // 3. Define Folds & Visibility (Decoupled Mechanics and Topology)
    // =========================================================================
    bool is_rigid = false;
    std::vector<Enrichment::FoldLine> all_creases;

    if (config.contains("folds")) {
        int fold_index = 0;
        for (const auto& fold_cfg : config["folds"]) {
            int crease_tag = fold_cfg["crease_tag"];
            double stiffness = fold_cfg.value("stiffness", 0.0);
            bool rigid_fold = fold_cfg["is_rigid"];
            bool is_driver = fold_cfg.value("is_driver", false);
            
            double fold_sign = 1.0;
            if (fold_cfg.contains("type")) {
                std::string type_str = fold_cfg["type"];
                if (type_str == "mountain" || type_str == "Mountain") {
                    fold_sign = -1.0;
                }
            }
            
            double angle_start_rad = fold_cfg.value("angle_start", 0.0) * M_PI / 180.0;
            double angle_target_rad = fold_cfg.value("angle_target", 0.0) * M_PI / 180.0;
            
            double initial_angle_rad = angle_start_rad * fold_sign;
            
            if (rigid_fold) is_rigid = true;

            if (edge_tag_to_idx.count(crease_tag)) {
                plate.addFold(edge_tag_to_idx[crease_tag], stiffness, initial_angle_rad, rigid_fold, fold_sign);
                plate.folds[fold_index].is_driver = is_driver;
                plate.folds[fold_index].crease_tag = crease_tag;
                plate.folds[fold_index].angle_start = angle_start_rad;
                plate.folds[fold_index].angle_target = angle_target_rad;

                // Tie of the displacement across the fold (plate/PlateData.h, FoldTie):
                //   "tie": {"type": "pointwise"} (default) | {"type": "subset", "every": k}
                //        | {"type": "mortar_p0", "segment": length} | {"type": "mortar_p1", "spacing": length}
                if (fold_cfg.contains("tie")) {
                    FoldTie& tie = plate.folds[fold_index].tie;
                    const std::string tt = fold_cfg["tie"].value("type", std::string("pointwise"));
                    if (tt == "pointwise") tie.type = FoldTie::Pointwise;
                    else if (tt == "subset") {
                        tie.type = FoldTie::Subset;
                        tie.every = fold_cfg["tie"].at("every").get<int>();
                        if (tie.every < 1) throw std::runtime_error("folds.tie.every must be >= 1");
                    }
                    else if (tt == "mortar_p0" || tt == "mortar_p1") {
                        tie.type = (tt == "mortar_p0") ? FoldTie::MortarP0 : FoldTie::MortarP1;
                        tie.length = fold_cfg["tie"].at(tt == "mortar_p0" ? "segment" : "spacing").get<double>();
                        if (!(tie.length > 0.0)) throw std::runtime_error("folds.tie: the segment / spacing must be > 0");
                    }
                    else throw std::runtime_error("folds.tie.type: unknown type \"" + tt + "\"");
                }
                
                if (fold_cfg.contains("force_left")) {
                    auto fl = fold_cfg["force_left"];
                    plate.folds[fold_index].left.bc.force = Eigen::Vector3d(fl[0], fl[1], fl[2]);
                }
                if (fold_cfg.contains("force_right")) {
                    auto fr = fold_cfg["force_right"];
                    plate.folds[fold_index].right.bc.force = Eigen::Vector3d(fr[0], fr[1], fr[2]);
                }
                fold_index++;
            } else {
                cerr << "!!! WARNING: Crease tag " << crease_tag << " not found in edges!" << endl;
            }
        }
    }

    if (config.contains("cuts")) {
        for (const auto& cut_cfg : config["cuts"]) {
            std::vector<int> tags = cut_cfg.get<std::vector<int>>();
            auto crease = MeshUtils::extractCurve(elements, raw_nodes, tags);
            all_creases.push_back(crease);
        }
    } else {
        if (config.contains("folds")) {
            for (const auto& fold_cfg : config["folds"]) {
                std::vector<int> tags = { fold_cfg["crease_tag"] };
                auto crease = MeshUtils::extractCurve(elements, raw_nodes, tags);
                all_creases.push_back(crease);
            }
        }
    }

    // =========================================================================
    // 4. Compute Global RKPM Shape Functions & Init CSR Assembly
    // =========================================================================
    auto start_rkpm = high_resolution_clock::now();
    SetupHelper::computeAllShapeFunctions(plate, all_creases);
    auto end_rkpm = high_resolution_clock::now();
    for (const auto& f : plate.folds) {
        if (f.tie.type == FoldTie::Pointwise) continue;
        cout << " >> Fold " << f.crease_tag << ": tie with " << Constraints::tieRowCount(f.left.gps, f.tie, &f.gp_active)
             << " rows of 3 constraints for " << f.left.gps.size() << " crease points" << endl;
    }
    cout << "RKPM Setup Time: " << duration_cast<milliseconds>(end_rkpm - start_rkpm).count() << " ms" << endl;

    plate.fold_exact_tangent = config["solver"].value("fold_exact_tangent", false);
    if (plate.fold_exact_tangent) {
        cout << ">>> Exact hinge tangent enabled (includes the dihedral-angle Hessian term)." << endl;
    }

    plate.initAssembly();

    // =========================================================================
    // 5. Arbitrary Boundary Conditions & Loads
    // =========================================================================
    if (config.contains("point_bcs")) {
        for (const auto& bc_cfg : config["point_bcs"]) {
            int target_tag = bc_cfg["tag"];
            for (auto& pt_dom : plate.points) {
                if (pt_dom.gps.empty()) continue;
                if (pt_dom.gps[0].entity_tag == target_tag) {
                    if (bc_cfg.contains("u")) { pt_dom.bc.active_u = true; pt_dom.bc.val_u = bc_cfg["u"]; }
                    if (bc_cfg.contains("v")) { pt_dom.bc.active_v = true; pt_dom.bc.val_v = bc_cfg["v"]; }
                    if (bc_cfg.contains("w")) { pt_dom.bc.active_w = true; pt_dom.bc.val_w = bc_cfg["w"]; }
                    if (bc_cfg.contains("force")) {
                        auto force_arr = bc_cfg["force"];
                        pt_dom.bc.force = Eigen::Vector3d(force_arr[0], force_arr[1], force_arr[2]);
                    }
                }
            }
        }
    }

    if (config.contains("edge_bcs")) {
            for (const auto& bc_cfg : config["edge_bcs"]) {
                int target_tag = bc_cfg["tag"];
                for (auto& edge_dom : plate.edges) {
                    if (edge_dom.gps.empty()) continue;
                    if (edge_dom.gps[0].entity_tag == target_tag) {
                        if (bc_cfg.contains("u")) { edge_dom.bc.active_u = true; edge_dom.bc.val_u = bc_cfg["u"]; }
                        if (bc_cfg.contains("v")) { edge_dom.bc.active_v = true; edge_dom.bc.val_v = bc_cfg["v"]; }
                        if (bc_cfg.contains("w")) { edge_dom.bc.active_w = true; edge_dom.bc.val_w = bc_cfg["w"]; }
                        if (bc_cfg.contains("rot")) { edge_dom.bc.active_rot = true; edge_dom.bc.val_rot = bc_cfg["rot"]; }
                        if (bc_cfg.contains("force")) {
                            auto force_arr = bc_cfg["force"];
                            edge_dom.bc.force = Eigen::Vector3d(force_arr[0], force_arr[1], force_arr[2]);
                        }
                        // bending moment per unit length, positive on a positive dw/dn (outward n)
                        if (bc_cfg.contains("moment")) edge_dom.bc.moment = bc_cfg["moment"];
                    }
                }
            }
        }
    
    std::vector<std::string> export_fields = {};
    if (config.contains("outputs")) {
        export_fields = config["outputs"].get<std::vector<std::string>>();
    }

    // =========================================================================
    // 6. SOLVER DISPATCHER
    // =========================================================================
    NonLinearSolver solver;
    
    solver.kmax  = config["solver"].value("kmax", 500);
    solver.ls_max_iters = config["solver"].value("ls_max_iters", 500);
    // Newton convergence tests (absolute; defaults as in ContinuationSolver): max
    // projected residual and |dx . R|. Small-angle runs scale them with the load.
    solver.tol_force  = config["solver"].value("tol_force", solver.tol_force);
    solver.tol_energy = config["solver"].value("tol_energy", solver.tol_energy);
    {
        // Default "accept", the original behaviour, for every analysis. Progressive
        // creasing too: with "fail" it stopped at step 1 of example C (29 Sep); its
        // safety net is the guard, which triggers the fallback instead.
        const std::string on_fail = config["solver"].value("ls_on_fail", std::string("accept"));
        if (on_fail != "accept" && on_fail != "fail") throw std::runtime_error("solver.ls_on_fail must be accept or fail");
        solver.ls_fail_on_no_descent = (on_fail == "fail");
        cout << ">>> LineSearch on no descent: " << on_fail << endl;
    }
    solver.psi   = config["solver"].value("psi", 1.0);
    solver.ds    = config["solver"].value("ds", 0.1);
    solver.dsmax = config["solver"].value("dsmax", 1e3 * solver.ds);
    solver.dsmin = config["solver"].value("dsmin", 1e-3 * solver.ds);
    solver.max_iters = config["solver"].value("max_iters", 30);
    solver.snapback_inertia_veto = config["solver"].value("snapback_inertia_veto", false);
    solver.asla_hold_after_reversal = config["solver"].value("asla_hold_after_reversal", false);
    if (solver.asla_hold_after_reversal) {
        cout << ">>> ASLA hold-after-reversal enabled: ds is not boosted within two steps"
             << " of a direction reversal." << endl;
    }
    if (solver.snapback_inertia_veto) {
        cout << ">>> Snap-back inertia veto enabled: direction reversals must be confirmed"
             << " by a change in the reduced-Hessian inertia." << endl;
    }
    solver.no_progress_abort      = config["solver"].value("no_progress_abort", true);
    solver.no_progress_window     = config["solver"].value("no_progress_window", 24);
    solver.no_progress_efficiency = config["solver"].value("no_progress_efficiency", 0.15);
    if (!solver.no_progress_abort) {
        cout << ">>> No-progress abort DISABLED: a run pinned at dsmin will continue"
             << " accepting steps until kmax." << endl;
    }
    
    string out_path = config["output_file"];

    size_t lastdot = out_path.find_last_of(".");
    string res_path = (lastdot != string::npos) ? out_path.substr(0, lastdot) + ".res" : out_path + ".res";
    solver.restart_file = res_path;

    bool is_restart = config.value("restart", false);
    if (is_restart) {
        cout << "\n--- RESTART MODE ACTIVATED: Appending to " << out_path << " ---" << endl;
    } else {
        std::remove(out_path.c_str());
        std::remove(res_path.c_str());
    }

    // Energy release rate of a crack, as post-processing at exported steps
    // (plate/JIntegral.h). One object, or a list of them for several tips:
    //   "j_integral": {"tip": [x, y], "direction": [dx, dy], "rings": [[rho1, rho2], ...],
    //                  "q": "cubic" | "linear", "steps": [..] (optional: default every step),
    //                  "fold_band": half-width (optional: band form of the fold's share),
    //                  "fold_band_plateau": half-width of its flat part (optional, default 0)}
    // Results: <output>_G.csv (or _G0.csv, _G1.csv, ...) and /step_NNNNN_G (G0, G1, ...) in the .h5.
    std::vector<JIntegral::Settings> jsets;
    if (config.contains("j_integral")) {
        std::vector<json> jl;
        if (config["j_integral"].is_array()) for (const auto& j : config["j_integral"]) jl.push_back(j);
        else jl.push_back(config["j_integral"]);
        const std::string stem = (lastdot != string::npos) ? out_path.substr(0, lastdot) : out_path;
        for (size_t i = 0; i < jl.size(); ++i) {
            JIntegral::Settings js;
            js.tip = Eigen::Vector2d(jl[i].at("tip")[0].get<double>(), jl[i].at("tip")[1].get<double>());
            js.dir = Eigen::Vector2d(jl[i].at("direction")[0].get<double>(), jl[i].at("direction")[1].get<double>()).normalized();
            for (const auto& r : jl[i].at("rings")) js.rings.push_back({r[0].get<double>(), r[1].get<double>()});
            const std::string q = jl[i].value("q", std::string("cubic"));
            if (q != "cubic" && q != "linear") throw std::runtime_error("j_integral.q must be cubic or linear");
            js.linear_q = (q == "linear");
            js.fold_band = jl[i].value("fold_band", 0.0);
            js.fold_band_plateau = jl[i].value("fold_band_plateau", 0.0);
            if (js.fold_band_plateau < 0.0 || (js.fold_band > 0.0 && js.fold_band_plateau >= js.fold_band))
                throw std::runtime_error("j_integral.fold_band_plateau must be >= 0 and smaller than fold_band");
            if (jl[i].contains("steps")) js.steps = jl[i]["steps"].get<std::vector<int>>();
            js.tag = (jl.size() == 1) ? "G" : "G" + std::to_string(i);
            js.csv = stem + "_" + js.tag + ".csv";
            jsets.push_back(js);
        }
    }
    // Stress along rays from a crack tip, as post-processing at exported steps
    // (plate/StressSampler.h). One object, or a list of them for several tips:
    //   "stress_sampler": {"tip": [x, y], "direction": [dx, dy], "r_max": ..., "dr": ...,
    //                      "angles_deg": [..] (optional), "steps": [..] (optional)}
    // Results: one CSV per ray, <output>_S_s<step>_p<phi>.csv (S0, S1, ... for a list).
    std::vector<StressSampler::Settings> ssets;
    if (config.contains("stress_sampler")) {
        std::vector<json> sl;
        if (config["stress_sampler"].is_array()) for (const auto& j : config["stress_sampler"]) sl.push_back(j);
        else sl.push_back(config["stress_sampler"]);
        const std::string stem = (lastdot != string::npos) ? out_path.substr(0, lastdot) : out_path;
        for (size_t i = 0; i < sl.size(); ++i) {
            StressSampler::Settings ss;
            ss.tip = Eigen::Vector2d(sl[i].at("tip")[0].get<double>(), sl[i].at("tip")[1].get<double>());
            ss.dir = Eigen::Vector2d(sl[i].at("direction")[0].get<double>(), sl[i].at("direction")[1].get<double>()).normalized();
            ss.r_max = sl[i].at("r_max").get<double>();
            ss.dr = sl[i].at("dr").get<double>();
            if (sl[i].contains("angles_deg")) ss.angles_deg = sl[i]["angles_deg"].get<std::vector<double>>();
            if (sl[i].contains("steps")) ss.steps = sl[i]["steps"].get<std::vector<int>>();
            ss.stem = stem + "_" + ((sl.size() == 1) ? std::string("S") : "S" + std::to_string(i));
            ssets.push_back(ss);
        }
    }
    // Every analysis exports a step through this: the usual .h5 step, then G.
    auto exportStep = [&](Meshless& pl, const Eigen::VectorXd& xx, int step, const std::vector<double>& angles,
                          const std::vector<int>& drivers, double disp, double W_ext, double E_mem, double E_ben,
                          double E_hin, const std::vector<std::string>& fields, const std::string& path) {
        const bool ok = Exporter::writeBinaryStep(pl, xx, step, angles, drivers, disp, W_ext, E_mem, E_ben, E_hin, fields, path);
        // F_ext . x of the unit (lambda = 1) load, so the total potential energy at the
        // step's load factor is Pi = E_mem + E_ben + E_hin - lambda * Fext_dot_x for dead
        // loads in any analysis (ArcLength's W_ext is a path integral, not this).
        if (ok) {
            try {
                H5::Exception::dontPrint();
                H5::H5File file(path, H5F_ACC_RDWR);
                char name[64];
                std::snprintf(name, sizeof(name), "/step_%05d", step);
                H5::DataSet ds = file.openDataSet(name);
                const double fx = pl.computeExternalForce(xx).dot(xx);
                ds.createAttribute("Fext_dot_x", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(H5S_SCALAR))
                  .write(H5::PredType::NATIVE_DOUBLE, &fx);
            } catch (...) {
                printf("!!! WARNING: could not write Fext_dot_x for step %d to %s\n", step, path.c_str());
            }
        }
        for (auto& js : jsets) JIntegral::report(pl, xx, step, js, path, CURVAMI_GIT_COMMIT);
        for (const auto& ss : ssets) StressSampler::report(pl, xx, step, ss, all_creases, CURVAMI_GIT_COMMIT);
        return ok;
    };

    Eigen::VectorXd x0;
    if (!is_restart) {
        x0 = solver.GetInitialConditionWithFolds(plate);
    } else {
        x0 = Eigen::VectorXd::Zero(plate.nodes.size() * 3);
    }

    // Construct static driver arrays for analyses that don't dynamically update
    std::vector<double> static_angles(std::max((size_t)1, plate.folds.size()), 0.0);
    std::vector<int> static_drivers(std::max((size_t)1, plate.folds.size()), 0);
    for(size_t i = 0; i < plate.folds.size(); ++i) {
        static_angles[i] = plate.folds[i].Ang0 * 180.0 / M_PI;
        static_drivers[i] = plate.folds[i].is_driver ? 1 : 0;
    }

    string analysis_type = config.value("analysis_type", "ArcLength");

    if (analysis_type == "Initial") {
        cout << "\n=======================================================" << endl;
        cout << "   DEBUG MODE: Exporting Initial Kinematic State (x0)    " << endl;
        cout << "=======================================================" << endl;
        
        exportStep(plate, x0, 0, static_angles, static_drivers, 0.0, 0.0, 0.0, 0.0, 0.0, export_fields, out_path);
        
        cout << ">> Initial state successfully exported to: " << out_path << endl;
    }
    else if (analysis_type == "LineSearch") {
        cout << "\n--- Starting Static Line Search Analysis ---" << endl;
        double target_load = config.value("target_load", 0.0);
        bool converged = solver.LineSearch(plate, target_load, x0, is_rigid);
        if (converged) {
            double disp = solver.ComputeTrackingDisplacement(plate, x0);
            
            double E_hin = plate.computeTotalHingeEnergy(x0);
            double E_mem = 0.0;
            double E_ben = 0.0;
            plate.computeStrainEnergies(x0, E_mem, E_ben);
            // work of the applied loads (dead loads: forces and edge moments), so the
            // total potential energy Pi = E_mem + E_ben + E_hin - W_ext is available
            const double W_ext = target_load * plate.computeExternalForce(x0).dot(x0);

            exportStep(plate, x0, 1, std::vector<double>{target_load}, static_drivers, disp, W_ext, E_mem, E_ben, E_hin, export_fields, out_path);
            // "report_reactions": true writes <output>_reactions.csv (plate/Reactions.h)
            if (config.value("report_reactions", false))
                Reactions::write(plate, x0, target_load, is_rigid,
                                 ((lastdot != string::npos) ? out_path.substr(0, lastdot) : out_path) + "_reactions.csv",
                                 CURVAMI_GIT_COMMIT);
            
            cout << "Line Search converged successfully!" << endl;
        } else {
            cout << "!!! WARNING: Line Search failed to converge at load = " << target_load << endl;
        }
    }
    else if (analysis_type == "ArcLength") {
        if (!is_restart) {
            cout << "\n--- Solving Equilibrium for Initial Bent State ---" << endl;
            solver.LineSearch(plate, 0.0, x0, is_rigid);
            
            double disp0 = solver.ComputeTrackingDisplacement(plate, x0);
            
            double E_hin_0 = plate.computeTotalHingeEnergy(x0);
            double E_mem_0 = 0.0;
            double E_ben_0 = 0.0;
            plate.computeStrainEnergies(x0, E_mem_0, E_ben_0);
            double W_ext_0 = 0.0;
            
            exportStep(plate, x0, 0, std::vector<double>{0.0}, static_drivers, disp0, W_ext_0, E_mem_0, E_ben_0, E_hin_0, export_fields, out_path);
        }
        
        cout << "\n--- Starting Arc-Length Continuation ---" << endl;
        
        auto on_step_converged = [&](int step, std::vector<double> current_angles, const Eigen::VectorXd& x_curr,
                                     double W_ext, double E_mem, double E_ben, double E_hin) {
            double disp = solver.ComputeTrackingDisplacement(plate, x_curr);
            
            exportStep(plate, x_curr, step, current_angles, static_drivers, disp,
                                      W_ext, E_mem, E_ben, E_hin,
                                      export_fields, out_path);
        };
        
        double max_load = config["solver"]["max_load"];
        solver.ArcLength(plate, 0.0, x0, max_load, is_rigid, on_step_converged);
    }
    else if (analysis_type == "Landscape") {
        cout << "\n=======================================================" << endl;
        cout << "   MAPPING ENERGY LANDSCAPE (HOT-START CONTINUATION)   " << endl;
        cout << "=======================================================" << endl;
        
        is_rigid = true;
        
        int grid_size = config.value("landscape_grid", 20);
        double f0_max = config.value("landscape_f0_max", 20.0) * M_PI / 180.0;
        double f1_max = config.value("landscape_f1_max", 30.0) * M_PI / 180.0;

        std::ofstream csv("landscape.csv");
        csv << "a0,a1,E_tot\n";

        Eigen::VectorXd x = x0;
        
        for (int i = 0; i < grid_size; ++i) {
            double a0 = static_cast<double>(i) / (grid_size - 1) * f0_max;
            
            for (int j = 0; j < grid_size; ++j) {
                int j_actual = (i % 2 == 0) ? j : (grid_size - 1 - j);
                double a1 = static_cast<double>(j_actual) / (grid_size - 1) * f1_max;
                
                plate.folds[0].angle_start = a0;
                plate.folds[0].angle_target = a0;
                plate.folds[0].Ang0 = plate.folds[0].fold_sign * a0;

                plate.folds[1].angle_start = a1;
                plate.folds[1].angle_target = a1;
                plate.folds[1].Ang0 = plate.folds[1].fold_sign * a1;

                bool converged = solver.LineSearch(plate, 0.0, x, is_rigid);
                
                double E_mem = 0.0, E_ben = 0.0, E_tot = std::nan("");
                if (converged) {
                    plate.computeStrainEnergies(x, E_mem, E_ben);
                    E_tot = E_mem + E_ben;
                } else {
                    cout << "!!! Diverged at F0=" << a0*180/M_PI << ", F1=" << a1*180/M_PI << endl;
                }
                
                csv << a0 * 180.0 / M_PI << "," << a1 * 180.0 / M_PI << "," << E_tot << "\n";
                csv.flush();
                printf(" Grid [%2d, %2d] -> F0: %5.1f deg | F1: %5.1f deg | E_tot: %10.5e\n",
                       i, j_actual, a0*180.0/M_PI, a1*180.0/M_PI, E_tot);
            }
        }
        csv.close();
        cout << "\n>>> Landscape successfully saved to 'landscape.csv' <<<" << endl;
    }
    else if (analysis_type == "KinematicDeployment") {
        
        int num_steps = config.value("num_steps", 50);
        printf(" -> Loaded KinematicDeployment with %d steps.\n", num_steps);

        auto on_step_converged = [&](int step, const Eigen::VectorXd& x_curr, const std::vector<double>& current_angles, const std::vector<int>& current_drivers, double E_mem, double E_ben, double E_hin) {
            exportStep(plate, x_curr, step, current_angles, current_drivers, 0.0, 0.0, E_mem, E_ben, E_hin, export_fields, out_path);
        };
        
        solver.KinematicDeployment(plate, x0, num_steps, on_step_converged);
    }
    else if (analysis_type == "ProgressiveCreasing") {
        // docs/Progressive_Creasing_Brief_for_Code.md, Step 1.
        const json& pc = config.at("creasing");
        Progressive::Config pcfg;
        pcfg.num_steps         = pc.value("num_steps", 40);
        pcfg.fallback_substeps = pc.value("fallback_substeps", 4);
        pcfg.L_ref             = pc.at("L_ref");
        if (pc.contains("refit")) {
            // true/false kept from the first re-fit runs: true is the l2 fit.
            if (pc["refit"].is_boolean()) pcfg.refit = pc["refit"].get<bool>() ? "l2" : "none";
            else pcfg.refit = pc["refit"].get<std::string>();
        }
        pcfg.diagnose_step     = pc.value("diagnose_step", -1);
        if (pc.contains("guard")) {
            const json& g = pc["guard"];
            pcfg.guard_max_theta_deg = g.value("max_theta_deg", pcfg.guard_max_theta_deg);
            pcfg.guard_energy_factor = g.value("energy_factor", pcfg.guard_energy_factor);
            pcfg.guard_max_motion    = g.value("max_motion", pcfg.guard_max_motion);
            pcfg.guard_max_motion_topology = g.value("max_motion_topology", pcfg.guard_max_motion_topology);
        }
        pcfg.log_file = (lastdot != string::npos) ? out_path.substr(0, lastdot) + "_steps.csv" : out_path + "_steps.csv";

        // The lines to make, in order. Each is one cut carrying one or more folds:
        //   "lines": [{"crease_tags": [7, 8], "start": [-1, 0], "tip_step": 1, "skip": []}, ...]
        // "start" picks the end the tip leaves from (the nearer end of the cut);
        // "mode" (one_end | one_end_reverse | middle_out) can be given instead.
        // The single-crease form "crease_tag" + "mode" + "tip_step" is one such line.
        std::vector<json> line_cfgs;
        if (pc.contains("lines")) {
            for (const auto& lc : pc["lines"]) line_cfgs.push_back(lc);
        } else {
            json lc;
            lc["crease_tags"] = std::vector<int>{pc.at("crease_tag").get<int>()};
            lc["mode"] = pc.value("mode", std::string("one_end"));
            lc["tip_step"] = pc.value("tip_step", 1);
            line_cfgs.push_back(lc);
        }

        // Each line's cut is built exactly as the all-at-once cut is, so a finished
        // line reproduces it. Every other cut stays as it is.
        std::vector<Progressive::Line> lines;
        std::vector<Enrichment::FoldLine> static_creases = all_creases;
        for (const auto& lc : line_cfgs) {
            Progressive::Line line;
            const std::vector<int> tags = lc.at("crease_tags").get<std::vector<int>>();
            for (int tag : tags) {
                int fi = -1;
                for (size_t i = 0; i < plate.folds.size(); ++i)
                    if (plate.folds[i].crease_tag == tag) fi = static_cast<int>(i);
                if (fi < 0) throw std::runtime_error("creasing: crease tag " + std::to_string(tag) + " is not one of the folds");
                if (plate.folds[fi].flagRig) throw std::runtime_error("a progressive crease must be elastic (is_rigid false)");
                line.folds.push_back(fi);
            }
            line.full = MeshUtils::extractCurve(elements, raw_nodes, tags);
            bool found = false;
            for (size_t i = 0; i < static_creases.size(); ++i) {
                if (static_creases[i].vertices == line.full.vertices) {
                    static_creases.erase(static_creases.begin() + static_cast<long>(i));
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("each progressive line must be one cut: list exactly its crease tags as one \"cuts\" entry (or omit \"cuts\" for a single crease)");
            line.mode = lc.value("mode", std::string("one_end"));
            if (lc.contains("start")) {
                const Eigen::Vector2d p(lc["start"][0].get<double>(), lc["start"][1].get<double>());
                line.mode = ((line.full.vertices.front() - p).norm() <= (line.full.vertices.back() - p).norm())
                          ? "one_end" : "one_end_reverse";
            }
            line.tip_step = lc.value("tip_step", 1);
            if (lc.contains("skip")) line.skip = lc["skip"].get<std::vector<int>>();
            lines.push_back(line);
        }

        auto on_step_converged = [&](int step, const Eigen::VectorXd& x_curr, const std::vector<double>& current_angles, const std::vector<int>& current_drivers, double E_mem, double E_ben, double E_hin) {
            exportStep(plate, x_curr, step, current_angles, current_drivers, 0.0, 0.0, E_mem, E_ben, E_hin, export_fields, out_path);
        };
        auto rebuild = [&](const std::vector<Enrichment::FoldLine>& cuts) {
            SetupHelper::recomputeShapeFunctions(plate, cuts);
        };
        Progressive::run(solver, plate, x0, lines, static_creases, pcfg, rebuild, on_step_converged);
    }
    else if (analysis_type == "TargetDirectedArcLength") {
            cout << "\n--- Starting Target-Directed Arc-Length Continuation ---" << endl;
            
            auto on_step_converged = [&](int step, std::vector<double> current_angles, const Eigen::VectorXd& x_curr,
                                         double W_ext, double E_mem, double E_ben, double E_hin) {
                double disp = solver.ComputeTrackingDisplacement(plate, x_curr);
                exportStep(plate, x_curr, step, current_angles, static_drivers, disp,
                                          W_ext, E_mem, E_ben, E_hin,
                                          export_fields, out_path);
            };
            
            double max_load = config["solver"]["max_load"];
            // Extract tolerance from JSON with a default fallback
            double target_tol_deg = config["solver"].value("target_tol_deg", 2.0);

            // Pass the tolerance variable into the updated function signature
            solver.TargetDirectedArcLength(plate, 0.0, x0, max_load, is_rigid, target_tol_deg, on_step_converged);
        }
    else if (analysis_type == "KinematicArcLength") {
            
            auto on_step_converged = [&](int step, std::vector<double> current_data, const Eigen::VectorXd& x_curr,
                                         double W_ext, double E_mem, double E_ben, double E_hin) {
                
                double disp = solver.ComputeTrackingDisplacement(plate, x_curr);
                
                // current_data contains [mu, angle1, angle2, ...], perfectly matching the structure expected by writeBinaryStep
                exportStep(plate, x_curr, step, current_data, static_drivers, disp,
                                          W_ext, E_mem, E_ben, E_hin,
                                          export_fields, out_path);
            };
            
            solver.KinematicArcLength(plate, x0, is_rigid, on_step_converged);
        }
    else if (analysis_type == "TangentCheck") {
            // =================================================================
            // Verify that the assembled K really is d(F_int)/dx.
            //
            // An inconsistent tangent does not make Newton fail - it degrades
            // quadratic convergence to linear, which looks like "needs more
            // iterations" rather than a bug. Central differences settle it.
            // =================================================================
            const int ns = static_cast<int>(plate.nodes.size()) * 3;

            int    n_dirs = config["solver"].value("fd_directions", 6);
            double h_rel  = config["solver"].value("fd_step", 1e-6);
            unsigned seed = config["solver"].value("fd_seed", 12345u);

            // Evaluate away from the undeformed state so the geometric and fold
            // terms are actually exercised.
            Eigen::VectorXd x = x0;
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> U(-1.0, 1.0);
            const double xs = std::max(1e-3, x.norm() / std::sqrt(static_cast<double>(ns)));
            for (int i = 0; i < ns; ++i) x(i) += 0.05 * xs * U(rng);

            const double h = h_rel * std::max(1.0, x.lpNorm<Eigen::Infinity>());

            for (int pass = 0; pass < 2; ++pass) {
                const bool rig = (pass == 0);
                printf("\n=================================================================\n");
                printf(" TANGENT CHECK  flagRig = %s   (%s)\n", rig ? "true" : "false",
                       rig ? "face terms only" : "face + elastic fold terms");
                printf("=================================================================\n");
                printf(" step h = %.3e,  %d random directions\n\n", h, n_dirs);

                Eigen::SparseMatrix<double> K;
                Eigen::VectorXd F_dummy;
                plate.assembleSystem(K, F_dummy, x, rig);

                printf(" dir        |K*v|          |FD|      rel.error\n");
                printf(" ---------------------------------------------\n");
                double worst = 0.0;
                for (int d = 0; d < n_dirs; ++d) {
                    Eigen::VectorXd v(ns);
                    for (int i = 0; i < ns; ++i) v(i) = U(rng);
                    v.normalize();

                    Eigen::VectorXd Fp = plate.assembleSystemInternalOnly(x + h * v, rig);
                    Eigen::VectorXd Fm = plate.assembleSystemInternalOnly(x - h * v, rig);
                    Eigen::VectorXd fd = (Fp - Fm) / (2.0 * h);
                    Eigen::VectorXd an = K.selfadjointView<Eigen::Upper>() * v;

                    const double denom = std::max(fd.norm(), 1e-30);
                    const double rel   = (fd - an).norm() / denom;
                    worst = std::max(worst, rel);
                    printf(" %3d   %12.5e  %12.5e   %10.3e\n", d, an.norm(), fd.norm(), rel);
                }
                printf("\n >>> worst relative error = %.3e   %s\n", worst,
                       worst < 1e-6  ? "CONSISTENT"
                     : worst < 1e-3  ? "SUSPECT - larger than finite-difference noise"
                                     : "INCONSISTENT - tangent does not match the residual");
            }
        }
    if (Exporter::failed_writes > 0) {
        cerr << "\n!!! WARNING: " << Exporter::failed_writes
             << " step(s) failed to write to " << out_path
             << ". The results file is INCOMPLETE." << endl;
    }

    auto total_end = high_resolution_clock::now();
    cout << "\n=======================================================" << endl;
    cout << "Total Program Execution:  " << duration_cast<milliseconds>(total_end - total_start).count() / 1000.0 << " s" << endl;
    cout << "=======================================================" << endl;

    return 0;
    
    
}

// Thin wrapper so a thrown error reports cleanly and sets an exit status, instead
// of terminating via an uncaught exception. Several paths throw: the RKPM
// moment-matrix guards in GlobalRKPM::solve, the KKT solve in
// GetInitialConditionWithFolds, the CSR mapping check in initSparsityPattern, and
// nlohmann::json on a malformed config.
int main(int argc, char* argv[]) {
    try {
        return run_solver(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "\n=======================================================\n"
                  << "!!! FATAL: " << e.what() << "\n"
                  << "=======================================================" << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n!!! FATAL: unknown exception; aborting." << std::endl;
        return 1;
    }
}
