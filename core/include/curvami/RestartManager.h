#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

namespace RestartManager {

    // Bumped whenever the record layout changes, so an old .res is rejected
    // rather than silently misread.
    inline constexpr int   FORMAT_VERSION = 1;
    inline constexpr char  MAGIC[8]       = {'C','U','R','V','R','E','S','\0'};

    // Instantly dumps the exact Nodal solver state to a binary file
    inline bool save(const std::string& filename, int k, double lambda, double ds, double dl_old, double W_ext,
                     const Eigen::VectorXd& x, const Eigen::VectorXd& dU_old) {
        if (x.size() != dU_old.size()) {
            std::cerr << "!!! Restart save refused: x and dU_old differ in size." << std::endl;
            return false;
        }

        std::ofstream out(filename, std::ios::binary);
        if (!out) return false;

        int version = FORMAT_VERSION;
        out.write(MAGIC, sizeof(MAGIC));
        out.write(reinterpret_cast<const char*>(&version), sizeof(int));

        out.write(reinterpret_cast<const char*>(&k), sizeof(int));
        out.write(reinterpret_cast<const char*>(&lambda), sizeof(double));
        out.write(reinterpret_cast<const char*>(&ds), sizeof(double));
        out.write(reinterpret_cast<const char*>(&dl_old), sizeof(double));
        out.write(reinterpret_cast<const char*>(&W_ext), sizeof(double));

        int ns = static_cast<int>(x.size());
        out.write(reinterpret_cast<const char*>(&ns), sizeof(int));
        out.write(reinterpret_cast<const char*>(x.data()), ns * sizeof(double));
        out.write(reinterpret_cast<const char*>(dU_old.data()), ns * sizeof(double));

        return static_cast<bool>(out);
    }

    // Loads the state back into the solver's memory.
    //
    // Everything is read into locals and only committed to the caller's variables
    // once the whole record has been verified. A truncated file - what a run killed
    // mid-save leaves behind - previously reached x.resize() with an uninitialised
    // size, and reported success either way.
    //
    // expected_ns is the DOF count of the problem being restarted; pass it so a .res
    // belonging to a different mesh is rejected instead of silently resizing.
    inline bool load(const std::string& filename, int& k, double& lambda, double& ds, double& dl_old, double& W_ext,
                     Eigen::VectorXd& x, Eigen::VectorXd& dU_old, int expected_ns = -1) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) return false;

        char magic[sizeof(MAGIC)] = {};
        int version = 0;
        in.read(magic, sizeof(MAGIC));
        in.read(reinterpret_cast<char*>(&version), sizeof(int));
        if (!in || std::string(magic, sizeof(MAGIC)) != std::string(MAGIC, sizeof(MAGIC))) {
            std::cerr << "!!! Restart file " << filename
                      << " is not a Curvami restart record; ignoring." << std::endl;
            return false;
        }
        if (version != FORMAT_VERSION) {
            std::cerr << "!!! Restart file " << filename << " has format version " << version
                      << ", expected " << FORMAT_VERSION << "; ignoring." << std::endl;
            return false;
        }

        int k_l = 0, ns = 0;
        double lambda_l = 0.0, ds_l = 0.0, dl_l = 0.0, W_l = 0.0;

        in.read(reinterpret_cast<char*>(&k_l), sizeof(int));
        in.read(reinterpret_cast<char*>(&lambda_l), sizeof(double));
        in.read(reinterpret_cast<char*>(&ds_l), sizeof(double));
        in.read(reinterpret_cast<char*>(&dl_l), sizeof(double));
        in.read(reinterpret_cast<char*>(&W_l), sizeof(double));
        in.read(reinterpret_cast<char*>(&ns), sizeof(int));
        if (!in) {
            std::cerr << "!!! Restart file " << filename << " is truncated in its header; ignoring."
                      << std::endl;
            return false;
        }

        if (ns <= 0) {
            std::cerr << "!!! Restart file " << filename << " reports a non-positive DOF count ("
                      << ns << "); ignoring." << std::endl;
            return false;
        }
        if (expected_ns > 0 && ns != expected_ns) {
            std::cerr << "!!! Restart file " << filename << " holds " << ns
                      << " DOFs but this problem has " << expected_ns
                      << " - it belongs to a different mesh; ignoring." << std::endl;
            return false;
        }

        Eigen::VectorXd x_l(ns), dU_l(ns);
        in.read(reinterpret_cast<char*>(x_l.data()), ns * sizeof(double));
        in.read(reinterpret_cast<char*>(dU_l.data()), ns * sizeof(double));
        if (!in) {
            std::cerr << "!!! Restart file " << filename << " is truncated in its state arrays; ignoring."
                      << std::endl;
            return false;
        }
        if (!x_l.allFinite() || !dU_l.allFinite()) {
            std::cerr << "!!! Restart file " << filename << " contains non-finite values; ignoring."
                      << std::endl;
            return false;
        }

        // Only now is it safe to overwrite the caller's state.
        k = k_l; lambda = lambda_l; ds = ds_l; dl_old = dl_l; W_ext = W_l;
        x = std::move(x_l); dU_old = std::move(dU_l);
        return true;
    }
}
