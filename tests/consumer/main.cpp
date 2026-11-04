// Downstream use of the Curvami core on a toy problem.
//
// Two DOFs: x0 on a grounded spring (k0), joined to x1 by a cubic spring,
// f(d) = kc*d + a*d^3 with d = x1 - x0. Load lambda on x1, constraint x0 = 0.
// Equilibrium: kc*x1 + a*x1^3 = lambda. With kc = a = 1, lambda = 2 gives x1 = 1.

// Every core header, to show each is self-contained under the curvami/ prefix.
#include "curvami/MeshfreeData.h"
#include "curvami/GmshReader.h"
#include "curvami/Quadrature.h"
#include "curvami/FRNN.h"
#include "curvami/GlobalRKPM.h"
#include "curvami/Enrichment.h"
#include "curvami/MeshUtils.h"
#include "curvami/RestartManager.h"
#include "curvami/solver/Problem.h"
#include "curvami/solver/MumpsSymmetricSolver.h"
#include "curvami/solver/ConstraintProjector.h"
#include "curvami/solver/StepController.h"
#include "curvami/solver/ContinuationSolver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

class CubicSpring : public Problem {
public:
    double k0 = 10.0, kc = 1.0, a = 1.0;

    int numDofs() const override { return 2; }

    void assemble(Eigen::SparseMatrix<double>& K, Eigen::VectorXd& F_int,
                  const Eigen::VectorXd& x) override {
        F_int = internalForce(x);
        const double d  = x(1) - x(0);
        const double kt = kc + 3.0 * a * d * d;
        std::vector<Eigen::Triplet<double>> t = {{0, 0, k0 + kt}, {0, 1, -kt}, {1, 1, kt}};
        K.resize(2, 2);
        K.setFromTriplets(t.begin(), t.end());   // upper triangle only
    }

    Eigen::VectorXd internalForce(const Eigen::VectorXd& x) override {
        const double d = x(1) - x(0);
        const double f = kc * d + a * d * d * d;
        return Eigen::Vector2d(k0 * x(0) - f, f);
    }

    Eigen::VectorXd externalForce(const Eigen::VectorXd&) override {
        return Eigen::Vector2d(0.0, 1.0);
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c,
                     Eigen::SparseMatrix<double>& J) override {
        c = Eigen::VectorXd::Constant(1, x(0));
        std::vector<Eigen::Triplet<double>> t = {{0, 0, 1.0}};
        J.resize(1, 2);
        J.setFromTriplets(t.begin(), t.end());
    }

    double monitor(const Eigen::VectorXd& x) const override { return x(1); }
};

// One rotational DOF theta: spring k*theta under a follower torque
// lambda*m*sin(beta - theta), which turns with the body. The torque depends on
// theta, so the exact tangent needs the load stiffness
// K_load = d/dtheta [m sin(beta - theta)] = -m cos(beta - theta).
// Exact equilibrium path: lambda(theta) = k*theta / (m sin(beta - theta)).
class FollowerSpring : public Problem {
public:
    double k = 2.0, m = 1.0, beta = 0.5 * M_PI;
    bool   use_load_stiffness = true;

    // Diagnostics: residual |F_int - lambda*F_ext| at every tangent assembly,
    // for the lambda the test sets (LineSearch runs at a fixed lambda).
    double lambda_for_log = 0.0;
    std::vector<double> residuals;
    int assemblies = 0;

    int numDofs() const override { return 1; }

    void assemble(Eigen::SparseMatrix<double>& K, Eigen::VectorXd& F_int,
                  const Eigen::VectorXd& x) override {
        ++assemblies;
        F_int = internalForce(x);
        residuals.push_back(std::abs(F_int(0) - lambda_for_log * externalForce(x)(0)));
        std::vector<Eigen::Triplet<double>> t = {{0, 0, k}};
        K.resize(1, 1);
        K.setFromTriplets(t.begin(), t.end());
    }

    Eigen::VectorXd internalForce(const Eigen::VectorXd& x) override {
        return Eigen::VectorXd::Constant(1, k * x(0));
    }

    Eigen::VectorXd externalForce(const Eigen::VectorXd& x) override {
        return Eigen::VectorXd::Constant(1, m * std::sin(beta - x(0)));
    }

    bool loadStiffness(Eigen::SparseMatrix<double>& K_load, const Eigen::VectorXd& x) override {
        if (!use_load_stiffness) return false;
        std::vector<Eigen::Triplet<double>> t = {{0, 0, -m * std::cos(beta - x(0))}};
        K_load.resize(1, 1);
        K_load.setFromTriplets(t.begin(), t.end());
        return true;
    }

    void constraints(const Eigen::VectorXd&, Eigen::VectorXd& c,
                     Eigen::SparseMatrix<double>& J) override {
        c.resize(0);
        J.resize(0, 1);
    }

    double monitor(const Eigen::VectorXd& x) const override { return x(0); }

    double pathError(double lambda, double theta) const {
        return std::abs(lambda - k * theta / (m * std::sin(beta - theta)));
    }
};

int main() {
    CubicSpring prob;
    int failures = 0;
    auto check = [&](const char* what, double err, double tol) {
        const bool ok = err <= tol;
        std::printf("[consumer] %-45s error %.3e  %s\n", what, err, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    };

    // Newton with backtracking at a fixed load.
    {
        ContinuationSolver solver;
        Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
        const bool converged = solver.LineSearch(prob, 2.0, x);
        check("LineSearch converged", converged ? 0.0 : 1.0, 0.0);
        check("LineSearch x1 = 1 at lambda = 2", std::abs(x(1) - 1.0), 1e-8);
        check("LineSearch constraint x0 = 0", std::abs(x(0)), 1e-12);
    }

    // Arc-length continuation to lambda = 5; every accepted step must satisfy
    // the equilibrium and the constraint.
    {
        ContinuationSolver solver;
        solver.psi = 1.0;
        solver.ds = 0.2;
        solver.dsmax = 0.5;
        int steps = 0;
        double worst_eq = 0.0, worst_c = 0.0, last_lambda = 0.0;
        solver.ArcLength(prob, 0.0, Eigen::VectorXd::Zero(2), 5.0,
            [&](int, double lambda, const Eigen::VectorXd& x, double) {
                ++steps;
                last_lambda = lambda;
                worst_eq = std::max(worst_eq, std::abs(x(1) + x(1) * x(1) * x(1) - lambda));
                worst_c  = std::max(worst_c, std::abs(x(0)));
            });
        check("ArcLength took steps and passed lambda = 5", (steps > 0 && last_lambda > 5.0) ? 0.0 : 1.0, 0.0);
        check("ArcLength equilibrium x1 + x1^3 = lambda", worst_eq, 1e-8);
        check("ArcLength constraint x0 = 0", worst_c, 1e-12);
        std::printf("[consumer] ArcLength: %d steps, final lambda %.4f\n", steps, last_lambda);
    }

    // Follower load: Newton with and without the load stiffness.
    {
        std::printf("\n[consumer] Follower torque, LineSearch at lambda = 1 (k = 2, m = 1, beta = pi/2)\n");
        int iters[2] = {0, 0};
        double theta[2] = {0.0, 0.0};
        for (int with = 1; with >= 0; --with) {
            FollowerSpring fs;
            fs.use_load_stiffness = (with == 1);
            fs.lambda_for_log = 1.0;
            ContinuationSolver solver;
            Eigen::VectorXd x = Eigen::VectorXd::Zero(1);
            const bool ok = solver.LineSearch(fs, 1.0, x);
            iters[with] = fs.assemblies;
            theta[with] = x(0);
            std::printf("[consumer]   %-24s converged=%d  Newton iterations %2d  theta %.15f\n",
                        with ? "with loadStiffness:" : "without loadStiffness:", ok, fs.assemblies, x(0));
            std::printf("[consumer]     residual per iteration:");
            for (double r : fs.residuals) std::printf(" %.1e", r);
            std::printf("\n");
            check(with ? "LineSearch converged (with K_load)" : "LineSearch converged (without K_load)",
                  ok ? 0.0 : 1.0, 0.0);
            check(with ? "  equilibrium on exact path (with)" : "  equilibrium on exact path (without)",
                  fs.pathError(1.0, x(0)), 1e-7);
        }
        check("LineSearch: fewer iterations with K_load", iters[1] < iters[0] ? 0.0 : 1.0, 0.0);
        check("LineSearch: same converged theta", std::abs(theta[1] - theta[0]), 1e-8);
    }
    // The solver converges the force residual to tol_force = 1e-8; on this path
    // that is a lambda error of up to 1e-8 / (m sin(beta - theta)) ~ 1.4e-8, so
    // path agreement is checked at 1e-7.
    {
        std::printf("\n[consumer] Follower torque, ArcLength to lambda = 2\n");
        int iters[2] = {0, 0}, steps[2] = {0, 0};
        double worst[2] = {0.0, 0.0};
        for (int with = 1; with >= 0; --with) {
            FollowerSpring fs;
            fs.use_load_stiffness = (with == 1);
            ContinuationSolver solver;
            solver.psi = 1.0;
            solver.ds = 0.2;
            solver.dsmax = 0.2;   // cap the step so both runs take comparable steps
            solver.ArcLength(fs, 0.0, Eigen::VectorXd::Zero(1), 2.0,
                [&](int, double lambda, const Eigen::VectorXd& x, double) {
                    ++steps[with];
                    worst[with] = std::max(worst[with], fs.pathError(lambda, x(0)));
                });
            iters[with] = fs.assemblies;
            std::printf("[consumer]   %-24s %2d steps, %3d tangent assemblies (%.1f per step), max path error %.1e\n",
                        with ? "with loadStiffness:" : "without loadStiffness:",
                        steps[with], iters[with], double(iters[with]) / std::max(steps[with], 1), worst[with]);
            check(with ? "ArcLength on exact path (with K_load)" : "ArcLength on exact path (without K_load)",
                  worst[with], 1e-7);
        }
        check("ArcLength: fewer iterations per step with K_load",
              double(iters[1]) / steps[1] < double(iters[0]) / steps[0] ? 0.0 : 1.0, 0.0);
    }

    // Past lambda* the tangent without K_load is too wrong to converge at all:
    // the corrector's contraction factor lambda*m*sin(theta)/k reaches 1 where
    // theta*tan(theta) = 1, i.e. theta = 0.8603, lambda* = 2.6245 (k = 2).
    {
        std::printf("\n[consumer] Follower torque, ArcLength to lambda = 3, at most 200 steps\n");
        double reached[2] = {0.0, 0.0};
        for (int with = 1; with >= 0; --with) {
            FollowerSpring fs;
            fs.use_load_stiffness = (with == 1);
            ContinuationSolver solver;
            solver.psi = 1.0;
            solver.ds = 0.2;
            solver.dsmax = 0.2;
            solver.kmax = 200;
            solver.ArcLength(fs, 0.0, Eigen::VectorXd::Zero(1), 3.0,
                [&](int, double lambda, const Eigen::VectorXd&, double) { reached[with] = lambda; });
            std::printf("[consumer]   %-24s reached lambda %.4f\n",
                        with ? "with loadStiffness:" : "without loadStiffness:", reached[with]);
        }
        std::printf("[consumer]   (without it the corrector cannot pass lambda* = 2.6245)\n");
        check("ArcLength with K_load passes lambda = 3", reached[1] > 3.0 ? 0.0 : 1.0, 0.0);
    }

    std::printf("[consumer] %s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
