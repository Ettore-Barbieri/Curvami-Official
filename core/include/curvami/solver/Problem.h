#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

// =========================================================
// PROBLEM INTERFACE FOR THE GENERIC CONTINUATION SOLVERS
// =========================================================
// What ContinuationSolver needs from a discretized problem: a residual
// F_int(x) - lambda * F_ext(x) = 0 subject to equality constraints c(x) = 0.
// The solver owns no physics and makes no assumption about DOFs per node; the
// state vector has numDofs() entries in whatever layout the problem uses.
//
// Methods are non-const: an implementation may cache state derived from x
// (the plate updates its node kinematics on every call).
class Problem {
public:
    virtual ~Problem() = default;

    virtual int numDofs() const = 0;

    // Tangent K = dF_int/dx and F_int at x.
    // K must hold the UPPER TRIANGLE only (the solvers read it through
    // selfadjointView<Upper> and factorize the KKT with symmetric MUMPS), and
    // its sparsity pattern should not change between calls: MUMPS repeats the
    // symbolic analysis only when the pattern changes.
    virtual void assemble(Eigen::SparseMatrix<double>& K,
                          Eigen::VectorXd& F_int,
                          const Eigen::VectorXd& x) = 0;

    // F_int at x without the tangent (residual evaluations in the line search).
    virtual Eigen::VectorXd internalForce(const Eigen::VectorXd& x) = 0;

    // Reference load vector; the solver scales it by lambda.
    virtual Eigen::VectorXd externalForce(const Eigen::VectorXd& x) = 0;

    // Load stiffness K_load = d(externalForce)/dx, for loads that depend on the
    // configuration (follower loads). The residual is F_int(x) - lambda*F_ext(x),
    // so its exact tangent is K - lambda*K_load; without this the solvers use K
    // alone and Newton converges only linearly when F_ext depends on x.
    //
    // Return false (the default) when F_ext does not depend on x; K_load is then
    // not read and the solvers do exactly what they did before this existed.
    // Return true after filling K_load, which must be:
    //  - the derivative of the REFERENCE load (lambda-independent): the solver
    //    applies the factor lambda itself;
    //  - numDofs() x numDofs(), UPPER TRIANGLE only, like assemble();
    //  - SYMMETRIC in full. The KKT is factorized by symmetric MUMPS, so this
    //    holds for conservative loads (F_ext = grad V, K_load = Hessian of V).
    //    A non-conservative follower load has a non-symmetric K_load; passing
    //    its symmetric part is allowed but Newton will not be quadratic.
    // Its pattern need NOT be a subset of K's: the solver forms K - lambda*K_load,
    // whose pattern is the union. Keep both patterns stable between calls, since
    // MUMPS repeats the symbolic analysis whenever the combined pattern changes.
    virtual bool loadStiffness(Eigen::SparseMatrix<double>& K_load, const Eigen::VectorXd& x) {
        (void)K_load; (void)x;
        return false;
    }

    // Equality constraints c(x) = 0 and their Jacobian J = dc/dx
    // (numDofs() columns; zero rows when unconstrained).
    virtual void constraints(const Eigen::VectorXd& x,
                             Eigen::VectorXd& c,
                             Eigen::SparseMatrix<double>& J) = 0;

    // Scalar printed next to lambda in the step header, for monitoring only.
    virtual double monitor(const Eigen::VectorXd& x) const { (void)x; return 0.0; }
};
