#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>

// =========================================================
// ARC-LENGTH STEP CONTROL
// =========================================================
// Step-size adaptation (ASLA and the curvature controller), direction-reversal
// and inertia tracking, and the no-progress stall check. Independent of the
// problem being continued: the continuation solver feeds it the corrector
// iteration count, the angle between successive increments, the state and the
// reduced-Hessian negative-mode count.
class StepController {
public:
    int max_iters = 100;

    double ds    = 0.1;
    double dsmin = 1e-6;
    double dsmax = 10.0;

    int Id = 6;

    // The predictor reverses direction when the tangent d_T flips sign, detected by
    // Delta_U_old . d_T < 0. That test cannot tell a genuine flip from a sharp turn,
    // and a single wrong call is self-sustaining: once Delta_U_old points backwards
    // the test keeps firing and the continuation retraces its own path to the start.
    //
    // d_T can only change sign where the KKT passes through singularity, which is
    // exactly a change of inertia. MUMPS (sym=2) reports the negative pivot count in
    // INFOG(12) as a byproduct of the factorization already performed, so the
    // crossing can be confirmed for free. For a regular saddle point with J of full
    // row rank the count is nc, hence neg(reduced) = INFOG(12) - nc is the number of
    // negative modes of the reduced Hessian: 0 while stable, 1 past a limit point.
    //
    // OPT-IN (default off). On configs where the snap-back test never fires this is
    // inert, but where it does fire the veto changes which branch the continuation
    // follows - on OneCurve it turns around at lambda = -4.9 and climbs to +114.7
    // in 8 steps instead of excursioning to lambda = -196 and back over 18. Both are
    // legitimate arc-length traversals; which is the physical one is a question about
    // the structure, not the solver. Enable with solver.snapback_inertia_veto.
    bool snapback_inertia_veto = false;

    int  prev_neg_modes_ = -1;   // -1 = nothing measured yet this run

    // Accepted steps since the predictor last reversed direction. Two consecutive
    // reversed increments point the same way as each other, so a retrace registers
    // as cos_theta ~ 1 - a "straight path" - and the curvature controller boosts ds,
    // accelerating the solver along a direction it has just called into question.
    // NO_REVERSAL_YET is large so a run that never reverses behaves as before.
    // OPT-IN (default off), like the veto: it only ever holds ds rather than raising
    // it, but different step sizing gives a different path on a nonlinear problem.
    // Recommended together with snapback_inertia_veto - they address the same
    // pathology from two sides, the spurious reversal and its amplification.
    bool asla_hold_after_reversal = false;

    static constexpr int NO_REVERSAL_YET = 1 << 20;
    int  steps_since_reversal_ = NO_REVERSAL_YET;

    void noteReversal()     { steps_since_reversal_ = 0; }
    void noteAcceptedStep() { if (steps_since_reversal_ < NO_REVERSAL_YET) ++steps_since_reversal_; }

    void resetInertiaTracking() { prev_neg_modes_ = -1;
                                  steps_since_reversal_ = NO_REVERSAL_YET;
                                  resetNoProgress(); }

    // A run pinned at dsmin can keep ACCEPTING steps that go nowhere: with
    // psi = 0 the arc-length metric is displacement-only, so every increment
    // still has length ~ds and nothing looks small - the increments simply
    // cancel, and the configuration oscillates in place. The dsmin abort never
    // fires because it is only reached when a step FAILS, so the run grinds on
    // to kmax. Observed on CurvedPleat h=0.005: 349 accepted steps moved the
    // fold angles 0.0045 deg in total and the mesh by 7e-5.
    //
    // So the test is NET movement over a window of accepted steps, not the size
    // of one increment, and it only arms once ds has bottomed out.
    // The increments are NOT small when this happens - each is still ~ds long -
    // they cancel. So the test is net displacement against distance travelled:
    // a window that walks 25 steps and ends where it began has an efficiency of
    // ~0, while any genuine progress, however slow, walks a straight-ish line
    // and keeps the ratio near 1. Measuring the increments themselves misses it
    // (on CurvedPleat h=0.005 they are ~1e-3 each while the net motion over 25
    // steps is ~5e-6, an efficiency of 0.005).
    // The stall is typically a period-2 limit cycle: the state alternates
    // between two configurations a single increment apart. An ODD window then
    // ends on the opposite phase and reports the full cycle width as "movement"
    // (measured: efficiency 0.00 for even windows, 0.065-0.09 for odd ones), so
    // the window is even and the threshold sits well above both. Genuine
    // progress, however slow, walks a near-straight line at efficiency ~1; a
    // cycle up to ~3 steps wide still comes in under 0.15.
    int    no_progress_window     = 24;    // accepted steps between comparisons
    double no_progress_efficiency = 0.15;  // net movement / path length
    bool   no_progress_abort      = true;  // set false to restore old behaviour

    Eigen::VectorXd no_progress_ref_;
    int    no_progress_count_ = 0;
    double no_progress_path_  = 0.0;

    void resetNoProgress() { no_progress_ref_.resize(0);
                             no_progress_count_ = 0;
                             no_progress_path_  = 0.0; }

    // Returns true when the state has stopped advancing at dsmin.
    bool stalledAtDsmin(const Eigen::VectorXd& x, const Eigen::VectorXd& dU) {
        if (!no_progress_abort) return false;
        if (ds > dsmin * (1.0 + 1e-12)) {     // still free to adapt: not a stall
            resetNoProgress();
            return false;
        }
        if (no_progress_ref_.size() != x.size()) {
            resetNoProgress();
            no_progress_ref_ = x;
            return false;
        }
        no_progress_path_ += dU.norm();
        if (++no_progress_count_ < no_progress_window) return false;

        const double moved = (x - no_progress_ref_).norm();
        const double path  = no_progress_path_;
        no_progress_ref_ = x;
        no_progress_count_ = 0;
        no_progress_path_  = 0.0;
        return path > 0.0 && moved <= no_progress_efficiency * path;
    }

    // Returns true if the requested reversal is confirmed by a limit-point crossing.
    // Always advances the tracker, so a rejected step that re-factorizes at the same
    // configuration sees no change and cannot manufacture a crossing.
    // Advances the tracker on steps where no reversal was requested, so the
    // comparison always refers to the immediately preceding predictor.
    void recordInertia(int neg_modes) {
        prev_neg_modes_ = neg_modes;
    }

    bool confirmReversalByInertia(int neg_modes, const char* legacy_msg, const char* label) {
        const int neg  = neg_modes;
        const bool had = (prev_neg_modes_ >= 0);
        const bool crossed = had && (neg != prev_neg_modes_);
        const int  prev = prev_neg_modes_;
        prev_neg_modes_ = neg;

        if (!snapback_inertia_veto) {
            printf("%s", legacy_msg);          // unchanged output when opted out
            return true;
        }
        if (!had) {
            // No baseline yet on the first predictor of a run: fall back to the
            // original behaviour rather than block a possibly legitimate reversal.
            printf("%s", legacy_msg);
            return true;
        }
        if (crossed) {
            printf(" >>> %s: confirmed by inertia change (%d -> %d) <<<\n", label, prev, neg);
            return true;
        }
        printf(" >>> %s VETOED: reduced-Hessian inertia unchanged at %d, so the tangent"
               " did not flip; continuing forward. <<<\n", label, neg);
        return false;
    }

    // =========================================================
    // ASLA: PURE CRISFIELD SQUARE-ROOT RULE WITH TARGET RELAXATION
    // =========================================================
    void ASLA(int icorr_OK, int k_corr, double cos_theta) {
        double opt_iter = static_cast<double>(Id);
        const double growth_cap = 4.0;
        
        if (cos_theta >= 0.99) {
            opt_iter = std::max(opt_iter, 12.0);
        } else if (cos_theta >= 0.90) {
            opt_iter = std::max(opt_iter, 8.0);
        }
        
        if (icorr_OK == 0) {
            double cut_factor = (k_corr < max_iters) ? 0.50 : 0.75;
            printf("   [ASLA] Step Failed -> Braking (Cut %0.f%%)\n", (1.0 - cut_factor) * 100);
            ds *= cut_factor;
            return;
        }
        
        double ratio = opt_iter / static_cast<double>(std::max(k_corr, 1));
        double scale_factor = std::sqrt(ratio);
        
        scale_factor = std::min(scale_factor, growth_cap);
        
        if (scale_factor > 0.90 && scale_factor < 1.10) {
            scale_factor = 1.0;
        }
        
        ds = std::min(std::max(ds * scale_factor, dsmin), dsmax);
        
        if (scale_factor != 1.0) {
            printf("   [ASLA] Scaling ds by %.2f to %.2e\n", scale_factor, ds);
        } else {
            printf("   [ASLA] Maintaining ds at %.2e\n", ds);
        }
    }
    
    // =========================================================
    // MATLAB-equivalent multi-tiered curvature controller
    // =========================================================
    void ApplyMatlabCurvatureController(int step, double cos_theta) {
        if (step <= 1) return;
        
        const double deg_theta = std::acos(cos_theta) * 180.0 / 3.14159265358979323846;
        
        if (cos_theta >= 0.99) {
            if (asla_hold_after_reversal && steps_since_reversal_ <= 1) {
                // Two consecutive reversed increments look straight to this test.
                // Hold ds rather than accelerate away along a questioned direction.
                printf("   [Path Straight (%.1f deg) but a direction reversal %d step(s) ago"
                       " -> holding ds at %.2e]\n", deg_theta, steps_since_reversal_, ds);
            } else {
                ds = std::min(ds * 1.5, dsmax);
                printf("   [Path Straight (%.1f deg) -> TURBO BOOSTING ds to %.2e]\n", deg_theta, ds);
            }
        } else if (cos_theta >= 0.90 && cos_theta < 0.99) {
            // Gentle curve: Leave ds unchanged after ASLA.
        } else if (cos_theta >= 0.75 && cos_theta < 0.90) {
            ds = std::max(ds * 0.8, dsmin);
            printf("   [Tight Curve (%.1f deg) -> Braking ds to %.2e]\n", deg_theta, ds);
        } else {
            ds = std::max(ds * 0.5, dsmin);
            printf("   [HAIRPIN TURN (%.1f deg) -> Hard Brake ds to %.2e]\n", deg_theta, ds);
        }
    }
};
