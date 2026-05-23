"""
022_validation_suite.py — LSPR Biosensor Validation Suite
==========================================================
Performs Validation 1: Bulk Sensitivity Linearity Test.

Validation 1 — Bulk Sensitivity Linearity
    Bare gold cylinder, sweep background RI (n = 1.33 → 1.45).
    λ_max vs n_bkg must be strictly linear (R² > 0.999).
    Proves the volume source contrast term scales correctly.

Note: Validation 2 (Surface Exponential Decay) has been moved to
      023_decay_test.py for faster standalone troubleshooting.

Usage
-----
    mpirun -n 4 python 022_validation_suite.py

Outputs
-------
    validation_bulk_linearity.png   — linearity test plot
    validation_bulk_results.npz     — raw data for reporting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpi4py import MPI
import importlib.util, sys, pathlib


# ── Load 020_main_solver.py via importlib ─────────────────────────────────────

def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).parent / filename
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_solver = _load("solver", "020_main_solver.py")

build_biosensor_mesh = _solver.build_biosensor_mesh
sweep_wavelengths    = _solver.sweep_wavelengths
BIOSENSOR_PARAMS     = _solver.BIOSENSOR_PARAMS


# ── Peak-finding helper ───────────────────────────────────────────────────────

def find_constrained_peak(wl_nm, q_ext, search_end=400.0):
    """
    Isolates the LSPR dipole peak using a constrained spline search.
    Ignores the taller interband transition above search_end nm.
    """
    spline          = CubicSpline(wl_nm, q_ext)
    dense_wl_search = np.linspace(wl_nm[0], search_end, 5000)
    peak            = float(dense_wl_search[np.argmax(spline(dense_wl_search))])
    return peak, spline


# ── Validation 1: Bulk Sensitivity Linearity ──────────────────────────────────

def run_bulk_linearity_sweep():
    """
    Sweep background refractive index on a bare gold cylinder (no shell).

    Phase 1 runs ONCE — geometry is identical for all n_bkg values.
    Phase 2 updates n_bkg in the context and re-solves per step.

    Returns
    -------
    n_backgrounds    : list of float
    peak_wavelengths : list of float [nm]
    slope            : float — bulk sensitivity [nm/RIU]
    r_squared        : float — linearity coefficient
    """
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print()
        print("=" * 70)
        print("VALIDATION 1 — BULK SENSITIVITY LINEARITY")
        print("=" * 70)
        print("  Bare gold cylinder (no shell), sweep n_bkg = 1.33 → 1.45")
        print()

    n_backgrounds = [1.33, 1.36, 1.39, 1.42, 1.45]

    # Phase 1 — build mesh ONCE (geometry does not change between n_bkg values)
    base_params = {
        **BIOSENSOR_PARAMS,
        "t_shell"  : 0.0,
        # ROI window — avoids interband region
        "wl_start" : 0.330,
        "wl_end"   : 0.400,
        "wl_points": 40,
        # Uncomment for fast troubleshooting (set wl_points=15 also):
        # "lc_au"  : (BIOSENSOR_PARAMS["radius_au"] / 5) * 1.3,
        # "lc_dom" : (0.330 / BIOSENSOR_PARAMS["n_bkg"] / 8) * 1.3,
        # "lc_pml" : (0.330 / BIOSENSOR_PARAMS["n_bkg"] / 5) * 1.3,
    }

    ctx    = build_biosensor_mesh(n_shell=1.33, params=base_params)
    p      = ctx["p"]
    wl_arr = np.linspace(p["wl_start"], p["wl_end"], p["wl_points"])
    wl_nm  = wl_arr * 1e3

    peak_wavelengths = []

    # Phase 2 — update n_bkg in context, re-solve per background index
    for n_bkg in n_backgrounds:

        if comm.rank == 0:
            print(f"  n_bkg = {n_bkg:.2f} ...")

        ctx["n_shell"]    = n_bkg
        ctx["p"]["n_bkg"] = n_bkg

        results = sweep_wavelengths(ctx, wl_arr, verbose=True)
        q_ext   = np.array([r["q_ext"] for r in results])

        peak, _ = find_constrained_peak(wl_nm, q_ext, search_end=400.0)
        peak_wavelengths.append(peak)

        if comm.rank == 0:
            print(f"    λ_max = {peak:.2f} nm")

    if comm.rank == 0:
        # Linear regression
        coeffs           = np.polyfit(n_backgrounds, peak_wavelengths, 1)
        slope, intercept = coeffs
        r_matrix         = np.corrcoef(n_backgrounds, peak_wavelengths)
        r_squared        = float(r_matrix[0, 1] ** 2)

        print()
        print(f"  Bulk sensitivity : {slope:.2f} nm/RIU")
        print(f"  Linearity R²     : {r_squared:.5f}  (target > 0.999)")
        if r_squared > 0.999:
            print("  ✓ LINEARITY CONFIRMED — R² > 0.999")
        elif r_squared > 0.99:
            print("  ⚠ R² > 0.99 — acceptable, consider more sweep points")
        else:
            print("  ✗ Non-linear response — check volume source formulation")

        # Plot
        n_arr    = np.array(n_backgrounds)
        fit_line = slope * n_arr + intercept

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(n_arr, peak_wavelengths, "ko", ms=9, zorder=3,
                label="FEM solver data")
        ax.plot(n_arr, fit_line, "r-", lw=2,
                label=f"Linear fit  (R²={r_squared:.4f})")
        ax.set(xlabel="Background Refractive Index [RIU]",
               ylabel="LSPR Peak Position λ_max [nm]",
               title="Validation 1 — Bulk Index Sensitivity Linearity")
        ax.text(0.05, 0.95,
                f"Sensitivity = {slope:.1f} nm/RIU\nR² = {r_squared:.5f}",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow",
                          edgecolor="orange", lw=1.5))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("validation_bulk_linearity.png", dpi=180,
                    bbox_inches="tight")
        print("  Saved: validation_bulk_linearity.png")

        return n_backgrounds, peak_wavelengths, slope, r_squared

    return n_backgrounds, [], 0.0, 0.0


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    comm = MPI.COMM_WORLD

    # Default values for non-rank-0 processes
    bulk_sens = 0.0
    r2        = 0.0
    n_arr     = []
    peaks     = []

    if comm.rank == 0:
        print("=" * 70)
        print("LSPR BIOSENSOR — VALIDATION SUITE (Bulk Linearity)")
        print("=" * 70)

    n_arr, peaks, bulk_sens, r2 = run_bulk_linearity_sweep()

    if comm.rank == 0:
        print()
        print("=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"  Bulk sensitivity : {bulk_sens:.1f} nm/RIU")
        print(f"  Linearity R²     : {r2:.5f}  "
              f"{'✓' if r2 > 0.999 else '⚠'}")
        print()

        np.savez("validation_bulk_results.npz",
                 n_backgrounds=np.array(n_arr),
                 peak_wavelengths=np.array(peaks),
                 bulk_sensitivity=bulk_sens,
                 r_squared=r2)
        print("  Saved: validation_bulk_results.npz")
        print()
        print("  Next: mpirun -n 4 python 023_decay_test.py")

        plt.show()