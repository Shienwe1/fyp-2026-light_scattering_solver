"""
021_mesh_convergence.py — LSPR Solver Mesh Convergence Study
=============================================================
Validates the numerical stability of the volume-source FEniCSx solver
by tracking convergence of the Extinction Efficiency (Q_ext) at the
LSPR peak wavelength as mesh density increases.

No analytical Mie solution exists for the multi-layered penetrable
geometry (Au core + protein shell + water background), so a
Self-Convergence (Grid Convergence Index) study is used instead:
- Fix the physics at a single demanding wavelength (LSPR peak)
- Solve on progressively finer meshes
- Treat the finest mesh as benchmark
- Compute relative change of coarser meshes against benchmark

Usage
-----
    python 021_mesh_convergence.py

Outputs
-------
    validation_mesh_convergence.png  — convergence plot
    convergence_data.npz             — raw data for reporting
"""

import numpy as np
import matplotlib.pyplot as plt
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


# ── Convergence study parameters ──────────────────────────────────────────────

# Fixed test case — LSPR peak wavelength with protein shell active
TEST_WL      = 0.3581          # [µm] = 358.1 nm
N_SHELL      = 1.45            # protein bound state

# Mesh refinement factors — coarse to ultra-fine
# lc_au, lc_shell, lc_dom are multiplied by each factor inside the loop
MESH_FACTORS = [1.5, 1.2, 1.0, 0.8, 0.6]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print("=" * 70)
        print("LSPR SOLVER — MESH CONVERGENCE STUDY")
        print("=" * 70)
        print(f"  Test wavelength : {TEST_WL*1e3:.1f} nm  (LSPR peak)")
        print(f"  Shell RI        : {N_SHELL} (protein bound state)")
        print(f"  Mesh factors    : {MESH_FACTORS}")
        print(f"  Benchmark mesh  : factor = {MESH_FACTORS[-1]} (finest)")
        print()

    element_counts = []
    dof_counts     = []
    q_ext_results  = []

    # ── Loop over mesh densities ───────────────────────────────────────────────
    for i, f in enumerate(MESH_FACTORS):

        if comm.rank == 0:
            print(f"{'─'*70}")
            print(f"  [{i+1}/{len(MESH_FACTORS)}] Mesh factor = {f:.1f}")

        # Scale mesh sizes by factor f — fed directly into GMSH
        p = {**BIOSENSOR_PARAMS}

        custom_params = {
            **p,
            "wl_start"  : TEST_WL,
            "wl_end"    : TEST_WL,
            "wl_points" : 1,
            # Scaled mesh sizes — without these GMSH ignores f entirely
            "lc_au"     : (p["radius_au"] / 5)               * f,
            "lc_shell"  : (p["t_shell"]   / 3)               * f,
            "lc_dom"    : (p["wl_start"]  / p["n_bkg"] / 8)  * f,
            "lc_pml"    : (p["wl_start"]  / p["n_bkg"] / 5)  * f,
        }

        # Phase 1 — build mesh at this refinement level
        ctx = build_biosensor_mesh(
            n_shell=N_SHELL,
            params=custom_params,
            mesh_filename=f"mesh_conv_f{f:.2f}.msh",
        )

        # Override mesh sizes via lc parameters stored in ctx
        # (mesh already built — track node/dof count for reporting)
        msh      = ctx["msh"]
        V        = ctx["V"]
        n_nodes  = msh.geometry.x.shape[0]
        n_dofs   = V.dofmap.index_map.size_global

        element_counts.append(n_nodes)
        dof_counts.append(n_dofs)

        # Phase 2 — single wavelength solve
        results = sweep_wavelengths(
            ctx, np.array([TEST_WL]), verbose=False
        )
        q_ext = results[0]["q_ext"]
        q_ext_results.append(q_ext)

        if comm.rank == 0:
            print(f"    Nodes  : {n_nodes}")
            print(f"    DOFs   : {n_dofs}")
            print(f"    Q_ext  : {q_ext:.6f}")

    # ── Post-processing ───────────────────────────────────────────────────────
    if comm.rank == 0:

        q_benchmark = q_ext_results[-1]   # finest mesh = ground truth
        rel_errors  = [
            abs(q - q_benchmark) / abs(q_benchmark) * 100
            for q in q_ext_results[:-1]
        ]

        print()
        print("=" * 70)
        print("CONVERGENCE RESULTS")
        print("=" * 70)
        print(f"  Benchmark Q_ext (factor={MESH_FACTORS[-1]}): {q_benchmark:.6f}")
        print()
        print(f"  {'Factor':>8}  {'Nodes':>8}  {'DOFs':>10}  "
              f"{'Q_ext':>12}  {'Rel. Error':>12}")
        print("  " + "─" * 60)

        for i, f in enumerate(MESH_FACTORS):
            err_str = (f"{rel_errors[i]:>10.4f}%"
                       if i < len(rel_errors) else "  (benchmark)")
            print(f"  {f:>8.1f}  {element_counts[i]:>8d}  {dof_counts[i]:>10d}  "
                  f"{q_ext_results[i]:>12.6f}  {err_str}")

        # Convergence verdict
        print()
        if rel_errors:
            finest_err = rel_errors[-1]
            if finest_err < 0.1:
                print(f"  ✓ CONVERGED — relative error {finest_err:.4f}% < 0.1%")
                print("    Mesh is grid-independent at standard resolution.")
            elif finest_err < 1.0:
                print(f"  ⚠ NEARLY CONVERGED — error {finest_err:.4f}% < 1%")
                print("    Consider one more refinement level for publication.")
            else:
                print(f"  ✗ NOT CONVERGED — error {finest_err:.4f}% > 1%")
                print("    Refine further or increase polynomial degree.")

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, ax1 = plt.subplots(figsize=(8, 5))
        fig.suptitle(
            f"Mesh Convergence Study — Penetrable LSPR Solver\n"
            f"λ = {TEST_WL*1e3:.1f} nm, Au 25 nm, shell n = {N_SHELL}",
            fontsize=12, fontweight="bold"
        )

        # Left axis: Q_ext value
        color_q = "tab:blue"
        ax1.set_xlabel("Number of Mesh Nodes", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Extinction Efficiency  Q_ext", color=color_q,
                       fontsize=12, fontweight="bold")
        ax1.plot(element_counts, q_ext_results, "bo-", lw=2.5, ms=9,
                 label="Q_ext", zorder=3)
        ax1.axhline(q_benchmark, color="b", ls="--", lw=1.5, alpha=0.5,
                    label="Asymptotic limit")
        ax1.tick_params(axis="y", labelcolor=color_q)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", fontsize=10)

        # Right axis: relative error
        if rel_errors:
            color_e = "tab:red"
            ax2 = ax1.twinx()
            ax2.set_ylabel("Relative Error vs Benchmark [%]", color=color_e,
                           fontsize=12, fontweight="bold")
            ax2.plot(element_counts[:-1], rel_errors, "rs--", lw=2, ms=8,
                     label="Relative error", zorder=3)
            ax2.axhline(0.1, color="orange", ls=":", lw=1.5, alpha=0.8,
                        label="0.1% threshold")
            ax2.tick_params(axis="y", labelcolor=color_e)
            ax2.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        plt.savefig("validation_mesh_convergence.png", dpi=180,
                    bbox_inches="tight")
        print()
        print("  Saved: validation_mesh_convergence.png")

        # ── Save data ──────────────────────────────────────────────────────────
        np.savez("convergence_data.npz",
                 mesh_factors=np.array(MESH_FACTORS),
                 element_counts=np.array(element_counts),
                 dof_counts=np.array(dof_counts),
                 q_ext=np.array(q_ext_results),
                 rel_errors=np.array(rel_errors + [0.0]),  # benchmark = 0
                 q_benchmark=q_benchmark,
                 test_wl=TEST_WL,
                 n_shell=N_SHELL)
        print("  Saved: convergence_data.npz")
        print()
        print("  Next: python 022_compare.py")

        plt.show()