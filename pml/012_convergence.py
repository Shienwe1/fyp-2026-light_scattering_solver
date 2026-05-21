"""
convergence.py — Mesh Convergence Study
========================================
Runs the PEC cylinder solver on progressively refined meshes,
computes L2 errors against the Mie reference, and plots convergence rates.

Imports
-------
    solver.py  — solve_pec_cylinder, MIE_PARAMS
    compare.py — compute_mie, compute_errors

Usage
-----
    python convergence.py

Outputs
-------
    convergence_results.png   — error vs h  +  norm vs DOFs
    convergence_results.npz   — raw data for further post-processing
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem import assemble_scalar
import ufl
from ufl import inner, dx
import importlib.util, sys, pathlib

def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).parent / filename
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_solver  = _load("solver",  "010_main_solver.py")
_compare = _load("compare", "011_compare.py")

MIE_PARAMS         = _solver.MIE_PARAMS
solve_pec_cylinder = _solver.solve_pec_cylinder
compute_mie        = _compare.compute_mie
compute_errors     = _compare.compute_errors


# ── Mesh refinement levels ────────────────────────────────────────────────────
# lc = mesh_factor × (base element size)
# Ordered coarse → fine.  Finest level is used as the norm reference.
MESH_FACTORS = [1.2, 0.9, 0.6, 0.45, 0.35]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _l2_norm(fn):
    """Global L2 norm of a DOLFINx Function."""
    val = assemble_scalar(form(inner(fn, fn) * dx))
    return float(np.sqrt(abs(val)))


def _eval_on_line(Es_cg, x_pts):
    """
    Sample a CG-interpolated scattered field along the x-axis.

    Parameters
    ----------
    Es_cg : dolfinx Function (Lagrange / CG space)
    x_pts : 1-D array of x-coordinates (y = 0)

    Returns
    -------
    E_mag : real array — |E| at each point
    """
    from dolfinx.geometry import (bb_tree, compute_collisions_points,
                                   compute_colliding_cells)

    mesh  = Es_cg.function_space.mesh
    pts3d = np.column_stack([x_pts, np.zeros_like(x_pts), np.zeros_like(x_pts)])
    tree  = bb_tree(mesh, mesh.topology.dim)
    cands = compute_collisions_points(tree, pts3d)
    cells = compute_colliding_cells(mesh, cands, pts3d)

    Ex = np.zeros(len(x_pts), dtype=np.complex128)
    Ey = np.zeros(len(x_pts), dtype=np.complex128)

    for i in range(len(x_pts)):
        cl = cells.links(i)
        if len(cl) > 0:
            val   = Es_cg.eval(pts3d[i:i+1], [cl[0]])
            Ex[i] = val[0]
            Ey[i] = val[1]

    return np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    p    = MIE_PARAMS
    wl   = p["wl"]
    k0   = 2 * np.pi / wl
    R_cyl, l_dom = p["R_cyl"], p["l_dom"]
    degree = p["degree"]

    # Comparison line (evaluate up to edge of square physical domain)
    x_pts = np.linspace(R_cyl * 1.05, (l_dom / 2) * 0.95, 300)
    y_pts = np.zeros_like(x_pts)

    # Mie reference (computed once)
    if comm.rank == 0:
        print("=" * 60)
        print("Computing Mie reference...")
    Ex_mie, Ey_mie = compute_mie(x_pts, y_pts)
    E_mag_mie = np.sqrt(np.abs(Ex_mie)**2 + np.abs(Ey_mie)**2)

    # Storage
    records = []   # one dict per mesh level

    # ── Solve on each mesh ────────────────────────────────────────────────────
    for level, mf in enumerate(MESH_FACTORS):
        lc_cyl = mf * wl / 15
        lc_dom = mf * wl / 5

        if comm.rank == 0:
            print()
            print(f"{'='*60}")
            print(f"Level {level+1}/{len(MESH_FACTORS)}  "
                  f"(mesh_factor = {mf})")
            print(f"  lc_cyl = {lc_cyl:.5f}  |  lc_dom = {lc_dom:.5f}")

        Es_h, Eb, mesh, num_cells, num_dofs, h_avg = solve_pec_cylinder(
            lc_cyl, lc_dom,
            mesh_filename=f"mesh_level{level+1}.msh",
        )

        # L2 norm of scattered field
        l2_norm = _l2_norm(Es_h)

        # Interpolate to CG for pointwise sampling
        V_cg  = functionspace(mesh, ("Lagrange", degree, (mesh.geometry.dim,)))
        Es_cg = Function(V_cg)
        Es_cg.interpolate(Es_h)
        E_mag_fem = _eval_on_line(Es_cg, x_pts)

        # L2 error vs Mie
        errors = compute_errors(x_pts, E_mag_fem, E_mag_mie, wl)

        kh = k0 * h_avg

        records.append(dict(
            level      = level + 1,
            mf         = mf,
            h_avg      = h_avg,
            kh         = kh,
            num_cells  = num_cells,
            num_dofs   = num_dofs,
            l2_norm    = l2_norm,
            l2_err     = errors["l2"],
            near_err   = errors["near_err"],
            far_err    = errors["far_err"],
        ))

        if comm.rank == 0:
            print(f"  Cells={num_cells}  DOFs={num_dofs}  "
                  f"h_avg={h_avg:.5f}  kh={kh:.4f}")
            print(f"  ‖Es‖={l2_norm:.4e}  L2_err={errors['l2']*100:.3f}%")

    # ── Convergence table ─────────────────────────────────────────────────────
    if comm.rank == 0:
        print()
        print("=" * 60)
        print("CONVERGENCE TABLE")
        print("=" * 60)
        header = (f"{'Lv':>3}  {'h_avg':>8}  {'kh':>6}  "
                  f"{'Cells':>7}  {'DOFs':>8}  "
                  f"{'‖Es‖':>10}  {'L2 err':>9}  {'Rate':>6}")
        print(header)
        print("-" * len(header))

        for i, r in enumerate(records):
            rate_str = "  —  "
            if i > 0:
                prev = records[i - 1]
                if prev["l2_err"] > 0 and r["l2_err"] > 0:
                    rate = (np.log(prev["l2_err"] / r["l2_err"])
                            / np.log(prev["h_avg"] / r["h_avg"]))
                    rate_str = f"{rate:+.2f}"
            regime = ("✓" if r["kh"] < 0.3
                      else "⚠" if r["kh"] < 0.5 else "✗")
            print(f"{r['level']:>3}  {r['h_avg']:>8.5f}  "
                  f"{r['kh']:>6.4f}{regime}  "
                  f"{r['num_cells']:>7d}  {r['num_dofs']:>8d}  "
                  f"{r['l2_norm']:>10.4e}  "
                  f"{r['l2_err']*100:>8.3f}%  {rate_str:>6}")

        print()
        print("Regime legend:  ✓ kh<0.3 (asymptotic)  "
              "⚠ kh<0.5 (near boundary)  ✗ kh≥0.5 (pre-asymptotic)")
        print(f"Expected rate for degree-{p['degree']} N1curl: ~{p['degree']:.0f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if comm.rank == 0:
        h_arr    = np.array([r["h_avg"]    for r in records])
        l2_arr   = np.array([r["l2_err"]   for r in records])
        norm_arr = np.array([r["l2_norm"]  for r in records])
        dof_arr  = np.array([r["num_dofs"] for r in records])
        kh_arr   = np.array([r["kh"]       for r in records])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Mesh Convergence Study — PEC Cylinder", fontsize=14,
                     fontweight="bold")

        # Left: L2 error vs h (vs Mie reference)
        ax = axes[0]
        mask = l2_arr > 0
        ax.loglog(h_arr[mask], l2_arr[mask], "bo-", lw=2.5, ms=9,
                  label="L2 error vs Mie", zorder=3)

        if mask.sum() >= 2:
            h_ref = np.array([h_arr[mask][0], h_arr[mask][-1]])
            for p_ord, ls, alpha in [(2, "--", 0.55), (3, "-.", 0.45)]:
                C = l2_arr[mask][0] / h_arr[mask][0] ** p_ord
                ax.loglog(h_ref, C * h_ref ** p_ord, ls, alpha=alpha,
                          lw=1.8, label=f"O(h^{p_ord})")

        # Colour scatter by regime
        colours = ["#2ecc71" if kh < 0.3 else "#f39c12" if kh < 0.5
                   else "#e74c3c" for kh in kh_arr[mask]]
        for hv, ev, col in zip(h_arr[mask], l2_arr[mask], colours):
            ax.loglog(hv, ev, "o", color=col, ms=9, zorder=4)

        ax.set(xlabel="Average element size h [m]",
               ylabel="L2 relative error  ‖FEM−Mie‖/‖Mie‖",
               title="Convergence: L2 error vs mesh size")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which="both", ls=":")
        ax.text(0.97, 0.05,
                "●  kh<0.3  ●  kh<0.5  ●  kh≥0.5",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray")

        # Right: solution norm vs DOFs
        ax = axes[1]
        ax.semilogx(dof_arr, norm_arr, "rs-", lw=2.5, ms=9,
                    markerfacecolor="red", markeredgecolor="darkred",
                    markeredgewidth=1.5, label="‖Es‖ (L2 norm)")

        conv_pct = (abs(norm_arr[-1] - norm_arr[-2]) / norm_arr[-2] * 100
                    if len(norm_arr) >= 2 else float("nan"))
        ax.text(0.05, 0.95,
                f"Δ (finest vs 2nd finest): {conv_pct:.2f}%",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow",
                          edgecolor="orange", lw=1.5))
        ax.set(xlabel="Number of DOFs",
               ylabel="‖Es‖ (L2 norm)",
               title="Solution norm vs DOFs")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which="both", ls=":")

        plt.tight_layout()
        plt.savefig("convergence_results.png", dpi=200, bbox_inches="tight")
        print("Saved: convergence_results.png")

        # Save data
        np.savez("convergence_results.npz",
                 h=h_arr,
                 kh=kh_arr,
                 l2_err=l2_arr,
                 l2_norm=norm_arr,
                 dofs=dof_arr,
                 mesh_factors=np.array(MESH_FACTORS))
        print("Saved: convergence_results.npz")

        plt.show()