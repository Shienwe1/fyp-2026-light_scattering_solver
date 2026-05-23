"""
024_export_field.py — Phase 3: Electric Field Export for ParaView
=================================================================
Performs a single, high-fidelity FEM solve at the exact LSPR peak
wavelength and exports the total electric field (E_total) to an
ADIOS2 (.bp) directory for high-resolution heatmap rendering in
ParaView.

Two exports are produced:
    E_field_water.bp   — reference state (water shell, n=1.33)
    E_field_protein.bp — bound state    (protein shell, n=1.45)

Opening both side-by-side in ParaView with identical colour map
ranges shows the near-field enhancement difference caused by
protein binding.

Usage
-----
    mpirun -n 4 python 024_export_field.py

Outputs
-------
    E_field_water.bp    — total E field, water reference
    E_field_protein.bp  — total E field, protein bound state
"""

import numpy as np
from mpi4py import MPI
from dolfinx.io import VTXWriter
from dolfinx.fem import functionspace, Function
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


# ── Helper: interpolate N1curl field to CG for VTX export ────────────────────

def export_field(ctx, wl, n_shell, filename):
    """
    Run a single solve and export E_total to a .bp file.

    Parameters
    ----------
    ctx      : context dict from build_biosensor_mesh
    wl       : float — wavelength [µm]
    n_shell  : float — shell refractive index (for labelling only)
    filename : str   — output .bp path
    """
    comm = ctx["msh"].comm
    msh  = ctx["msh"]

    if comm.rank == 0:
        print(f"  Solving at λ = {wl*1e3:.1f} nm  (n_shell = {n_shell}) ...")

    sweep_wavelengths(ctx, np.array([wl]), verbose=False)

    # Interpolate to CG for VTX — N1curl fields cannot be written directly
    degree = ctx["p"]["degree"]
    V_cg   = functionspace(msh, ("Lagrange", degree, (msh.geometry.dim,)))

    E_tot_cg = Function(V_cg)
    E_tot_cg.name = "E_total"
    E_tot_cg.interpolate(ctx["E_total"])

    with VTXWriter(comm, filename, [E_tot_cg], engine="BP4") as vtx:
        vtx.write(0.0)

    if comm.rank == 0:
        print(f"  ✓ Saved: {filename}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print("=" * 70)
        print("PHASE 3 — ELECTRIC FIELD EXPORT FOR PARAVIEW")
        print("=" * 70)
        print()

    # Target wavelength — LSPR peak identified in Phase 2
    target_wl = 0.35817   # 358.17 nm

    # Ultra-dense near-field mesh for a publication-quality heatmap
    p = {
        **BIOSENSOR_PARAMS,
        "wl_start" : target_wl,
        "wl_end"   : target_wl,
        "wl_points": 1,
        "lc_au"    : 0.002,   # 2 nm elements inside gold
        "lc_shell" : 0.002,   # 2 nm elements in shell
    }

    # ── Export 1: Water reference state ───────────────────────────────────────
    if comm.rank == 0:
        print("─" * 70)
        print("Export 1/2 — Water reference (n_shell = 1.33)")
        print("─" * 70)

    ctx_water = build_biosensor_mesh(
        n_shell=1.33, params=p,
        mesh_filename="mesh_heatmap_water.msh"
    )
    export_field(ctx_water, target_wl, n_shell=1.33,
                 filename="E_field_water.bp")

    # ── Export 2: Protein bound state ─────────────────────────────────────────
    if comm.rank == 0:
        print()
        print("─" * 70)
        print("Export 2/2 — Protein bound state (n_shell = 1.45)")
        print("─" * 70)

    ctx_protein = build_biosensor_mesh(
        n_shell=1.45, params=p,
        mesh_filename="mesh_heatmap_protein.msh"
    )
    export_field(ctx_protein, target_wl, n_shell=1.45,
                 filename="E_field_protein.bp")

    # ── ParaView instructions ─────────────────────────────────────────────────
    if comm.rank == 0:
        print()
        print("=" * 70)
        print("PARAVIEW WORKFLOW")
        print("=" * 70)
        print()
        print("  1. Open ParaView")
        print("  2. File → Open → E_field_water.bp   (open as ADIOS2 BP4)")
        print("  3. File → Open → E_field_protein.bp")
        print("  4. On each: Apply → Filters → Calculator:")
        print("       Result name : E_magnitude")
        print("       Expression  : sqrt(E_total_0^2 + E_total_1^2)")
        print("  5. Split screen (View → Split Horizontal)")
        print("  6. Set identical colour map range on both panels:")
        print("       Rescale → Custom Range → use same min/max")
        print("  7. Screenshot both panels for thesis figures")
        print()
        print("  Tip: Use 'Rescale to Data Range' on the protein panel first,")
        print("  then manually apply the same range to the water panel.")
        print("  This shows the absolute field enhancement from protein binding.")