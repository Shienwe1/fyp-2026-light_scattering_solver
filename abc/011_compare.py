"""
02_mie_to_bp.py — Mie Analytical Solution → ParaView Export
=============================================================
Evaluates the 2D Mie scattered field for a PEC cylinder at every
node of the FEM mesh and exports it to E_mie_scattered.bp for
direct side-by-side comparison with E_scattered.bp in ParaView.

Requirements
------------
Run 010_main_solver.py first to generate mesh.msh.

Usage
-----
    python 02_mie_to_bp.py

Outputs
-------
    E_mie_abc.bp   — Mie scattered field on ABC square domain mesh
"""

import numpy as np
from scipy.special import hankel2, jv
from mpi4py import MPI
from dolfinx.io.gmsh import read_from_msh
from dolfinx.fem import functionspace, Function
from dolfinx.io import VTXWriter


# ── Parameters ────────────────────────────────────────────────────────────────

wl    = 1.0
R_cyl = 0.5
k0    = 2 * np.pi / wl


# ── Bessel / Hankel derivative helpers ───────────────────────────────────────

def _jv_prime(n, z):
    return 0.5 * (jv(n - 1, z) - jv(n + 1, z))

def _h2_prime(n, z):
    return 0.5 * (hankel2(n - 1, z) - hankel2(n + 1, z))


# ── Mie coefficients ──────────────────────────────────────────────────────────

n_max = 20
coefficients = {
    n: -_jv_prime(n, k0 * R_cyl) / _h2_prime(n, k0 * R_cyl)
    for n in range(-n_max, n_max + 1)
}


# ── Vectorised Mie field evaluation ──────────────────────────────────────────

def compute_mie_field(x_pts, y_pts):
    """Evaluate Mie scattered field at arbitrary (x, y) arrays."""
    x_pts = np.asarray(x_pts, dtype=np.float64)
    y_pts = np.asarray(y_pts, dtype=np.float64)

    rho = np.sqrt(x_pts**2 + y_pts**2)
    phi = np.arctan2(y_pts, x_pts)
    rho = np.where(rho < 1e-14, 1e-14, rho)

    E_rho = np.zeros(len(x_pts), dtype=np.complex128)
    E_phi = np.zeros(len(x_pts), dtype=np.complex128)

    for n, a_n in coefficients.items():
        if abs(a_n) < 1e-15:
            continue
        H_n  = hankel2(n, k0 * rho)
        Hp_n = _h2_prime(n, k0 * rho)
        phase = ((-1j)**n) * np.exp(1j * n * phi)

        if n != 0:
            E_rho += a_n * (n / (k0 * rho)) * H_n * phase
        E_phi += a_n * 1j * Hp_n * phase

    Ex = E_rho * np.cos(phi) - E_phi * np.sin(phi)
    Ey = E_rho * np.sin(phi) + E_phi * np.cos(phi)
    return Ex, Ey


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("MIE → PARAVIEW EXPORT")
    print("=" * 60)
    print(f"  λ = {wl} m  |  k₀ = {k0:.4f} rad/m  |  R_cyl = {R_cyl} m")
    print()
    print("  Reading mesh.msh ...")

    msh, _, _, _, _, _ = read_from_msh(
        "mesh.msh", MPI.COMM_WORLD, 0, gdim=2
    )

    x_coords = msh.geometry.x[:, 0]
    y_coords = msh.geometry.x[:, 1]

    print(f"  Evaluating Mie at {len(x_coords)} mesh nodes ...")

    # Mirror x-coordinates to flip wave direction and match ABC solver convention
    # 1. Evaluate field using mirrored x to flip wave propagation direction
    Ex_orig, Ey_orig = compute_mie_field(-x_coords, y_coords)

    # 2. Invert Ex component to match the mirrored coordinate space
    Ex_mie = -Ex_orig
    Ey_mie = Ey_orig

    # Degree-1 Lagrange: DOF = geometry node → 1-to-1 array assignment
    V_cg = functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    E_mie = Function(V_cg, dtype=np.complex128)
    E_mie.name = "E_mie"
    E_mie.x.array[0::2] = Ex_mie
    E_mie.x.array[1::2] = Ey_mie

    with VTXWriter(msh.comm, "E_mie_abc.bp", [E_mie]) as f:
        f.write(0.0)

    print("  ✓ Saved: E_mie_abc.bp")
    print()
    print("  ParaView workflow:")
    print("  1. Load E_scattered_abc.bp   — FEM scattered field (ABC)")
    print("  2. Load E_mie_abc.bp         — Mie analytical field (ABC mesh)")
    print("  3. On each: Filters → Calculator →")
    print("       sqrt(f_real_0^2 + f_real_1^2)")
    print("  4. Split screen, identical colour map range")
    print("  5. Screenshot for validation figure")