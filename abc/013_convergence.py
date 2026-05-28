"""
Mesh Convergence Study — PEC Cylinder Scattering
=================================================
Validates solver accuracy by comparing FEM solutions at progressively
refined mesh levels against the Mie series analytical reference.

Convergence pipeline:
    01_pec_cylinder.py       → produces fem_scattered.npz (run first)
    02_math_analytical.py    → Mie reference (formulas reused here)
    03_compare_fem_mie.py    → single-mesh validation
    04_convergence_study.py  → this script, multi-mesh convergence

Expected result:
    Log-log plot of L2 error vs h should show slope ≈ 3
    (theoretical rate for degree-3 N1curl elements)

Usage:
    python 04_convergence_study.py
"""

import numpy as np
import gmsh
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from scipy.special import jv, hankel2
from dolfinx.io.gmsh import read_from_msh
from dolfinx.fem import (Function, functionspace, Constant,
                         dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import (bb_tree, compute_collisions_points,
                               compute_colliding_cells)
import ufl
from ufl import inner, curl, Measure, FacetNormal, TrialFunction, TestFunction


# ============================================================================
# PHYSICAL PARAMETERS — must match 01_pec_cylinder.py exactly
# ============================================================================

WL    = 1.0                  # Wavelength [m]
K0    = 2 * np.pi / WL       # Wavenumber [rad/m]
R_CYL = 0.5 * WL             # Cylinder radius [m]
R_OUT = 15.0 * WL             # Outer boundary radius [m]
DEG   = 3                    # N1curl polynomial degree

# Comparison line — same as 03_compare_fem_mie.py
X_COMPARE = np.linspace(R_CYL * 1.05, R_OUT * 0.95, 1000)
Y_COMPARE = np.zeros_like(X_COMPARE)


# ============================================================================
# MESH REFINEMENT LEVELS
# Adjust mesh_factors if memory or time is limited.
# Rule: lc_cyl = factor * WL/15
# Coarser (factor closer to 1.0) = fewer DOFs = faster but less accurate
# Finer  (factor closer to 0.2)  = more DOFs  = slower but more accurate
# ============================================================================

MESH_FACTORS = [4.0, 3.0, 2.0, 1.0, 0.5]


# ============================================================================
# HELPER FUNCTIONS — UFL cross products (identical to 01_pec_cylinder.py)
# ============================================================================

def cross_z(a, b):
    return a[0]*b[1] - a[1]*b[0]

def cross_xy(a, b):
    return ufl.as_vector((a[1]*b, -a[0]*b, 0))


# ============================================================================
# HELPER FUNCTIONS — Mie series (identical to 03_compare_fem_mie.py)
# ============================================================================

def jv_prime(n, z):
    """Derivative of Bessel function J_n via recurrence."""
    return 0.5 * (jv(n-1, z) - jv(n+1, z))

def h2_prime(n, z):
    """Derivative of Hankel function H_n^(2) via recurrence."""
    return 0.5 * (hankel2(n-1, z) - hankel2(n+1, z))

def compute_mie_field(x_pts, y_pts, k0, R_cyl, n_max=20):
    """
    Compute Mie series scattered field magnitude along a set of points.

    Uses derivative-form coefficients and (-j)^n plane wave phase factor
    to match the FEM incident field e^{-ik0x}.

    Parameters
    ----------
    x_pts, y_pts : array_like
        Coordinates of evaluation points.
    k0 : float
        Wavenumber [rad/m].
    R_cyl : float
        Cylinder radius [m].
    n_max : int
        Series truncation order.

    Returns
    -------
    E_mag : ndarray
        Field magnitude |E| at each point.
    """
    # Mie coefficients — derivative form for PEC boundary condition
    coeffs = {}
    for n in range(-n_max, n_max + 1):
        coeffs[n] = -jv_prime(n, k0*R_cyl) / h2_prime(n, k0*R_cyl)

    Ex = np.zeros(len(x_pts), dtype=np.complex128)
    Ey = np.zeros(len(x_pts), dtype=np.complex128)

    for i, (x, y) in enumerate(zip(x_pts, y_pts)):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        E_rho = 0j
        E_phi = 0j

        for n, a_n in coeffs.items():
            if abs(a_n) < 1e-15:
                continue
            H_n  = hankel2(n, k0 * rho)
            Hp_n = h2_prime(n, k0 * rho)
            # (-j)^n phase factor matches e^{-ik0x} incident field
            phase = ((-1j)**n) * np.exp(1j * n * phi)
            if n != 0:
                E_rho += a_n * (1j * n / (k0 * rho)) * H_n * phase
            E_phi += a_n * Hp_n * phase

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        Ex[i] = E_rho * cos_phi - E_phi * sin_phi
        Ey[i] = E_rho * sin_phi + E_phi * cos_phi

    return np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)


# ============================================================================
# MESH GENERATION — identical to 01_pec_cylinder.py
# ============================================================================

def create_mesh(R_cyl, R_out, lc_cyl, lc_out, filename="mesh.msh"):
    gmsh.initialize()
    gmsh.model.add("pec")

    cyl = gmsh.model.occ.addCircle(0, 0, 0, R_cyl)
    out = gmsh.model.occ.addCircle(0, 0, 0, R_out)

    cyl_loop = gmsh.model.occ.addCurveLoop([cyl])
    out_loop = gmsh.model.occ.addCurveLoop([out])
    surf     = gmsh.model.occ.addPlaneSurface([out_loop, cyl_loop])

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, [out],  tag=1, name="ABC")
    gmsh.model.addPhysicalGroup(1, [cyl],  tag=2, name="PEC")
    gmsh.model.addPhysicalGroup(2, [surf], tag=1, name="Domain")

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_out)

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [cyl])
    gmsh.model.mesh.field.setNumber(1,  "Sampling",   100)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField",  1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin",  lc_cyl)
    gmsh.model.mesh.field.setNumber(2, "SizeMax",  lc_out)
    gmsh.model.mesh.field.setNumber(2, "DistMin",  R_cyl)
    gmsh.model.mesh.field.setNumber(2, "DistMax",  R_out)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()

    return filename


# ============================================================================
# FEM SOLVER — weak form and solver identical to 01_pec_cylinder.py
# ============================================================================

def solve_and_sample(mesh_factor):
    """
    Solve PEC scattering at one mesh refinement level and sample
    the scattered field along the comparison line.

    Parameters
    ----------
    mesh_factor : float
        Controls mesh coarseness. Smaller = finer mesh.
        lc_cyl = mesh_factor * WL/15
        lc_out = mesh_factor * WL/5

    Returns
    -------
    h_avg : float
        Average element size [m].
    num_dofs : int
        Total degrees of freedom.
    E_mag_fem : ndarray
        Scattered field magnitude along X_COMPARE.
    """
    lc_cyl = mesh_factor * WL / 15
    lc_out = mesh_factor * WL / 5

    mesh_file = create_mesh(R_CYL, R_OUT, lc_cyl, lc_out,
                            filename=f"mesh_f{mesh_factor:.2f}.msh")
    mesh, cell_tags, facet_tags, _, _, _ = read_from_msh(
        mesh_file, MPI.COMM_WORLD, 0, gdim=2)

    # Average element size
    import dolfinx.cpp as cpp
    num_cells = mesh.topology.index_map(2).size_local
    h_cells   = cpp.mesh.h(mesh._cpp_object, 2,
                           np.arange(num_cells, dtype=np.int32))
    h_avg     = float(np.mean(h_cells))

    # Function space
    V       = functionspace(mesh, ("N1curl", DEG))
    num_dofs = V.dofmap.index_map.size_global

    # Measures
    dx_m = Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds_m = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    n    = FacetNormal(mesh)

    # Trial / test functions
    Es  = TrialFunction(V)
    v   = TestFunction(V)
    Es_3d = ufl.as_vector((Es[0], Es[1], 0))
    v_3d  = ufl.as_vector((v[0],  v[1],  0))

    # Incident field — e^{-ik0x}, y-polarised
    Eb = Function(V)
    Eb.interpolate(lambda x: (
        np.zeros(x.shape[1], dtype=np.complex128),
        np.exp(-1j * K0 * x[0])))

    # PEC Dirichlet BC: Es = -Ei on cylinder surface
    tdim       = mesh.topology.dim
    pec_facets = facet_tags.find(2)
    pec_dofs   = locate_dofs_topological(V, tdim-1, pec_facets)

    E_pec = Function(V)
    E_pec.interpolate(lambda x: (
        np.zeros(x.shape[1], dtype=np.complex128),
        -np.exp(-1j * K0 * x[0])))
    bc = dirichletbc(E_pec, dofs=pec_dofs)

    # Bilinear form — IDENTICAL sign convention to 01_pec_cylinder.py
    zero_c = Constant(mesh, PETSc.ScalarType(0))
    L_ufl  = inner(zero_c, v[0]) * dx_m
    a_ufl  = (
        inner(curl(Es), curl(v)) * dx_m
        - K0**2 * inner(Es, v) * dx_m
        + 1j * K0 * inner(cross_xy(n, cross_z(n, Es_3d)),
                          cross_xy(n, cross_z(n, v_3d))) * ds_m(1)
    )

    # Solver — MUMPS direct LU, identical to 01_pec_cylinder.py
    problem = LinearProblem(
    a_ufl, L_ufl, bcs=[bc],
    petsc_options={"ksp_type":  "preonly",
                   "pc_type":   "lu",
                   "pc_factor_mat_solver_type": "mumps"},
    petsc_options_prefix=f"pec_level{level}_")
    Es_h = problem.solve()

    # Interpolate to CG for clean point evaluation
    V_cg  = functionspace(mesh, ("Lagrange", DEG, (mesh.geometry.dim,)))
    Es_cg = Function(V_cg)
    Es_cg.interpolate(Es_h)

    # Sample along comparison line
    pts_3d     = np.column_stack([X_COMPARE, Y_COMPARE,
                                  np.zeros_like(X_COMPARE)])
    tree       = bb_tree(mesh, mesh.topology.dim)
    candidates = compute_collisions_points(tree, pts_3d)
    cells      = compute_colliding_cells(mesh, candidates, pts_3d)

    Ex_line = np.zeros(len(X_COMPARE), dtype=np.complex128)
    Ey_line = np.zeros(len(X_COMPARE), dtype=np.complex128)

    for i in range(len(X_COMPARE)):
        cell_list = cells.links(i)
        if len(cell_list) > 0:
            val = Es_cg.eval(pts_3d[i:i+1], [cell_list[0]])
            Ex_line[i] = val[0]
            Ey_line[i] = val[1]

    E_mag_fem = np.sqrt(np.abs(Ex_line)**2 + np.abs(Ey_line)**2)

    return h_avg, num_dofs, E_mag_fem


# ============================================================================
# MAIN — convergence loop
# ============================================================================

if __name__ == "__main__":

    if MPI.COMM_WORLD.rank != 0:
        # Non-root ranks participate in solving but skip analysis/plotting
        for mf in MESH_FACTORS:
            solve_and_sample(mf)
        raise SystemExit

    print("=" * 65)
    print("MESH CONVERGENCE STUDY — PEC Cylinder Scattering")
    print("=" * 65)
    print(f"  Wavenumber k0  : {K0:.4f} rad/m")
    print(f"  Cylinder radius: {R_CYL} m  (ka = {K0*R_CYL:.4f})")
    print(f"  Outer radius   : {R_OUT} m")
    print(f"  Element degree : {DEG}")
    print(f"  Mesh levels    : {len(MESH_FACTORS)}")

    # Pre-compute Mie reference — same for all mesh levels
    print("\nComputing Mie reference field...")
    E_mag_mie = compute_mie_field(X_COMPARE, Y_COMPARE, K0, R_CYL)
    mie_norm  = np.sqrt(np.trapz(E_mag_mie**2, X_COMPARE))
    print(f"  Mie |E| range: {E_mag_mie.min():.4e} — {E_mag_mie.max():.4e}")

    # Storage
    h_list    = []
    dof_list  = []
    l2_errors = []

    # Convergence loop
    for level, mf in enumerate(MESH_FACTORS):
        print(f"\nLevel {level+1}/{len(MESH_FACTORS)}  "
              f"(mesh_factor = {mf})")

        h_avg, num_dofs, E_mag_fem = solve_and_sample(mf)

        # True L2 error vs Mie analytical solution
        l2_num = np.sqrt(np.trapz((E_mag_fem - E_mag_mie)**2, X_COMPARE))
        l2_err = l2_num / mie_norm

        h_list.append(h_avg)
        dof_list.append(num_dofs)
        l2_errors.append(l2_err)

        kh = K0 * h_avg
        print(f"  h_avg   = {h_avg:.5f} m  (kh = {kh:.3f})")
        print(f"  DOFs    = {num_dofs}")
        print(f"  L2 error vs Mie = {l2_err*100:.3f}%")

    # ========================================================================
    # CONVERGENCE RATE ANALYSIS
    # ========================================================================

    print("\n" + "=" * 65)
    print("CONVERGENCE RATE ANALYSIS")
    print("=" * 65)

    print(f"\n{'Level':<6} {'h_avg':>10} {'kh':>8} {'DOFs':>8} "
          f"{'L2 error':>12} {'Rate':>8}")
    print("-" * 60)

    rates = []
    for i in range(len(MESH_FACTORS)):
        kh = K0 * h_list[i]
        if i == 0:
            rate_str = "   —"
        else:
            rate = (np.log(l2_errors[i-1]) - np.log(l2_errors[i])) / \
                   (np.log(h_list[i-1])    - np.log(h_list[i]))
            rates.append(rate)
            rate_str = f"{rate:6.2f}"

        print(f"{i+1:<6} {h_list[i]:>10.5f} {kh:>8.3f} "
              f"{dof_list[i]:>8} {l2_errors[i]*100:>11.3f}% {rate_str:>8}")

    if rates:
        avg_rate = np.mean(rates)
        print(f"\n  Average convergence rate: {avg_rate:.2f}")
        print(f"  Expected (degree {DEG}):       ~{DEG:.1f}")

        if avg_rate >= 2.5:
            print("  Result: EXCELLENT — matches theoretical order")
        elif avg_rate >= 2.0:
            print("  Result: GOOD — correct convergence behaviour")
        elif avg_rate >= 1.0:
            print("  Result: SLOW — check kh values and ABC quality")
        else:
            print("  Result: NOT CONVERGING — check weak form and BCs")

    # ========================================================================
    # PLOTS
    # ========================================================================

    h_arr  = np.array(h_list)
    err_arr = np.array(l2_errors)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Mesh Convergence Study — PEC Cylinder  ka = {K0*R_CYL:.3f}",
        fontsize=13, fontweight='bold')

    # Plot 1: L2 error vs h (log-log) — the main convergence plot
    ax = axes[0]
    ax.loglog(h_arr, err_arr * 100, 'bo-', lw=2, ms=8,
              label='L2 error vs Mie')

    # Reference slope lines for order 2 and order 3
    h_ref = np.array([h_arr[0], h_arr[-1]])
    for p, ls, lbl in [(2, '--', 'O(h²)'), (3, '-.', 'O(h³)')]:
        C = (err_arr[0] * 100) / h_arr[0]**p
        ax.loglog(h_ref, C * h_ref**p, ls, lw=1.5, alpha=0.6, label=lbl)

    ax.set_xlabel('Average element size h [m]', fontsize=12)
    ax.set_ylabel('Relative L2 error [%]',      fontsize=12)
    ax.set_title('Error vs mesh size',            fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Plot 2: L2 error vs DOFs (log-log) — cost vs accuracy
    ax = axes[1]
    ax.loglog(dof_list, err_arr * 100, 'rs-', lw=2, ms=8,
              label='L2 error vs Mie')

    # O(1/N) reference line — in 2D: h ~ 1/sqrt(N), Error ~ h^2 ~ 1/N
    dof_arr = np.array(dof_list)
    dof_ref = np.array([dof_arr[0], dof_arr[-1]])
    C_dof   = (err_arr[0] * 100) * dof_arr[0]
    ax.loglog(dof_ref, C_dof / dof_ref, 'k--', lw=1.5, alpha=0.6,
              label='O(1/N) reference')

    ax.set_xlabel('Number of DOFs',             fontsize=12)
    ax.set_ylabel('Relative L2 error [%]',      fontsize=12)
    ax.set_title('Error vs degrees of freedom', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    final_err = err_arr[-1] * 100
    ax.text(0.05, 0.95,
            f"Finest Mesh Error: {final_err:.3f}%",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow",
                      edgecolor="orange", lw=1.5))
    
    plt.tight_layout()
    plt.savefig('convergence_study.png', dpi=180, bbox_inches='tight')
    print("\nSaved: convergence_study.png")

    # Save numerical results
    np.savez('convergence_results.npz',
             h=h_arr,
             dofs=np.array(dof_list),
             l2_errors=err_arr,
             rates=np.array(rates) if rates else np.array([]),
             k0=K0, R_cyl=R_CYL, degree=DEG)
    print("Saved: convergence_results.npz")

    plt.show()