"""
010_main_solver.py — PEC Cylinder FEM Solver with Square Domain + ABC
======================================================================
Electromagnetic scattering from a PEC cylinder using FEniCSx.

Domain layout matches pml/010_main_solver.py exactly (square computational
domain) but uses a first-order Silver-Müller ABC on the outer square
boundary instead of PML — enabling direct ABC vs PML comparison.

Mesh layout (physical tags)
---------------------------
    1 : PEC cylinder surface  (edge)
    2 : Computational domain  (surface, background)
    3 : Outer square boundary (edge, ABC)

Usage
-----
Run directly:
    python 010_main_solver.py

Import into other scripts:
    import importlib.util, sys, pathlib
    ...
    MIE_PARAMS         = solver.MIE_PARAMS
    solve_pec_cylinder = solver.solve_pec_cylinder

Outputs
-------
    E_scattered_abc.bp    — scattered field
    E_total_abc.bp        — total field
    E_incident_abc.bp     — incident field
    fem_scattered_abc.npz — sampled field along x-axis (for 011_compare.py)
"""

import numpy as np
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io.gmsh import read_from_msh
from dolfinx.fem import (Function, functionspace, form,
                          dirichletbc, locate_dofs_topological,
                          assemble_scalar, Constant)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter
import ufl
from ufl import inner, Measure, FacetNormal, TrialFunction, TestFunction


# ── Shared physical parameters ────────────────────────────────────────────────
# Domain size matches pml/010_main_solver.py so ABC vs PML comparison is fair

MIE_PARAMS = dict(
    wl     = 1.0,   # Wavelength [m]
    R_cyl  = 0.5,   # Cylinder radius [m]
    l_dom  = 6.0,   # Square domain side length [m] — matches PML solver
    degree = 3,     # N1curl polynomial degree
)

# Physical group tags
_TAG_PEC  = 1   # PEC cylinder surface (edge)
_TAG_BKG  = 2   # Background computational domain (surface)
_TAG_ABC  = 3   # Outer square boundary (edge, ABC)


# ── UFL helpers ───────────────────────────────────────────────────────────────

def curl_2d(a):
    """2D curl returning a 3-vector (0, 0, dAy/dx - dAx/dy)."""
    return ufl.as_vector((0, 0, a[1].dx(0) - a[0].dx(1)))

def cross_z(a, b):
    """2D scalar cross product: a × b where b is a scalar (z-component)."""
    return a[0] * b[1] - a[1] * b[0]

def cross_xy(a, b):
    """Cross product: (ax, ay, 0) × (0, 0, b) = (ay*b, -ax*b, 0)."""
    return ufl.as_vector((a[1] * b, -a[0] * b, 0))


# ── Mesh generation ───────────────────────────────────────────────────────────

def create_mesh(R_cyl, l_dom, lc_cyl, lc_dom, filename="mesh.msh"):
    """
    Generate mesh: PEC cylinder inside square domain with ABC on outer edge.

    Physical tags
    -------------
    Edge   1 : PEC cylinder surface
    Surface 2 : Background computational domain
    Edge   3 : Outer square boundary (ABC)

    Parameters
    ----------
    R_cyl         : float — cylinder radius [m]
    l_dom         : float — square domain side length [m]
    lc_cyl        : float — mesh size at cylinder surface
    lc_dom        : float — mesh size in domain
    filename      : str   — output .msh path
    """
    gmsh.initialize()
    gmsh.model.add("pec_abc_square")

    # ── PEC cylinder ──────────────────────────────────────────────────────────
    cyl      = gmsh.model.occ.addCircle(0, 0, 0, R_cyl)
    cl       = gmsh.model.occ.addCurveLoop([cyl])
    cyl_surf = gmsh.model.occ.addPlaneSurface([cl])

    # ── Square computational domain ───────────────────────────────────────────
    dom_rect = gmsh.model.occ.addRectangle(
        -l_dom / 2, -l_dom / 2, 0, l_dom, l_dom
    )

    # Subtract cylinder from domain
    bkg_frags, _ = gmsh.model.occ.cut(
        [(2, dom_rect)], [(2, cyl_surf)], removeTool=False
    )

    gmsh.model.occ.synchronize()

    # ── Physical groups ───────────────────────────────────────────────────────
    # PEC cylinder edge
    gmsh.model.addPhysicalGroup(1, [cyl], tag=_TAG_PEC, name="PEC")

    # Background domain surfaces
    bkg_tags = [t for _, t in bkg_frags]
    gmsh.model.addPhysicalGroup(2, bkg_tags, tag=_TAG_BKG, name="Background")

    # Outer square ABC boundary — get curves of the square boundary
    # After cut, find boundary curves that are NOT the cylinder
    all_curves = [tag for _, tag in gmsh.model.getEntities(1)]
    outer_curves = [c for c in all_curves if c != cyl]
    gmsh.model.addPhysicalGroup(1, outer_curves, tag=_TAG_ABC, name="ABC")

    # ── Mesh sizing ───────────────────────────────────────────────────────────
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [cyl])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField",  1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin",  lc_cyl)
    gmsh.model.mesh.field.setNumber(2, "SizeMax",  lc_dom)
    gmsh.model.mesh.field.setNumber(2, "DistMin",  R_cyl)
    gmsh.model.mesh.field.setNumber(2, "DistMax",  l_dom / 2)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()

    return filename


# ── Core FEM solver ───────────────────────────────────────────────────────────

def solve_pec_cylinder(lc_cyl, lc_dom, params=None,
                        mesh_filename="mesh.msh"):
    """
    Solve PEC cylinder scattering with square domain + Silver-Müller ABC.

    Parameters
    ----------
    lc_cyl        : float — mesh size at cylinder surface
    lc_dom        : float — mesh size in computational domain
    params        : dict  — override any key in MIE_PARAMS
    mesh_filename : str   — .msh output path

    Returns
    -------
    Es_h      : dolfinx Function — scattered field (N1curl)
    Eb        : dolfinx Function — incident field   (N1curl)
    mesh      : dolfinx Mesh
    num_cells : int
    num_dofs  : int
    h_avg     : float — mean element diameter in background domain [m]
    """
    p      = {**MIE_PARAMS, **(params or {})}
    wl     = p["wl"]
    R_cyl  = p["R_cyl"]
    l_dom  = p["l_dom"]
    degree = p["degree"]
    k0     = 2 * np.pi / wl

    # ── Mesh ──────────────────────────────────────────────────────────────────
    if MPI.COMM_WORLD.rank == 0:
        create_mesh(R_cyl, l_dom, lc_cyl, lc_dom, mesh_filename)
    MPI.COMM_WORLD.barrier()

    msh, cell_tags, facet_tags, _, _, _ = read_from_msh(
        mesh_filename, MPI.COMM_WORLD, 0, gdim=2
    )

    num_cells = msh.topology.index_map(2).size_local
    import dolfinx.cpp as cpp
    h_cells = cpp.mesh.h(msh._cpp_object, 2,
                          np.arange(num_cells, dtype=np.int32))
    h_avg = float(np.mean(h_cells))

    # ── Function space ─────────────────────────────────────────────────────────
    V        = functionspace(msh, ("N1curl", degree))
    num_dofs = V.dofmap.index_map.size_global

    # ── Measures ───────────────────────────────────────────────────────────────
    dx   = Measure("dx", domain=msh, subdomain_data=cell_tags)
    ds   = Measure("ds", domain=msh, subdomain_data=facet_tags)
    dDom = dx(_TAG_BKG)
    dABC = ds(_TAG_ABC)
    n    = FacetNormal(msh)

    # ── Trial / test functions ──────────────────────────────────────────────────
    Es    = TrialFunction(V)
    v     = TestFunction(V)
    Es_3d = ufl.as_vector((Es[0], Es[1], 0))
    v_3d  = ufl.as_vector((v[0],  v[1],  0))

    # ── Incident field  Ei = ŷ exp(−ik₀x) ─────────────────────────────────────
    Eb = Function(V)
    Eb.interpolate(lambda x: (
        np.zeros(x.shape[1], dtype=np.complex128),
        np.exp(-1j * k0 * x[0]),
    ))

    # ── PEC Dirichlet BC:  Es|_PEC = −Ei|_PEC ─────────────────────────────────
    tdim       = msh.topology.dim
    pec_facets = facet_tags.find(_TAG_PEC)
    pec_dofs   = locate_dofs_topological(V, tdim - 1, pec_facets)

    E_pec = Function(V)
    E_pec.interpolate(lambda x: (
        np.zeros(x.shape[1], dtype=np.complex128),
        -np.exp(-1j * k0 * x[0]),
    ))
    bc = dirichletbc(E_pec, dofs=pec_dofs)

    # ── Weak form ──────────────────────────────────────────────────────────────
    # Domain: homogeneous scattered field equation (source via Dirichlet BC)
    # ABC:    first-order Silver-Müller on outer square boundary
    #         ∇×Es × n + ik₀(n × Es) × n = 0
    a_ufl = (
        -inner(curl_2d(Es), curl_2d(v)) * dDom
        + k0**2 * inner(Es, v) * dDom
        + 1j * k0 * inner(
            cross_xy(n, cross_z(n, Es_3d)),
            cross_xy(n, cross_z(n, v_3d))
        ) * dABC
    )
    L_ufl = inner(Constant(msh, PETSc.ScalarType(0.0 + 0j)), v[0]) * dDom

    # ── Solve ──────────────────────────────────────────────────────────────────
    problem = LinearProblem(
        a_ufl, L_ufl, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type":  "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="pec_abc_",
    )
    Es_h = problem.solve()
    Es_h.name = "Scattered_Field"

    return Es_h, Eb, msh, num_cells, num_dofs, h_avg


# ── Stand-alone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    p    = MIE_PARAMS
    wl   = p["wl"]
    k0   = 2 * np.pi / wl

    lc_cyl = wl / 30
    lc_dom = wl / 15   # tightened from wl/10 to match PML solver cell density

    if comm.rank == 0:
        print("=" * 60)
        print("PEC CYLINDER SCATTERING SOLVER  (Square Domain + ABC)")
        print("=" * 60)
        print(f"  λ      = {wl} m  |  k₀ = {k0:.4f} rad/m")
        print(f"  R_cyl  = {p['R_cyl']} m")
        print(f"  l_dom  = {p['l_dom']} m  (square, matches PML solver)")
        print(f"  lc_cyl = {lc_cyl:.4f} m  |  lc_dom = {lc_dom:.4f} m")
        print()

    Es_h, Eb, msh, num_cells, num_dofs, h_avg = solve_pec_cylinder(
        lc_cyl, lc_dom
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    if comm.rank == 0:
        from ufl import inner, dx
        Es_energy = assemble_scalar(form(inner(Es_h, Es_h) * dx))
        Ei_energy = assemble_scalar(form(inner(Eb,   Eb)   * dx))
        Es_norm   = float(np.sqrt(abs(Es_energy)))
        Ei_norm   = float(np.sqrt(abs(Ei_energy)))

        print(f"  Cells    : {num_cells}  |  DOFs : {num_dofs}")
        print(f"  h_avg    : {h_avg:.5f} m  |  kh = {k0*h_avg:.4f}")
        print(f"  ‖Es‖     : {Es_norm:.4e}  |  ‖Ei‖ : {Ei_norm:.4e}")
        print(f"  ‖Es‖/‖Ei‖: {Es_norm/Ei_norm:.4f}")
        print()

    # ── Interpolate to CG for visualisation ───────────────────────────────────
    degree  = p["degree"]
    V_cg    = functionspace(msh, ("Lagrange", degree, (msh.geometry.dim,)))

    E_total = Function(Es_h.function_space)
    E_total.x.array[:] = Eb.x.array[:] + Es_h.x.array[:]

    Es_cg   = Function(V_cg); Es_cg.interpolate(Es_h)
    Etot_cg = Function(V_cg); Etot_cg.interpolate(E_total)
    Ei_cg   = Function(V_cg); Ei_cg.interpolate(Eb)

    with VTXWriter(msh.comm, "E_scattered.bp", [Es_cg])   as f: f.write(0.0)
    with VTXWriter(msh.comm, "E_total.bp",     [Etot_cg]) as f: f.write(0.0)
    with VTXWriter(msh.comm, "E_incident.bp",  [Ei_cg])   as f: f.write(0.0)

    if comm.rank == 0:
        print("  Saved: E_scattered.bp, E_total.bp, E_incident.bp")

    # ── Export scattered field along x-axis for 011_compare.py ───────────────
    from dolfinx.geometry import (bb_tree, compute_collisions_points,
                                   compute_colliding_cells)
    R_cyl = p["R_cyl"]
    l_dom = p["l_dom"]
    x_line = np.linspace(R_cyl * 1.05, (l_dom / 2) * 0.95, 300)
    pts_3d = np.column_stack([x_line, np.zeros_like(x_line),
                               np.zeros_like(x_line)])

    tree      = bb_tree(msh, msh.topology.dim)
    cands     = compute_collisions_points(tree, pts_3d)
    cells_out = compute_colliding_cells(msh, cands, pts_3d)

    Ex_line = np.zeros(len(x_line), dtype=np.complex128)
    Ey_line = np.zeros(len(x_line), dtype=np.complex128)
    for i in range(len(x_line)):
        cl = cells_out.links(i)
        if len(cl) > 0:
            val        = Es_cg.eval(pts_3d[i:i+1], [cl[0]])
            Ex_line[i] = val[0]
            Ey_line[i] = val[1]

    np.savez("fem_scattered_abc.npz",
             x=x_line,
             Ex=Ex_line,
             Ey=Ey_line,
             E_mag=np.sqrt(np.abs(Ex_line)**2 + np.abs(Ey_line)**2))

    if comm.rank == 0:
        print("  Saved: fem_scattered_abc.npz  (for 011_compare.py)")
        print()
        print("  Next: python 011_compare.py")