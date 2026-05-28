"""
010_main_solver.py — PEC Cylinder FEM Solver with PML (Importable Core)
========================================================================
Electromagnetic scattering from a PEC cylinder using FEniCSx.

Boundary condition upgrade: ABC → Rectangular PML
--------------------------------------------------
The first-order Silver-Müller ABC is replaced with a rectangular
Perfectly Matched Layer (PML) following the demo_pml.py formulation
(Castriotta, Baratta, Dokken — FEniCSx team).

PML approach
------------
- Square computational domain surrounded by rectangular PML strips
  and corner PML squares
- Complex coordinate stretching:
    x' = x + (j*alpha/k0) * x*(|x| - l_dom/2) / (l_pml/2 - l_dom/2)^2
- PML regions treated as anisotropic material via Jacobian transformation:
    eps_pml = det(J) * A^-1 * eps_bkg * A^-T
    mu_pml  = det(J) * A^-1 * mu_bkg  * A^-T
- No boundary condition needed on the outer PML edge (field decays to ~0)

Mesh layout (physical tags)
---------------------------
    1 : PEC cylinder surface  (edge)
    2 : Computational domain  (surface, background)
    3 : PML corners           (surface)
    4 : PML strips along x    (surface)
    5 : PML strips along y    (surface)

Usage
-----
Run directly:
    python 010_main_solver.py

Import into other scripts:
    import importlib.util, sys, pathlib
    ...
    MIE_PARAMS         = solver.MIE_PARAMS
    solve_pec_cylinder = solver.solve_pec_cylinder
"""

import numpy as np
import gmsh
from functools import reduce
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io.gmsh import read_from_msh
from dolfinx.fem import (Function, functionspace, form,
                          dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter
from dolfinx.fem import assemble_scalar
import ufl
from ufl import inner, Measure, FacetNormal, TrialFunction, TestFunction


# ── Shared physical parameters ────────────────────────────────────────────────

MIE_PARAMS = dict(
    wl     = 1.0,   # Wavelength [m]
    R_cyl  = 0.5,   # Cylinder radius [m]
    l_dom  = 6.0,   # Side length of square computational domain [m]
    l_pml  = 8.0,   # Side length including PML layer — 1.0λ thick on each side
    alpha  = 15.0,  # PML absorption strength — reflection ~4.5e-5
    degree = 3,     # N1curl polynomial degree
)

# Physical group tags
_TAG_PEC     = 1
_TAG_BKG     = 2
_TAG_PML_XY  = 3   # corners
_TAG_PML_X   = 4   # strips along x
_TAG_PML_Y   = 5   # strips along y


# ── UFL helpers ───────────────────────────────────────────────────────────────

def curl_2d(a):
    """2D curl returning a 3-vector (0, 0, dAy/dx - dAx/dy)."""
    return ufl.as_vector((0, 0, a[1].dx(0) - a[0].dx(1)))


def pml_coordinates(x_coord, alpha, k0, l_dom, l_pml):
    """
    Complex coordinate stretching for PML region.
    Strictly avoids Abs() to guarantee perfect UFL Jacobian differentiation.
    Sign: -1j matches e^(-jk0x) convention so PML absorbs rather than amplifies.
    """
    sign_x = ufl.conditional(x_coord > 0, 1.0, -1.0)
    d_x    = sign_x * x_coord - l_dom / 2
    L_p    = (l_pml - l_dom) / 2
    return x_coord - sign_x * 1j * (alpha / k0) * (d_x**3) / (3 * L_p**2)


def create_eps_mu(pml_vec, eps_bkg, mu_bkg):
    """
    Compute PML effective permittivity and permeability tensors
    from the Jacobian of the complex coordinate transformation.

    Based on Ward & Pendry (1996):
        eps_pml = det(J) * A^-1 * eps_bkg * A^-T
        mu_pml  = det(J) * A^-1 * mu_bkg  * A^-T
    where A = J^-1 = grad(pml_vec).
    Uses ufl.dot for strict tensor matrix multiplication.
    """
    J = ufl.grad(pml_vec)
    J = ufl.as_matrix(((J[0, 0], 0, 0),
                       (0, J[1, 1], 0),
                       (0, 0,       1)))
    A       = ufl.inv(J)
    eps_pml = ufl.det(J) * ufl.dot(A, ufl.dot(eps_bkg, ufl.transpose(A)))
    mu_pml  = ufl.det(J) * ufl.dot(A, ufl.dot(mu_bkg,  ufl.transpose(A)))
    return eps_pml, mu_pml


# ── Mesh generation ───────────────────────────────────────────────────────────

def create_mesh(R_cyl, l_dom, l_pml, lc_cyl, lc_dom, lc_pml,
                filename="mesh.msh"):
    """
    Generate mesh: PEC cylinder inside square domain + rectangular PML.

    Physical tags
    -------------
    Edge   1 : PEC cylinder surface
    Surface 2 : Background computational domain
    Surface 3 : PML corners
    Surface 4 : PML strips along x
    Surface 5 : PML strips along y

    Parameters
    ----------
    R_cyl            : float — cylinder radius [m]
    l_dom            : float — square domain side length [m]
    l_pml            : float — total side length including PML [m]
    lc_cyl, lc_dom, lc_pml : float — characteristic mesh sizes
    filename         : str   — output .msh path
    """
    gmsh.initialize()
    gmsh.model.add("pec_pml")

    delta = (l_pml - l_dom) / 2   # PML thickness

    # ── Cylinder (PEC) ────────────────────────────────────────────────────────
    cyl  = gmsh.model.occ.addCircle(0, 0, 0, R_cyl)
    cl   = gmsh.model.occ.addCurveLoop([cyl])
    cyl_surf = gmsh.model.occ.addPlaneSurface([cl])

    # ── Square computational domain ───────────────────────────────────────────
    dom_rect = gmsh.model.occ.addRectangle(
        -l_dom/2, -l_dom/2, 0, l_dom, l_dom
    )

    # Subtract cylinder from domain
    bkg_frags, _ = gmsh.model.occ.cut(
        [(2, dom_rect)], [(2, cyl_surf)], removeTool=False
    )

    # ── PML regions ───────────────────────────────────────────────────────────
    # Corners (4)
    pml_c1 = gmsh.model.occ.addRectangle(-l_pml/2, -l_pml/2, 0, delta, delta)
    pml_c2 = gmsh.model.occ.addRectangle( l_dom/2, -l_pml/2, 0, delta, delta)
    pml_c3 = gmsh.model.occ.addRectangle(-l_pml/2,  l_dom/2, 0, delta, delta)
    pml_c4 = gmsh.model.occ.addRectangle( l_dom/2,  l_dom/2, 0, delta, delta)
    corners = [(2, t) for t in [pml_c1, pml_c2, pml_c3, pml_c4]]

    # Strips along x (left, right)
    pml_x1 = gmsh.model.occ.addRectangle(-l_pml/2, -l_dom/2, 0, delta, l_dom)
    pml_x2 = gmsh.model.occ.addRectangle( l_dom/2, -l_dom/2, 0, delta, l_dom)
    x_pmls = [(2, t) for t in [pml_x1, pml_x2]]

    # Strips along y (bottom, top)
    pml_y1 = gmsh.model.occ.addRectangle(-l_dom/2, -l_pml/2, 0, l_dom, delta)
    pml_y2 = gmsh.model.occ.addRectangle(-l_dom/2,  l_dom/2, 0, l_dom, delta)
    y_pmls = [(2, t) for t in [pml_y1, pml_y2]]

    # Fragment all surfaces so they share boundary nodes — prevents field
    # discontinuities at domain/PML interfaces
    all_domains = bkg_frags + corners + x_pmls + y_pmls
    gmsh.model.occ.fragment(all_domains, [])

    gmsh.model.occ.synchronize()

    # ── Physical groups ───────────────────────────────────────────────────────
    # PEC cylinder edge
    gmsh.model.addPhysicalGroup(1, [cyl], tag=_TAG_PEC, name="PEC")

    # Background domain surfaces
    bkg_tags = [t for _, t in bkg_frags]
    gmsh.model.addPhysicalGroup(2, bkg_tags, tag=_TAG_BKG, name="Background")

    # PML surfaces
    gmsh.model.addPhysicalGroup(
        2, [pml_c1, pml_c2, pml_c3, pml_c4], tag=_TAG_PML_XY, name="PML_corners"
    )
    gmsh.model.addPhysicalGroup(
        2, [pml_x1, pml_x2], tag=_TAG_PML_X, name="PML_x"
    )
    gmsh.model.addPhysicalGroup(
        2, [pml_y1, pml_y2], tag=_TAG_PML_Y, name="PML_y"
    )

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

    gmsh.model.mesh.field.add("MathEval", 3)
    gmsh.model.mesh.field.setString(3, "F", f"{lc_pml}")

    gmsh.model.mesh.field.add("Min", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
    gmsh.model.mesh.field.setAsBackgroundMesh(4)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()

    return filename


# ── Core FEM solver ───────────────────────────────────────────────────────────

def solve_pec_cylinder(lc_cyl, lc_dom, lc_pml=None, params=None,
                        mesh_filename="mesh.msh"):
    """
    Solve the PEC cylinder scattering problem with PML truncation.

    Parameters
    ----------
    lc_cyl  : float — mesh size at cylinder surface
    lc_dom  : float — mesh size in computational domain
    lc_pml  : float — mesh size in PML (defaults to lc_dom)
    params  : dict  — override any key in MIE_PARAMS
    mesh_filename : str — .msh output path

    Returns
    -------
    Es_h      : dolfinx Function — scattered field (N1curl)
    Eb        : dolfinx Function — incident field   (N1curl)
    mesh      : dolfinx Mesh
    cell_tags : dolfinx MeshTags — subdomain markers (tag 2 = background)
    num_cells : int
    num_dofs  : int
    h_avg     : float — mean element diameter in background domain [m]
    """
    p      = {**MIE_PARAMS, **(params or {})}
    wl     = p["wl"]
    R_cyl  = p["R_cyl"]
    l_dom  = p["l_dom"]
    l_pml  = p["l_pml"]
    alpha  = p["alpha"]
    degree = p["degree"]
    k0     = 2 * np.pi / wl
    eps_bkg = 1.0
    mu_bkg  = 1.0

    if lc_pml is None:
        lc_pml = lc_dom

    # ── Mesh ──────────────────────────────────────────────────────────────────
    mesh_file = create_mesh(R_cyl, l_dom, l_pml,
                             lc_cyl, lc_dom, lc_pml, mesh_filename)
    msh, cell_tags, facet_tags, _, _, _ = read_from_msh(
        mesh_file, MPI.COMM_WORLD, 0, gdim=2
    )

    bkg_cells = cell_tags.find(_TAG_BKG)
    num_cells  = msh.topology.index_map(2).size_local

    import dolfinx.cpp as cpp
    h_cells = cpp.mesh.h(msh._cpp_object, 2,
                          np.arange(num_cells, dtype=np.int32))
    h_avg = float(np.mean(h_cells[bkg_cells]))  # background only

    # ── Function space ─────────────────────────────────────────────────────────
    V       = functionspace(msh, ("N1curl", degree))
    num_dofs = V.dofmap.index_map.size_global

    # ── Measures ───────────────────────────────────────────────────────────────
    dx      = Measure("dx", domain=msh, subdomain_data=cell_tags)
    dDom    = dx(_TAG_BKG)
    dPml_xy = dx(_TAG_PML_XY)
    dPml_x  = dx(_TAG_PML_X)
    dPml_y  = dx(_TAG_PML_Y)

    # ── Trial / test functions ──────────────────────────────────────────────────
    Es   = TrialFunction(V)
    v    = TestFunction(V)
    Es_3d = ufl.as_vector((Es[0], Es[1], 0))
    v_3d  = ufl.as_vector((v[0],  v[1],  0))

    # ── Incident field  Ei = ŷ exp(−ik₀x) ─────────────────────────────────────
    Eb = Function(V)
    Eb.interpolate(lambda x: (
        np.zeros(x.shape[1], dtype=np.complex128),
        np.exp(-1j * k0 * x[0]),
    ))

    # ── PEC Dirichlet BC:  Es|_PEC = −Ei|_PEC ─────────────────────────────────
    tdim      = msh.topology.dim
    pec_facets = facet_tags.find(_TAG_PEC)
    pec_dofs   = locate_dofs_topological(V, tdim - 1, pec_facets)

    E_pec = Function(V)
    E_pec.interpolate(lambda x: (
        np.zeros(x.shape[1], dtype=np.complex128),
        -np.exp(-1j * k0 * x[0]),
    ))
    bc = dirichletbc(E_pec, dofs=pec_dofs)

    # ── PML coordinate transformations ─────────────────────────────────────────
    x = ufl.SpatialCoordinate(msh)

    xy_pml = ufl.as_vector((
        pml_coordinates(x[0], alpha, k0, l_dom, l_pml),
        pml_coordinates(x[1], alpha, k0, l_dom, l_pml),
    ))
    x_pml = ufl.as_vector((
        pml_coordinates(x[0], alpha, k0, l_dom, l_pml),
        x[1],
    ))
    y_pml = ufl.as_vector((
        x[0],
        pml_coordinates(x[1], alpha, k0, l_dom, l_pml),
    ))

    # ── PML material tensors ────────────────────────────────────────────────────
    eps_bkg_t = ufl.as_matrix(((eps_bkg, 0, 0),
                                (0, eps_bkg, 0),
                                (0, 0, eps_bkg)))
    mu_bkg_t  = ufl.as_matrix(((mu_bkg,  0, 0),
                                (0, mu_bkg,  0),
                                (0, 0, mu_bkg)))

    eps_x,  mu_x  = create_eps_mu(x_pml,  eps_bkg_t, mu_bkg_t)
    eps_y,  mu_y  = create_eps_mu(y_pml,  eps_bkg_t, mu_bkg_t)
    eps_xy, mu_xy = create_eps_mu(xy_pml, eps_bkg_t, mu_bkg_t)

    # ── Weak form ──────────────────────────────────────────────────────────────
    # Background: homogeneous scattered field equation (source via Dirichlet BC)
    # PML regions: modified material tensors with ufl.dot for strict matrix ops
    from dolfinx.fem import Constant
    from petsc4py import PETSc

    a_ufl = (
        -inner(curl_2d(Es), curl_2d(v)) * dDom
        + k0**2 * eps_bkg * inner(Es, v) * dDom

        - inner(ufl.dot(ufl.inv(mu_xy), curl_2d(Es)), curl_2d(v)) * dPml_xy
        + k0**2 * inner(ufl.dot(eps_xy, Es_3d), v_3d) * dPml_xy

        - inner(ufl.dot(ufl.inv(mu_x), curl_2d(Es)), curl_2d(v)) * dPml_x
        + k0**2 * inner(ufl.dot(eps_x, Es_3d), v_3d) * dPml_x

        - inner(ufl.dot(ufl.inv(mu_y), curl_2d(Es)), curl_2d(v)) * dPml_y
        + k0**2 * inner(ufl.dot(eps_y, Es_3d), v_3d) * dPml_y
    )
    L_ufl = inner(Constant(msh, PETSc.ScalarType(0)), v[0]) * dDom

    # ── Solve ──────────────────────────────────────────────────────────────────
    problem = LinearProblem(
        a_ufl, L_ufl, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type":  "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="pec_pml_",
    )
    Es_h = problem.solve()
    Es_h.name = "Scattered_Field"

    return Es_h, Eb, msh, cell_tags, num_cells, num_dofs, h_avg


# ── Stand-alone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    p    = MIE_PARAMS
    wl   = p["wl"]
    k0   = 2 * np.pi / wl

    lc_cyl = wl / 30
    lc_dom = wl / 10
    lc_pml = wl / 8

    if comm.rank == 0:
        print("=" * 60)
        print("PEC CYLINDER SCATTERING SOLVER  (PML)")
        print("=" * 60)
        print(f"  λ      = {wl} m  |  k₀ = {k0:.4f} rad/m")
        print(f"  R_cyl  = {p['R_cyl']} m")
        print(f"  l_dom  = {p['l_dom']} m  |  l_pml = {p['l_pml']} m")
        print(f"  alpha  = {p['alpha']}  (PML absorption strength)")
        print()

    Es_h, Eb, msh, cell_tags, num_cells, num_dofs, h_avg = solve_pec_cylinder(
        lc_cyl, lc_dom, lc_pml
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    if comm.rank == 0:
        from dolfinx.fem import form as fem_form
        from ufl import inner, dx

        Es_energy = assemble_scalar(fem_form(inner(Es_h, Es_h) * dx))
        Ei_energy = assemble_scalar(fem_form(inner(Eb,   Eb)   * dx))
        Es_norm   = float(np.sqrt(abs(Es_energy)))
        Ei_norm   = float(np.sqrt(abs(Ei_energy)))

        print(f"  Cells    : {num_cells}  |  DOFs : {num_dofs}")
        print(f"  h_avg    : {h_avg:.5f} m  |  kh = {k0*h_avg:.4f}")
        print(f"  ‖Es‖     : {Es_norm:.4e}  |  ‖Ei‖ : {Ei_norm:.4e}")
        print(f"  ‖Es‖/‖Ei‖: {Es_norm/Ei_norm:.4f}")

    # ── Interpolate to CG for visualisation ───────────────────────────────────
    degree = p["degree"]
    V_cg   = functionspace(msh, ("Lagrange", degree, (msh.geometry.dim,)))

    E_total = Function(Es_h.function_space)
    E_total.x.array[:] = Eb.x.array[:] + Es_h.x.array[:]

    Es_cg   = Function(V_cg); Es_cg.interpolate(Es_h)
    Etot_cg = Function(V_cg); Etot_cg.interpolate(E_total)
    Ei_cg   = Function(V_cg); Ei_cg.interpolate(Eb)

    with VTXWriter(msh.comm, "E_scattered.bp", [Es_cg])   as f: f.write(0.0)
    with VTXWriter(msh.comm, "E_total.bp",     [Etot_cg]) as f: f.write(0.0)
    with VTXWriter(msh.comm, "E_incident.bp",  [Ei_cg])   as f: f.write(0.0)

    # ── Export scattered field along x-axis for compare.py ───────────────────
    from dolfinx.geometry import (bb_tree, compute_collisions_points,
                                   compute_colliding_cells)
    R_cyl = p["R_cyl"]
    l_dom = p["l_dom"]
    x_line  = np.linspace(R_cyl * 1.05, l_dom / 2 * 0.95, 300)
    pts_3d  = np.column_stack([x_line, np.zeros_like(x_line),
                                np.zeros_like(x_line)])
    tree      = bb_tree(msh, msh.topology.dim)
    cands     = compute_collisions_points(tree, pts_3d)
    cells_out = compute_colliding_cells(msh, cands, pts_3d)

    Ex_line = np.zeros(len(x_line), dtype=np.complex128)
    Ey_line = np.zeros(len(x_line), dtype=np.complex128)
    for i in range(len(x_line)):
        cl = cells_out.links(i)
        if len(cl) > 0:
            val = Es_cg.eval(pts_3d[i:i+1], [cl[0]])
            Ex_line[i] = val[0]
            Ey_line[i] = val[1]

    np.savez("fem_scattered.npz",
             x=x_line,
             Ex=Ex_line,
             Ey=Ey_line,
             E_mag=np.sqrt(np.abs(Ex_line)**2 + np.abs(Ey_line)**2))

    if comm.rank == 0:
        print()
        print("  Saved: E_scattered.bp, E_total.bp, E_incident.bp")
        print("  Saved: fem_scattered.npz  (for compare.py)")
        print()
        print("  Next: python 011_compare.py")  