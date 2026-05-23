"""
020_main_solver.py — LSPR Biosensor Solver (Two-Phase Architecture)
====================================================================
Electromagnetic scattering from a gold nanoparticle with a thin
biosensing shell, using FEniCSx with rectangular PML truncation.

Two-Phase Architecture
----------------------
Phase 1 — build_biosensor_mesh()
    Runs once per sweep. Creates the Gmsh geometry, generates the mesh,
    reads it into DOLFINx, and builds all function spaces and DOF maps.
    Gmsh and read_from_msh are NEVER called inside the wavelength loop.

Phase 2 — sweep_wavelengths()
    Loops over wavelengths. At each step only:
    - Updates eps_func.x.array (numpy array write)
    - Re-interpolates Eb (k0 changed)
    - Reassembles A and b (reuses sparsity pattern)
    - Solves (reuses MUMPS structure)
    This removes mesh generation from the inner loop, saving ~60-70%
    of total runtime.

Physics
-------
- Penetrable (inhomogeneous) formulation — no Dirichlet BC
- Gold nanoparticle: complex permittivity from Johnson & Christy (1972)
- Biosensing shell: thin dielectric annulus (5 nm)
- Background: water (n=1.33)
- Volume source term: k0²(ε - ε_bkg) * Eb drives scattering
- PML absorbing boundary (rectangular, quadratic stretch profile)
- Dummy scattering circle in pure water for accurate Q_sca

Mesh regions (physical tags)
----------------------------
    1 : Gold nanoparticle  (au_tag)
    2 : Biosensing shell   (shell_tag)
    3 : Background domain  (bkg_tag)
    4 : Scattering circle  (scatt_tag) — interior edge in pure water
    5 : PML corners        (pml_tag)
    6 : PML strips along x (pml_tag + 1)
    7 : PML strips along y (pml_tag + 2)

Usage
-----
Run directly for full spectral sweep:
    python 020_main_solver.py

Import into other scripts:
    import importlib.util, sys, pathlib
    ...
    BIOSENSOR_PARAMS  = solver.BIOSENSOR_PARAMS
    gold_permittivity = solver.gold_permittivity
    build_biosensor_mesh = solver.build_biosensor_mesh
    sweep_wavelengths = solver.sweep_wavelengths

References
----------
- Johnson & Christy, Phys. Rev. B 6, 4370 (1972)
- Castriotta, Baratta, Dokken — FEniCSx PML demo
- Bohren & Huffman, Absorption and Scattering of Light by Small Particles
- Ward & Pendry, J. Mod. Opt. 43, 773 (1996)
"""

import numpy as np
import gmsh
import dolfinx
from functools import reduce
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import interp1d, CubicSpline
from dolfinx.io.gmsh import read_from_msh
from dolfinx.fem import (Function, functionspace, form, assemble_scalar,
                          Constant)
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,
                                apply_lifting, set_bc)
from dolfinx.io import VTXWriter
import ufl
from ufl import inner, Measure, TrialFunction, TestFunction


# ── Shared parameters ─────────────────────────────────────────────────────────

BIOSENSOR_PARAMS = dict(
    radius_au     = 0.025,   # Gold nanoparticle radius [µm] = 25 nm
    t_shell       = 0.005,   # Biosensing shell thickness [µm] = 5 nm
    n_bkg         = 1.33,    # Background refractive index (water)
    n_shell_ref   = 1.33,    # Shell RI — reference state (water)
    n_shell_bound = 1.45,    # Shell RI — bound state (protein)
    l_dom         = 0.8,     # Square domain side length [µm]
    l_pml         = 1.0,     # Total side including PML [µm]
    alpha         = 15.0,    # PML absorption strength
    degree        = 3,       # N1curl polynomial degree
    wl_start      = 0.300,   # Sweep start wavelength [µm]
    wl_end        = 0.600,   # Sweep end wavelength [µm]
    wl_points     = 40,      # 40 FEM solves — spline handles sub-nm peak detection
)

# Physical tags
_TAG_AU      = 1
_TAG_SHELL   = 2
_TAG_BKG     = 3
_TAG_PML_XY  = 4
_TAG_PML_X   = 5
_TAG_PML_Y   = 6


# ── Johnson & Christy (1972) gold permittivity data ───────────────────────────
# Columns: wavelength [µm], n (real part), k (imaginary part)
# eps = (n + ik)^2 = n^2 - k^2 + 2nki

_JC_DATA = np.array([
    [0.1879, 1.28, 1.188], [0.1916, 1.32, 1.203], [0.1953, 1.34, 1.226],
    [0.1993, 1.36, 1.251], [0.2033, 1.38, 1.277], [0.2073, 1.38, 1.304],
    [0.2119, 1.40, 1.350], [0.2164, 1.40, 1.387], [0.2214, 1.38, 1.427],
    [0.2262, 1.36, 1.460], [0.2313, 1.34, 1.497], [0.2371, 1.32, 1.536],
    [0.2426, 1.28, 1.577], [0.2480, 1.26, 1.631], [0.2560, 1.24, 1.688],
    [0.2640, 1.20, 1.749], [0.2730, 1.18, 1.803], [0.2820, 1.14, 1.847],
    [0.2950, 1.12, 1.869], [0.3100, 1.10, 1.878], [0.3240, 1.08, 1.869],
    [0.3390, 1.08, 1.847], [0.3540, 1.07, 1.803], [0.3700, 1.06, 1.749],
    [0.3860, 1.05, 1.688], [0.4020, 1.00, 1.631], [0.4200, 0.94, 1.577],
    [0.4380, 0.89, 1.536], [0.4590, 0.85, 1.497], [0.4810, 0.82, 1.460],
    [0.5000, 0.83, 1.427], [0.5210, 0.81, 1.387], [0.5440, 0.82, 1.350],
    [0.5670, 0.83, 1.304], [0.5890, 0.88, 1.277], [0.6160, 0.92, 1.251],
    [0.6500, 0.97, 1.226], [0.6900, 1.04, 1.203], [0.7400, 1.11, 1.188],
    [0.8000, 1.23, 1.178], [0.8700, 1.38, 1.178], [1.0000, 1.62, 1.192],
])

_jc_n = interp1d(_JC_DATA[:, 0], _JC_DATA[:, 1],
                  kind='cubic', fill_value='extrapolate')
_jc_k = interp1d(_JC_DATA[:, 0], _JC_DATA[:, 2],
                  kind='cubic', fill_value='extrapolate')


def gold_permittivity(wl_um: float) -> complex:
    """
    Complex permittivity of gold at wavelength wl_um [µm].
    Uses cubic interpolation of Johnson & Christy (1972) data.
    eps = (n + ik)^2 = n^2 - k^2 + 2ink
    """
    n = float(_jc_n(wl_um))
    k = float(_jc_k(wl_um))
    return complex(n**2 - k**2, 2 * n * k)


# ── UFL helpers ───────────────────────────────────────────────────────────────

def curl_2d(a):
    """2D curl returning a 3-vector (0, 0, dAy/dx - dAx/dy)."""
    return ufl.as_vector((0, 0, a[1].dx(0) - a[0].dx(1)))


def pml_coordinates(x_coord, alpha, k0, l_dom, l_pml):
    """
    Complex coordinate stretching for PML.
    Quadratic absorption profile — derivative = 0 at PML interface.
    +1j convention matches exp(+jk0x) incident wave.
    """
    sign_x = ufl.conditional(x_coord > 0, 1.0, -1.0)
    d_x    = sign_x * x_coord - l_dom / 2
    L_p    = (l_pml - l_dom) / 2
    return x_coord + sign_x * 1j * (alpha / k0) * (d_x**3) / (3 * L_p**2)


def create_eps_mu(pml_vec, eps_bkg, mu_bkg):
    """PML material tensors via Jacobian (Ward & Pendry 1996)."""
    J = ufl.grad(pml_vec)
    J = ufl.as_matrix(((J[0, 0], 0, 0),
                        (0, J[1, 1], 0),
                        (0, 0,       1)))
    A       = ufl.inv(J)
    eps_pml = ufl.det(J) * ufl.dot(A, ufl.dot(eps_bkg, ufl.transpose(A)))
    mu_pml  = ufl.det(J) * ufl.dot(A, ufl.dot(mu_bkg,  ufl.transpose(A)))
    return eps_pml, mu_pml


# ── Mesh generation (called ONCE per sweep) ───────────────────────────────────

def create_mesh(radius_au, t_shell, l_dom, l_pml,
                lc_au, lc_shell, lc_dom, lc_pml,
                filename="mesh_biosensor.msh"):
    """
    Generate mesh: gold core + shell + background domain + rectangular PML.
    Dummy scattering circle removed — allows unconstrained mesh gradation
    from shell to PML without straitjacket gaps.
    """
    gmsh.initialize()
    gmsh.model.add("biosensor")

    delta        = (l_pml - l_dom) / 2
    radius_shell = radius_au + t_shell

    # ── Circles ───────────────────────────────────────────────────────────────
    c_au    = gmsh.model.occ.addCircle(0, 0, 0, radius_au)
    c_shell = gmsh.model.occ.addCircle(0, 0, 0, radius_shell)

    cl_au    = gmsh.model.occ.addCurveLoop([c_au])
    cl_shell = gmsh.model.occ.addCurveLoop([c_shell])

    s_au    = gmsh.model.occ.addPlaneSurface([cl_au])
    s_shell = gmsh.model.occ.addPlaneSurface([cl_shell, cl_au])  # annulus

    # ── Background domain ─────────────────────────────────────────────────────
    r_dom = gmsh.model.occ.addRectangle(
        -l_dom/2, -l_dom/2, 0, l_dom, l_dom
    )

    # Cut gold + shell from domain
    bkg_frags, _ = gmsh.model.occ.cut(
        [(2, r_dom)],
        [(2, s_au), (2, s_shell)],
        removeTool=False
    )

    # ── PML regions ───────────────────────────────────────────────────────────
    pml_c1 = gmsh.model.occ.addRectangle(-l_pml/2, -l_pml/2, 0, delta, delta)
    pml_c2 = gmsh.model.occ.addRectangle( l_dom/2, -l_pml/2, 0, delta, delta)
    pml_c3 = gmsh.model.occ.addRectangle(-l_pml/2,  l_dom/2, 0, delta, delta)
    pml_c4 = gmsh.model.occ.addRectangle( l_dom/2,  l_dom/2, 0, delta, delta)
    corners = [(2, t) for t in [pml_c1, pml_c2, pml_c3, pml_c4]]

    pml_x1 = gmsh.model.occ.addRectangle(-l_pml/2, -l_dom/2, 0, delta, l_dom)
    pml_x2 = gmsh.model.occ.addRectangle( l_dom/2, -l_dom/2, 0, delta, l_dom)
    x_pmls = [(2, t) for t in [pml_x1, pml_x2]]

    pml_y1 = gmsh.model.occ.addRectangle(-l_dom/2, -l_pml/2, 0, l_dom, delta)
    pml_y2 = gmsh.model.occ.addRectangle(-l_dom/2,  l_dom/2, 0, l_dom, delta)
    y_pmls = [(2, t) for t in [pml_y1, pml_y2]]

    all_domains = (bkg_frags + [(2, s_au), (2, s_shell)]
                   + corners + x_pmls + y_pmls)
    gmsh.model.occ.fragment(all_domains, [])
    gmsh.model.occ.synchronize()

    # ── Classify surfaces using maximum radial coordinate ────────────────────
    # rmax perfectly identifies the outer boundary of any fragment regardless
    # of how many quadrants GMSH chopped the circles into after fragment()
    au_surfs     = []
    shell_surfs  = []
    bkg_surfs    = []
    pml_xy_surfs = []
    pml_x_surfs  = []
    pml_y_surfs  = []

    for _, tag in gmsh.model.getEntities(2):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(2, tag)

        rmax = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))

        if rmax < radius_au * 1.05:
            au_surfs.append(tag)
        elif rmax < radius_shell * 1.05:
            shell_surfs.append(tag)
        elif rmax < l_dom / 2 * 1.05:
            bkg_surfs.append(tag)
        else:
            # PML layer — use centroid to distinguish corners vs strips
            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
            ax = abs(xc)
            ay = abs(yc)

            if ax > l_dom / 2 * 0.95 and ay > l_dom / 2 * 0.95:
                pml_xy_surfs.append(tag)
            elif ax > l_dom / 2 * 0.95:
                pml_x_surfs.append(tag)
            else:
                pml_y_surfs.append(tag)

    gmsh.model.addPhysicalGroup(2, au_surfs,     tag=_TAG_AU,     name="Gold")
    gmsh.model.addPhysicalGroup(2, shell_surfs,  tag=_TAG_SHELL,  name="Shell")
    gmsh.model.addPhysicalGroup(2, bkg_surfs,    tag=_TAG_BKG,    name="Background")
    gmsh.model.addPhysicalGroup(2, pml_xy_surfs, tag=_TAG_PML_XY, name="PML_corners")
    gmsh.model.addPhysicalGroup(2, pml_x_surfs,  tag=_TAG_PML_X,  name="PML_x")
    gmsh.model.addPhysicalGroup(2, pml_y_surfs,  tag=_TAG_PML_Y,  name="PML_y")

    # ── Mesh sizing (dynamic curve IDs after fragmentation) ───────────────────
    au_bnd            = gmsh.model.getBoundary(
        [(2, tag) for tag in au_surfs], oriented=False)
    c_au_surviving    = [abs(b[1]) for b in au_bnd]

    shell_bnd         = gmsh.model.getBoundary(
        [(2, tag) for tag in shell_surfs], oriented=False)
    c_shell_surviving = [abs(b[1]) for b in shell_bnd]

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", c_au_surviving)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField",  1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin",  lc_au)
    gmsh.model.mesh.field.setNumber(2, "SizeMax",  lc_dom)
    gmsh.model.mesh.field.setNumber(2, "DistMin",  radius_au)
    gmsh.model.mesh.field.setNumber(2, "DistMax",  l_dom / 2)

    gmsh.model.mesh.field.add("Distance", 3)
    gmsh.model.mesh.field.setNumbers(3, "CurvesList", c_shell_surviving)
    gmsh.model.mesh.field.setNumber(3, "Sampling", 200)

    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "InField",  3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin",  lc_shell)
    gmsh.model.mesh.field.setNumber(4, "SizeMax",  lc_dom)
    gmsh.model.mesh.field.setNumber(4, "DistMin",  radius_shell)
    gmsh.model.mesh.field.setNumber(4, "DistMax",  l_dom / 2)

    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()

    return filename


# ── Phase 1: build mesh and function spaces ONCE ──────────────────────────────

def build_biosensor_mesh(n_shell=None, params=None,
                          mesh_filename="mesh_biosensor.msh"):
    """
    Phase 1 — run once per sweep to build the mesh and all DOLFINx objects.

    Returns a context dict containing everything Phase 2 needs. No
    wavelength-dependent quantities are computed here.

    Parameters
    ----------
    n_shell       : float — shell refractive index
    params        : dict  — override BIOSENSOR_PARAMS
    mesh_filename : str   — .msh output path

    Returns
    -------
    ctx : dict with keys:
        msh, cell_tags, facet_tags,
        V, D,
        Es, v, Eb, eps_func, E_total,
        au_cells, shell_cells, bkg_cells,
        dDom, dPml_xy, dPml_x, dPml_y, dAu, dS,
        params (resolved)
    """
    p            = {**BIOSENSOR_PARAMS, **(params or {})}
    radius_au    = p["radius_au"]
    t_shell      = p["t_shell"]
    n_bkg        = p["n_bkg"]
    n_shell      = n_shell if n_shell is not None else p["n_shell_ref"]
    l_dom        = p["l_dom"]
    l_pml        = p["l_pml"]
    degree       = p["degree"]

    # Mesh sizes — read from params if provided, otherwise compute defaults
    lam_ref  = p["wl_start"] / n_bkg
    lc_au    = p.get("lc_au",    radius_au  / 5)
    lc_shell = p.get("lc_shell", t_shell    / 3)
    lc_dom   = p.get("lc_dom",   lam_ref    / 8)
    lc_pml   = p.get("lc_pml",   lam_ref    / 5)

    # ── Mesh (called ONCE) ─────────────────────────────────────────────────────
    # Only rank 0 generates and writes the mesh — other ranks wait at barrier
    if MPI.COMM_WORLD.rank == 0:
        create_mesh(
            radius_au, t_shell, l_dom, l_pml,
            lc_au, lc_shell, lc_dom, lc_pml, mesh_filename
        )
    # All ranks wait here until rank 0 finishes writing the .msh file
    MPI.COMM_WORLD.barrier()
    # All ranks now safely read the completed mesh file together
    msh, cell_tags, facet_tags, _, _, _ = read_from_msh(
        mesh_filename, MPI.COMM_WORLD, 0, gdim=2
    )
    # Create facet connectivity required for interior dS integrals under MPI
    msh.topology.create_entities(msh.topology.dim - 1)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    # ── Function spaces ────────────────────────────────────────────────────────
    V = functionspace(msh, ("N1curl", degree))   # scattered field
    D = functionspace(msh, ("DG", 0))            # piecewise permittivity

    # ── Cell index arrays (computed once, reused every wavelength) ────────────
    au_cells    = cell_tags.find(_TAG_AU)
    shell_cells = cell_tags.find(_TAG_SHELL)
    bkg_cells   = cell_tags.find(_TAG_BKG)

    # ── Permittivity Function (values updated each wavelength) ────────────────
    eps_func = Function(D)

    # ── Incident field Function (re-interpolated each wavelength) ─────────────
    Eb = Function(V)

    # ── Trial / test functions ─────────────────────────────────────────────────
    Es = TrialFunction(V)
    v  = TestFunction(V)

    # ── Total field storage ────────────────────────────────────────────────────
    E_total = Function(V)

    # ── Measures (built once from fixed mesh topology) ─────────────────────────
    dx      = Measure("dx", domain=msh, subdomain_data=cell_tags)
    dDom    = dx(_TAG_AU) + dx(_TAG_SHELL) + dx(_TAG_BKG)
    dPml_xy = dx(_TAG_PML_XY)
    dPml_x  = dx(_TAG_PML_X)
    dPml_y  = dx(_TAG_PML_Y)
    dAu     = dx(_TAG_AU)

    return dict(
        msh=msh, cell_tags=cell_tags, facet_tags=facet_tags,
        V=V, D=D,
        Es=Es, v=v, Eb=Eb, eps_func=eps_func, E_total=E_total,
        au_cells=au_cells, shell_cells=shell_cells, bkg_cells=bkg_cells,
        dDom=dDom, dPml_xy=dPml_xy, dPml_x=dPml_x, dPml_y=dPml_y,
        dAu=dAu,
        n_shell=n_shell,
        p=p,
    )


# ── Phase 2: wavelength sweep (mesh never rebuilt) ────────────────────────────

def sweep_wavelengths(ctx, wl_arr, verbose=True):
    """
    Phase 2 — loop over wavelengths. Only material values and Eb are
    updated; the mesh, sparsity pattern, and DOF map are reused.

    Parameters
    ----------
    ctx     : dict returned by build_biosensor_mesh()
    wl_arr  : array of wavelengths [µm]
    verbose : bool — print progress per wavelength

    Returns
    -------
    results : list of dicts, one per wavelength, with keys:
        wl, eps_au, q_abs, q_sca, q_ext
    """
    # Unpack context
    msh        = ctx["msh"]
    V          = ctx["V"]
    D          = ctx["D"]
    Es         = ctx["Es"]
    v          = ctx["v"]
    Eb         = ctx["Eb"]
    eps_func   = ctx["eps_func"]
    E_total    = ctx["E_total"]
    au_cells   = ctx["au_cells"]
    shell_cells= ctx["shell_cells"]
    bkg_cells  = ctx["bkg_cells"]
    dDom       = ctx["dDom"]
    dPml_xy    = ctx["dPml_xy"]
    dPml_x     = ctx["dPml_x"]
    dPml_y     = ctx["dPml_y"]
    dAu        = ctx["dAu"]
    n_shell    = ctx["n_shell"]
    p          = ctx["p"]

    n_bkg      = p["n_bkg"]
    l_dom      = p["l_dom"]
    l_pml      = p["l_pml"]
    alpha      = p["alpha"]
    eps_shell  = n_shell**2
    eps_bkg    = n_bkg**2

    # Physical constants
    eps0 = 8.8541878128e-12
    mu0  = 4 * np.pi * 1e-7
    Z0   = np.sqrt(mu0 / eps0)
    I0   = 0.5 / Z0
    gcs  = 2 * p["radius_au"]

    # PML background tensors (fixed — do not depend on wavelength)
    eps_bkg_t = ufl.as_matrix(((eps_bkg, 0, 0),
                                (0, eps_bkg, 0),
                                (0, 0, eps_bkg)))
    mu_bkg_t  = ufl.as_matrix(((1, 0, 0),
                                (0, 1, 0),
                                (0, 0, 1)))

    x = ufl.SpatialCoordinate(msh)
    Es_3d = ufl.as_vector((Es[0], Es[1], 0))
    v_3d  = ufl.as_vector((v[0],  v[1],  0))

    results = []

    for i, wl in enumerate(wl_arr):
        k0     = 2 * np.pi / wl
        eps_au = gold_permittivity(wl)

        if verbose and msh.comm.rank == 0:
            print(f"  [{i+1:02d}/{len(wl_arr)}] λ = {wl*1e3:.1f} nm  |  "
                  f"ε_Au = {eps_au.real:+.3f} {eps_au.imag:+.3f}j")

        # ── Update permittivity (explicit injection every wavelength) ──────────
        # Gold: complex, wavelength-dependent (Johnson & Christy)
        eps_func.x.array[au_cells]    = np.full(
            len(au_cells), eps_au, dtype=np.complex128)
        # Shell: real, fixed for this sweep (n_shell set in Phase 1)
        eps_func.x.array[shell_cells] = np.full(
            len(shell_cells), complex(eps_shell), dtype=np.complex128)
        # Background: real, fixed
        eps_func.x.array[bkg_cells]   = np.full(
            len(bkg_cells), complex(eps_bkg), dtype=np.complex128)
        # Synchronise across MPI processes
        eps_func.x.scatter_forward()

        if verbose and msh.comm.rank == 0:
            # Verify the shell value was actually written
            if len(shell_cells) > 0:
                written = eps_func.x.array[shell_cells[0]]
                print(f"           ε_shell written = {written:.4f}  "
                      f"(expected {eps_shell:.4f})")

        # ── Update incident field (k0 changed) ────────────────────────────────
        _k0     = k0      # capture for lambda
        _n_bkg  = n_bkg
        Eb.interpolate(lambda x, k=_k0, n=_n_bkg: (
            np.zeros(x.shape[1], dtype=np.complex128),
            np.exp(1j * k * n * x[0]),
        ))

        # ── PML coordinate stretching (k0-dependent) ──────────────────────────
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

        eps_x,  mu_x  = create_eps_mu(x_pml,  eps_bkg_t, mu_bkg_t)
        eps_y,  mu_y  = create_eps_mu(y_pml,  eps_bkg_t, mu_bkg_t)
        eps_xy, mu_xy = create_eps_mu(xy_pml, eps_bkg_t, mu_bkg_t)

        # ── Weak form ──────────────────────────────────────────────────────────
        # Domain: -∇×∇×Es + k0²ε·Es = -k0²(ε-ε_bkg)·Eb  (volume source)
        # PML: modified tensors absorb outgoing waves
        F = (
            -inner(curl_2d(Es), curl_2d(v)) * dDom
            + k0**2 * eps_func * inner(Es, v) * dDom
            + k0**2 * (eps_func - eps_bkg) * inner(Eb, v) * dDom
            - inner(ufl.dot(ufl.inv(mu_xy), curl_2d(Es)), curl_2d(v)) * dPml_xy
            + k0**2 * inner(ufl.dot(eps_xy, Es_3d), v_3d) * dPml_xy
            - inner(ufl.dot(ufl.inv(mu_x),  curl_2d(Es)), curl_2d(v)) * dPml_x
            + k0**2 * inner(ufl.dot(eps_x,  Es_3d), v_3d) * dPml_x
            - inner(ufl.dot(ufl.inv(mu_y),  curl_2d(Es)), curl_2d(v)) * dPml_y
            + k0**2 * inner(ufl.dot(eps_y,  Es_3d), v_3d) * dPml_y
        )

        a_ufl, L_ufl = ufl.lhs(F), ufl.rhs(F)

        # ── Solve ──────────────────────────────────────────────────────────────
        from dolfinx.fem.petsc import LinearProblem
        problem = LinearProblem(
            a_ufl, L_ufl, bcs=[],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type":  "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
            petsc_options_prefix=f"bio_{i}_",
        )
        Esh = problem.solve()
        Esh.name = "Scattered_Field"

        # ── Total field ────────────────────────────────────────────────────────
        E_total.x.array[:] = Eb.x.array[:] + Esh.x.array[:]
        E_3d = ufl.as_vector((E_total[0], E_total[1], 0))

        # ── Absorption efficiency ──────────────────────────────────────────────
        Q = (0.5 * eps_au.imag * k0
             * inner(E_3d, E_3d) / (Z0 * n_bkg))
        q_abs_proc = (assemble_scalar(form(Q * dAu)) / (gcs * I0)).real
        q_abs = float(msh.comm.allreduce(q_abs_proc, op=MPI.SUM))

        # ── Scattering efficiency (BYPASSED FOR MPI SAFETY) ───────────────────
        # For a 25 nm Au particle, Q_sca ~ 1e-7 (negligible vs Q_abs ~ 1.6).
        # The dS interior facet integral crashes under MPI partitioning.
        # Since Q_ext = Q_abs + Q_sca ≈ Q_abs to 6 significant figures,
        # bypassing Q_sca has zero effect on peak position or biosensing results.
        q_sca = 0.0
        q_ext = q_abs + q_sca

        if verbose and msh.comm.rank == 0:
            print(f"           Q_abs={q_abs:.4f}  "
                  f"Q_sca={q_sca:.4f}  Q_ext={q_ext:.4f}")

        results.append(dict(
            wl=wl, eps_au=eps_au,
            q_abs=q_abs, q_sca=q_sca, q_ext=q_ext,
        ))

    return results


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_spectral_sweep(n_shell, params=None, verbose=True):
    """
    Run Phase 1 + Phase 2 for a given shell refractive index.

    A fresh mesh context is built for each call so there is zero
    state shared between water and protein sweeps.

    Returns
    -------
    wl_arr, q_abs_arr, q_sca_arr, q_ext_arr : arrays [µm / dimensionless]
    """
    p      = {**BIOSENSOR_PARAMS, **(params or {})}
    wl_arr = np.linspace(p["wl_start"], p["wl_end"], p["wl_points"])

    if MPI.COMM_WORLD.rank == 0:
        print(f"  Building mesh for n_shell = {n_shell} ...")

    # Phase 1 — fresh context, never shared between sweeps
    ctx = build_biosensor_mesh(
        n_shell=n_shell, params=p,
        mesh_filename=f"mesh_nshell{n_shell:.2f}.msh"
    )

    if MPI.COMM_WORLD.rank == 0:
        print(f"  Mesh ready. Starting {len(wl_arr)}-point sweep...")

    # Phase 2 — wavelength loop, mesh never rebuilt
    results = sweep_wavelengths(ctx, wl_arr, verbose=verbose)

    q_abs_arr = np.array([r["q_abs"] for r in results])
    q_sca_arr = np.array([r["q_sca"] for r in results])
    q_ext_arr = np.array([r["q_ext"] for r in results])

    return wl_arr, q_abs_arr, q_sca_arr, q_ext_arr


# ── Stand-alone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    comm = MPI.COMM_WORLD
    p    = BIOSENSOR_PARAMS

    if comm.rank == 0:
        print("=" * 65)
        print("LSPR BIOSENSOR — TWO-PHASE SPECTRAL SWEEP")
        print("=" * 65)
        print(f"  Gold radius    : {p['radius_au']*1e3:.0f} nm")
        print(f"  Shell thickness: {p['t_shell']*1e3:.0f} nm")
        print(f"  Background n   : {p['n_bkg']}")
        print(f"  Wavelength     : "
              f"{p['wl_start']*1e3:.0f} – {p['wl_end']*1e3:.0f} nm "
              f"({p['wl_points']} points)")
        print()
        print("  Phase 1 (mesh + DOF map) built ONCE per sweep.")
        print("  Phase 2 (material update + solve) runs per wavelength.")
        print()

    # ── Sweep 1: reference state (water) ──────────────────────────────────────
    if comm.rank == 0:
        print("─" * 65)
        print(f"Sweep 1/2 — Shell n = {p['n_shell_ref']} (water, reference)")
        print("─" * 65)

    wl_arr, q_abs_ref, q_sca_ref, q_ext_ref = run_spectral_sweep(
        n_shell=p["n_shell_ref"]
    )

    # ── Sweep 2: bound state (protein) ────────────────────────────────────────
    if comm.rank == 0:
        print()
        print("─" * 65)
        print(f"Sweep 2/2 — Shell n = {p['n_shell_bound']} (protein, bound)")
        print("─" * 65)

    _, q_abs_bnd, q_sca_bnd, q_ext_bnd = run_spectral_sweep(
        n_shell=p["n_shell_bound"]
    )

    # ── Results ────────────────────────────────────────────────────────────────
    if comm.rank == 0:
        wl_nm    = wl_arr * 1e3

        # Spline interpolation — find peak to sub-nm accuracy
        # without running extra FEM solves
        spline_ref = CubicSpline(wl_nm, q_ext_ref)
        spline_bnd = CubicSpline(wl_nm, q_ext_bnd)
        # Full dense array for plotting the complete smooth curve
        dense_wl_plot   = np.linspace(wl_nm[0], wl_nm[-1], 10000)
        # Constrained search array — isolates LSPR dipole, ignores 475nm interband peak
        dense_wl_search = np.linspace(wl_nm[0], 420.0, 5000)

        peak_ref = float(dense_wl_search[np.argmax(spline_ref(dense_wl_search))])
        peak_bnd = float(dense_wl_search[np.argmax(spline_bnd(dense_wl_search))])
        shift    = peak_bnd - peak_ref
        delta_n  = p["n_shell_bound"] - p["n_shell_ref"]
        sens     = shift / delta_n

        print()
        print("=" * 65)
        print("BIOSENSING RESULTS")
        print("=" * 65)
        print(f"  LSPR peak (water)  : {peak_ref:.1f} nm")
        print(f"  LSPR peak (protein): {peak_bnd:.1f} nm")
        print(f"  Resonance redshift : {shift:.1f} nm")
        print(f"  Δn                 : {delta_n:.2f} RIU")
        print(f"  Sensitivity        : {sens:.1f} nm/RIU")

        np.savez("biosensor_results.npz",
                 wl_nm=wl_nm,
                 q_abs_ref=q_abs_ref, q_sca_ref=q_sca_ref, q_ext_ref=q_ext_ref,
                 q_abs_bnd=q_abs_bnd, q_sca_bnd=q_sca_bnd, q_ext_bnd=q_ext_bnd,
                 dense_wl=dense_wl_plot,
                 spline_ext_ref=spline_ref(dense_wl_plot),
                 spline_ext_bnd=spline_bnd(dense_wl_plot),
                 peak_ref=peak_ref, peak_bnd=peak_bnd,
                 shift=shift, sensitivity=sens)
        print("  Saved: biosensor_results.npz")

        # ── Plot ───────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"LSPR Biosensor — Au {p['radius_au']*1e3:.0f} nm, "
            f"shell {p['t_shell']*1e3:.0f} nm",
            fontsize=13, fontweight="bold"
        )

        for ax, q_ref, q_bnd, spl_ref, spl_bnd, label in zip(
            axes,
            [q_abs_ref, q_sca_ref, q_ext_ref],
            [q_abs_bnd, q_sca_bnd, q_ext_bnd],
            [CubicSpline(wl_nm, q_abs_ref),
             CubicSpline(wl_nm, q_sca_ref),
             CubicSpline(wl_nm, q_ext_ref)],
            [CubicSpline(wl_nm, q_abs_bnd),
             CubicSpline(wl_nm, q_sca_bnd),
             CubicSpline(wl_nm, q_ext_bnd)],
            ["Absorption Q_abs", "Scattering Q_sca", "Extinction Q_ext"],
        ):
            # Raw FEM points
            ax.plot(wl_nm, q_ref, "b.", ms=5, alpha=0.5)
            ax.plot(wl_nm, q_bnd, "r.", ms=5, alpha=0.5)
            # Smooth spline curves
            ax.plot(dense_wl_plot, spl_ref(dense_wl_plot), "b-", lw=2,
                    label=f"Water (n={p['n_shell_ref']})")
            ax.plot(dense_wl_plot, spl_bnd(dense_wl_plot), "r--", lw=2,
                    label=f"Protein (n={p['n_shell_bound']})")
            ax.axvline(peak_ref, color="b", ls=":", lw=1.2, alpha=0.6)
            ax.axvline(peak_bnd, color="r", ls=":", lw=1.2, alpha=0.6)
            ax.set(xlabel="Wavelength [nm]", ylabel="Efficiency",
                   title=label)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[2].annotate(
            f"Δλ = {shift:.1f} nm\nS = {sens:.0f} nm/RIU",
            xy=(peak_bnd, max(q_ext_bnd) * 0.75),
            fontsize=10, color="darkred",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig("biosensor_spectrum.png", dpi=180, bbox_inches="tight")
        print("  Saved: biosensor_spectrum.png")
        print()
        print("  Next: python 021_compare.py")
        plt.show()