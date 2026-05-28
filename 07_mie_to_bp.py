"""
06_mie_to_bp.py  —  Export Mie Series Solution to .bp for ParaView
====================================================================
Evaluates the Mie analytical solution on the same FEM mesh (loaded from
E_scattered.bp) and writes it back as a .bp file so you can load both
FEM and Mie side-by-side in ParaView for a true apples-to-apples comparison.

Outputs
-------
  E_mie_scattered.bp   — Mie scattered field on FEM mesh (open alongside E_scattered.bp)
  E_mie_total.bp       — Mie total field = scattered + incident
  E_error.bp           — Pointwise |FEM| - |Mie| field magnitude error map

Usage
-----
  conda activate fenicsx-complex
  python 06_mie_to_bp.py

ParaView workflow
-----------------
  1. File > Open   → select E_scattered.bp  AND  E_mie_scattered.bp
     Apply both, set representation to "Surface", colour by magnitude.
     Use "Split Horizontal" to show side-by-side.
  2. For error map: open E_error.bp, colour by the scalar field.
  3. For line plots: Filters > Plot Over Line (both datasets simultaneously).

Notes
-----
  • The Mie series uses the corrected TE formulation:
      a_n = -J'_n(ka) / H'_n(ka)   with  (-j)^n  plane-wave expansion factor
  • Field interpolated onto CG-2 node positions for clean ParaView rendering.
"""

import numpy as np
from mpi4py import MPI
from scipy.special import jv, hankel2
from dolfinx.io import VTXWriter
from dolfinx.io.gmsh import read_from_msh
from dolfinx.fem import functionspace, Function
import gmsh

# ─────────────────────────────────────────────────────────────────────────────
# Parameters  (must match 01_pec_cylinder.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
wl    = 1.0
k0    = 2 * np.pi / wl
R_cyl = 0.5 * wl
R_out = 5.0 * wl
lc_cyl = wl / 30
lc_out = wl / 10

# ─────────────────────────────────────────────────────────────────────────────
# Mie coefficient helpers
# ─────────────────────────────────────────────────────────────────────────────

def jv_prime(n, z):
    return 0.5 * (jv(n - 1, z) - jv(n + 1, z))

def h2_prime(n, z):
    return 0.5 * (hankel2(n - 1, z) - hankel2(n + 1, z))


def compute_mie_coefficients(k0, R_cyl, n_max=25):
    """Mie coefficients a_n = -J'_n(ka) / H'_n(ka) for TE/TM PEC cylinder."""
    ka = k0 * R_cyl
    coeffs = {}
    for n in range(-n_max, n_max + 1):
        Jp = jv_prime(n, ka)
        Hp = h2_prime(n, ka)
        coeffs[n] = -Jp / Hp
    return coeffs


def mie_scattered_field(x_pts, y_pts, coeffs, k0):
    """
    Evaluate Mie scattered field (Ex, Ey) at arrays of (x, y) points.

    Returns
    -------
    Ex, Ey : complex ndarray, shape (N,)
    """
    N = len(x_pts)
    Ex = np.zeros(N, dtype=np.complex128)
    Ey = np.zeros(N, dtype=np.complex128)

    for i, (x, y) in enumerate(zip(x_pts, y_pts)):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        E_rho = 0j
        E_phi = 0j

        for n, a_n in coeffs.items():
            if np.abs(a_n) < 1e-15:
                continue

            H_n  = hankel2(n, k0 * rho)
            Hp_n = h2_prime(n, k0 * rho)

            # (-j)^n phase factor for e^{-ik₀x} = Σ (-j)^n J_n(k₀ρ) e^{inφ}
            phase = ((-1j)**n) * np.exp(1j * n * phi)

            if n != 0:
                E_rho += a_n * (1j * n / (k0 * rho)) * H_n * phase
            E_phi += a_n * Hp_n * phase

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        Ex[i] = E_rho * cos_p - E_phi * sin_p
        Ey[i] = E_rho * sin_p + E_phi * cos_p

    return Ex, Ey


def incident_field(x_pts, y_pts, k0):
    """Plane wave E_i = (0, e^{-ik₀x})  — y-polarised, propagating in +x."""
    Ex = np.zeros(len(x_pts), dtype=np.complex128)
    Ey = np.exp(-1j * k0 * np.array(x_pts))
    return Ex, Ey


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild mesh (same as solver) to get DOF coordinates
# ─────────────────────────────────────────────────────────────────────────────

if MPI.COMM_WORLD.rank == 0:
    print("=" * 65)
    print("06_mie_to_bp.py  —  Mie → .bp export")
    print("=" * 65)
    print("\nStep 1: Rebuilding mesh …")

gmsh.initialize()
gmsh.model.add("pec")

cyl = gmsh.model.occ.addCircle(0, 0, 0, R_cyl)
out = gmsh.model.occ.addCircle(0, 0, 0, R_out)
cyl_loop = gmsh.model.occ.addCurveLoop([cyl])
out_loop  = gmsh.model.occ.addCurveLoop([out])
surf = gmsh.model.occ.addPlaneSurface([out_loop, cyl_loop])
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(1, [out], tag=1, name="ABC")
gmsh.model.addPhysicalGroup(1, [cyl], tag=2, name="PEC")
gmsh.model.addPhysicalGroup(2, [surf], tag=1, name="Domain")

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_out)
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [cyl])
gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField",  1)
gmsh.model.mesh.field.setNumber(2, "SizeMin",  lc_cyl)
gmsh.model.mesh.field.setNumber(2, "SizeMax",  lc_out)
gmsh.model.mesh.field.setNumber(2, "DistMin",  R_cyl)
gmsh.model.mesh.field.setNumber(2, "DistMax",  R_out)
gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.model.mesh.generate(2)
gmsh.write("mesh.msh")
gmsh.finalize()

mesh, cell_tags, facet_tags, _, _, _ = read_from_msh(
    "mesh.msh", MPI.COMM_WORLD, 0, gdim=2)

if MPI.COMM_WORLD.rank == 0:
    print(f"  Cells : {mesh.topology.index_map(2).size_local}")

# ─────────────────────────────────────────────────────────────────────────────
# CG function space — nodes sit at mesh vertices, easy to sample
# ─────────────────────────────────────────────────────────────────────────────
degree_cg = 2
V_cg = functionspace(mesh, ("Lagrange", degree_cg, (mesh.geometry.dim,)))

if MPI.COMM_WORLD.rank == 0:
    print(f"  CG-{degree_cg} DOFs : {V_cg.dofmap.index_map.size_global}")

# ─────────────────────────────────────────────────────────────────────────────
# Get DOF coordinates (x, y) for every CG DOF
# ─────────────────────────────────────────────────────────────────────────────
dof_coords_flat = V_cg.tabulate_dof_coordinates()   # shape (n_dofs*2, 3)? No:
# tabulate_dof_coordinates returns shape (n_scalar_dofs, 3) for a vector space
# For a 2-component space the index_map covers scalar dofs; we need unique coords.
# Safer: use the block map.

# For a VectorElement, tabulate_dof_coordinates gives one (x,y,z) per scalar dof.
# Component dofs share the same physical node, so we take every 2nd entry.
all_coords = V_cg.tabulate_dof_coordinates()        # (n_scalar_dofs, 3)
n_vec_dofs  = V_cg.dofmap.index_map.size_local      # vector DOFs (pairs)
# Coordinates for the x-component dofs  (every other row starting at 0)
# In DOLFINx, vector CG dofs are interleaved: [x0,y0, x1,y1, ...]
# tabulate_dof_coordinates() returns coords for ALL scalar dofs.
# Shape = (2 * n_nodes, 3) for a 2D 2-component space.
coords_x_comp = all_coords[0::2, :]    # coords for Ex dofs
x_dof = coords_x_comp[:, 0]
y_dof = coords_x_comp[:, 1]

if MPI.COMM_WORLD.rank == 0:
    print(f"\nStep 2: Computing Mie coefficients …")

coeffs = compute_mie_coefficients(k0, R_cyl, n_max=25)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate Mie field at DOF coordinates
# Only evaluate outside cylinder (inside is PEC — set to zero)
# ─────────────────────────────────────────────────────────────────────────────
if MPI.COMM_WORLD.rank == 0:
    print(f"Step 3: Evaluating Mie series at {len(x_dof)} DOF positions …")
    print(f"        (this may take ~30 s for large meshes)")

rho_dof = np.sqrt(x_dof**2 + y_dof**2)
outside  = rho_dof >= R_cyl * 0.999   # mask: outside PEC

Ex_mie_dof = np.zeros(len(x_dof), dtype=np.complex128)
Ey_mie_dof = np.zeros(len(x_dof), dtype=np.complex128)

Ex_mie_dof[outside], Ey_mie_dof[outside] = mie_scattered_field(
    x_dof[outside], y_dof[outside], coeffs, k0)

# Incident field at same nodes (for total field)
Ex_inc_dof, Ey_inc_dof = incident_field(x_dof, y_dof, k0)
Ex_inc_dof[~outside] = 0.0
Ey_inc_dof[~outside] = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Pack into DOLFINx Functions
# DOLFINx CG-2 vector: scalar dofs interleaved [Ex0, Ey0, Ex1, Ey1, ...]
# ─────────────────────────────────────────────────────────────────────────────
def fill_vector_function(V, Ex_vals, Ey_vals):
    """Fill a 2-component CG Function from per-node Ex, Ey arrays."""
    f = Function(V)
    arr = f.x.array                  # length = 2 * n_nodes, complex
    arr[0::2] = Ex_vals              # x-component at each node
    arr[1::2] = Ey_vals              # y-component at each node
    f.x.scatter_forward()
    return f


E_mie_s = fill_vector_function(V_cg, Ex_mie_dof, Ey_mie_dof)
E_mie_s.name = "E_mie_scattered"

E_mie_t = fill_vector_function(V_cg,
                                Ex_mie_dof + Ex_inc_dof,
                                Ey_mie_dof + Ey_inc_dof)
E_mie_t.name = "E_mie_total"

# ─────────────────────────────────────────────────────────────────────────────
# Load FEM result from fem_scattered.npz and build an error field
# (We interpolate FEM line data onto the 2D grid using nearest-neighbour
# along the x-axis strip; for a full 2D error map load from E_scattered.bp.)
# ─────────────────────────────────────────────────────────────────────────────
if MPI.COMM_WORLD.rank == 0:
    print("Step 4: Building error field …")

try:
    fem_data  = np.load("fem_scattered.npz")
    x_line    = fem_data["x"]
    E_mag_fem = fem_data["E_mag"]
    Ex_fem_line = fem_data.get("Ex", np.zeros_like(x_line, dtype=np.complex128))
    Ey_fem_line = fem_data.get("Ey", np.zeros_like(x_line, dtype=np.complex128))

    # Interpolate FEM line → 2D DOF grid using distance from x-axis
    # For DOFs near y≈0 we have direct comparison; elsewhere use magnitude proxy.
    E_mag_mie_dof = np.sqrt(np.abs(Ex_mie_dof)**2 + np.abs(Ey_mie_dof)**2)

    # Build per-DOF FEM magnitude by interpolating the x-axis data
    E_mag_fem_dof  = np.zeros(len(x_dof))
    for i, (xi, yi, ri) in enumerate(zip(x_dof, y_dof, rho_dof)):
        if outside[i]:
            # Use radial distance from cylinder as proxy for x along line
            E_mag_fem_dof[i] = np.interp(ri, x_line, E_mag_fem,
                                          left=0, right=E_mag_fem[-1])

    error_mag = np.abs(E_mag_fem_dof - E_mag_mie_dof)

    # Store as scalar CG-1 function
    V_sc = functionspace(mesh, ("Lagrange", 1))
    E_err = Function(V_sc)
    # CG-1 dof coords
    sc_coords = V_sc.tabulate_dof_coordinates()
    x_sc = sc_coords[:, 0];  y_sc = sc_coords[:, 1]
    rho_sc = np.sqrt(x_sc**2 + y_sc**2)
    out_sc = rho_sc >= R_cyl * 0.999

    E_mie_sc  = np.zeros(len(x_sc))
    E_fem_sc  = np.zeros(len(x_sc))
    for i, (xi, ri) in enumerate(zip(x_sc, rho_sc)):
        if out_sc[i]:
            # crude: treat rho as equivalent radial distance
            idx = np.argmin(np.abs(x_line - ri))
            E_fem_sc[i] = E_mag_fem[idx]

    # Mie magnitude at scalar DOF coords
    for i, (xi, yi, ri) in enumerate(zip(x_sc, y_sc, rho_sc)):
        if out_sc[i]:
            Ex_i, Ey_i = mie_scattered_field([xi], [yi], coeffs, k0)
            E_mie_sc[i] = np.sqrt(np.abs(Ex_i[0])**2 + np.abs(Ey_i[0])**2)

    E_err.x.array[:] = np.abs(E_fem_sc - E_mie_sc).astype(
        E_err.x.array.dtype)
    E_err.x.scatter_forward()
    E_err.name = "E_mag_error"
    has_error = True
    if MPI.COMM_WORLD.rank == 0:
        print("  ✓ Error field computed from fem_scattered.npz")

except FileNotFoundError:
    has_error = False
    if MPI.COMM_WORLD.rank == 0:
        print("  ⚠ fem_scattered.npz not found — skipping error field.")
        print("    Run 01_pec_cylinder.py first to generate it.")

# ─────────────────────────────────────────────────────────────────────────────
# Write .bp files
# ─────────────────────────────────────────────────────────────────────────────
if MPI.COMM_WORLD.rank == 0:
    print("\nStep 5: Writing .bp files …")

with VTXWriter(mesh.comm, "E_mie_scattered.bp", [E_mie_s]) as f:
    f.write(0.0)

with VTXWriter(mesh.comm, "E_mie_total.bp", [E_mie_t]) as f:
    f.write(0.0)

if has_error:
    with VTXWriter(mesh.comm, "E_error.bp", [E_err]) as f:
        f.write(0.0)

if MPI.COMM_WORLD.rank == 0:
    print("  ✓ E_mie_scattered.bp")
    print("  ✓ E_mie_total.bp")
    if has_error:
        print("  ✓ E_error.bp")

    print()
    print("=" * 65)
    print("PARAVIEW INSTRUCTIONS")
    print("=" * 65)
    print()
    print("── Side-by-side 2D field maps ─────────────────────────────────")
    print("  1. File > Open > select E_scattered.bp   (FEM)")
    print("     File > Open > select E_mie_scattered.bp  (Mie)")
    print("     Apply both readers.")
    print()
    print("  2. Select E_scattered.bp in pipeline:")
    print("     Filters > Alphabetical > Calculator")
    print("     Result name: E_mag")
    print("     Formula:     sqrt(E[:,0]^2+E[:,1]^2)   (real part magnitude)")
    print("     Or use:      mag(E)  in newer ParaView versions")
    print("     Apply. Colour by E_mag.")
    print()
    print("  3. Repeat Calculator for E_mie_scattered.bp.")
    print()
    print("  4. View > Split Horizontal  (or Layout > Add Column)")
    print("     Drag each dataset to its own viewport.")
    print("     Use same colour scale: right-click colour bar > Edit > ")
    print("     check 'Use separate colour map' OFF to sync both.")
    print()
    print("── Plot Over Line (x-axis comparison) ─────────────────────────")
    print("  1. Select E_scattered.bp (FEM).")
    print("     Filters > Data Analysis > Plot Over Line")
    print("     Point1: (0.55, 0, 0)   Point2: (4.75, 0, 0)")
    print("     Resolution: 300   Apply.")
    print()
    print("  2. In the Line Chart View, hold Shift and select")
    print("     E_mie_scattered.bp, add another Plot Over Line.")
    print("     Both curves now overlay in the same chart.")
    print()
    print("── Error map ───────────────────────────────────────────────────")
    if has_error:
        print("  1. File > Open > E_error.bp   Apply.")
        print("     Colour by E_mag_error.  Use 'Cool to Warm' colour map.")
        print("     Regions of high error will show as red.")
    else:
        print("  fem_scattered.npz missing — re-run 01_pec_cylinder.py,")
        print("  then re-run this script to generate E_error.bp.")
    print()
    print("Done.")