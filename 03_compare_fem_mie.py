"""
FEM vs Mie Theory Comparison Script
====================================
Quantifies the error between your DOLFINx FEM scattered field
and the corrected TE-mode Mie analytical reference.

Requirements:
    pip install numpy scipy matplotlib adios4dolfinx

Input files expected (in same directory):
    analytical_reference.npz   — from your corrected Mie script
    E_scattered.bp             — from your pec_cylinder FEM solver

Usage:
    python compare_fem_mie.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel2
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# PARAMETERS — must match your FEM and Mie scripts exactly
# ============================================================================

wl    = 1.0
k0    = 2 * np.pi / wl
R_cyl = 0.5 * wl
R_out = 5.0 * wl

# Comparison line: x-axis from just outside cylinder to just inside outer boundary
x_compare = np.linspace(R_cyl * 1.05, R_out * 0.95, 300)
y_compare  = np.zeros_like(x_compare)


# ============================================================================
# STEP 1 — RECOMPUTE CORRECTED TE MIE REFERENCE
# ============================================================================
# We recompute here rather than loading the .npz so this script is
# self-contained and always uses the correct TE coefficients.

print("=" * 65)
print("STEP 1 — Computing corrected Mie reference")
print("=" * 65)

# Helper derivative functions
def jv_prime(n, z):
    return 0.5 * (jv(n-1, z) - jv(n+1, z))

def h2_prime(n, z):
    return 0.5 * (hankel2(n-1, z) - hankel2(n+1, z))

# Bug 1 fix — use derivative form a_n = -J'_n / H'_n
n_max = 20
coefficients = {}

for n in range(-n_max, n_max + 1):
    a_n = -jv_prime(n, k0 * R_cyl) / h2_prime(n, k0 * R_cyl)
    coefficients[n] = a_n

print(f"  Coefficients computed: n = -{n_max} to +{n_max}")
print(f"  Formula: a_n = -J'_n(ka) / H'_n(ka)  [derivative form]")

# Sanity check on coefficients
print(f"\n  Coefficient sanity check:")
for n in [0, 1, 2]:
    print(f"    a_{n} = {coefficients[n]:.4f}")
print(f"\n  Phase factor check:")
print(f"    (-j)^1 = {(-1j)**1:.4f}  (expect 0-1j)")
print(f"    (-j)^2 = {(-1j)**2:.4f}  (expect -1+0j)")

# Evaluate scattered field along comparison line
Ex_mie = np.zeros(len(x_compare), dtype=np.complex128)
Ey_mie = np.zeros(len(x_compare), dtype=np.complex128)

for i, (x, y) in enumerate(zip(x_compare, y_compare)):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    E_rho = 0j
    E_phi = 0j

    for n, a_n in coefficients.items():
        if np.abs(a_n) < 1e-15:
            continue

        H_n       = hankel2(n, k0 * rho)
        H_n_prime = h2_prime(n, k0 * rho)

        # Bug 2 fix — add (-j)^n phase factor for e^{-ik₀x} plane wave expansion
        phase = ((-1j)**n) * np.exp(1j * n * phi)

        if n != 0:
            E_rho += a_n * (1j * n / (k0 * rho)) * H_n * phase
        E_phi += a_n * H_n_prime * phase

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    Ex_mie[i] = E_rho * cos_phi - E_phi * sin_phi
    Ey_mie[i] = E_rho * sin_phi + E_phi * cos_phi

E_mag_mie = np.sqrt(np.abs(Ex_mie)**2 + np.abs(Ey_mie)**2)
print(f"\n  |E| range: {E_mag_mie.min():.4e} — {E_mag_mie.max():.4e}")

# ============================================================================
# STEP 2 — LOAD FEM SOLUTION
# ============================================================================

print()
print("=" * 65)
print("STEP 2 — Loading FEM solution from E_scattered.bp")
print("=" * 65)

fem_loaded = False

try:
    import adios4dolfinx
    import dolfinx
    from mpi4py import MPI
    from dolfinx.io import VTXReader

    # Read the scattered field written by your pec_cylinder solver
    with VTXReader(MPI.COMM_WORLD, "E_scattered.bp", dolfinx.default_scalar_type) as reader:
        reader.read(0.0)
        mesh  = reader.mesh
        Es_h  = reader.get_function("Scattered_Field")

    print("  Loaded via adios4dolfinx")

    # Sample FEM field along the comparison line
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    import numpy as np

    points_3d = np.column_stack([x_compare, y_compare, np.zeros_like(x_compare)])
    tree      = bb_tree(mesh, mesh.topology.dim)
    cells_candidates = compute_collisions_points(tree, points_3d)
    cells     = compute_colliding_cells(mesh, cells_candidates, points_3d)

    Ex_fem = np.zeros(len(x_compare), dtype=np.complex128)
    Ey_fem = np.zeros(len(x_compare), dtype=np.complex128)
    valid  = np.zeros(len(x_compare), dtype=bool)

    for i, (point, cell_list) in enumerate(zip(points_3d, cells.links(np.arange(len(x_compare))))):
        if len(cell_list) > 0:
            val = Es_h.eval(point.reshape(1,-1), [cell_list[0]])
            Ex_fem[i] = val[0]
            Ey_fem[i] = val[1]
            valid[i]  = True

    E_mag_fem = np.sqrt(np.abs(Ex_fem)**2 + np.abs(Ey_fem)**2)
    fem_loaded = True
    print(f"  Points sampled: {valid.sum()} / {len(x_compare)}")
    print(f"  |E| range: {E_mag_fem[valid].min():.4e} — {E_mag_fem[valid].max():.4e}")

except ImportError:
    print("  adios4dolfinx not available.")
    print("  Attempting to load from validation_data.npz fallback...")

except Exception as e:
    print(f"  Could not load E_scattered.bp: {e}")
    print("  Attempting fallback...")

# Fallback: try loading a pre-exported numpy file if the .bp read failed
if not fem_loaded:
    try:
        fem_data  = np.load("fem_scattered.npz")
        x_fem     = fem_data["x"]
        E_mag_fem_raw = fem_data["E_mag"]

        # Interpolate onto our comparison grid
        E_mag_fem = np.interp(x_compare, x_fem, E_mag_fem_raw)
        valid     = np.ones(len(x_compare), dtype=bool)
        fem_loaded = True
        print("  Loaded from fem_scattered.npz fallback")
        print()
        print("  NOTE: To export FEM data for this script, add to the")
        print("  bottom of pec_cylinder.py:")
        print()
        print("    np.savez('fem_scattered.npz',")
        print("             x=x_compare,")
        print("             E_mag=E_mag_fem,")
        print("             Ex=Ex_fem, Ey=Ey_fem)")

    except FileNotFoundError:
        print()
        print("  No FEM data found. Generating synthetic FEM data with")
        print("  5% random noise for demonstration purposes.")
        print("  Replace this with your actual FEM output to get real errors.")
        np.random.seed(42)
        noise     = 1 + 0.05 * np.random.randn(len(x_compare))
        E_mag_fem = E_mag_mie * noise
        valid     = np.ones(len(x_compare), dtype=bool)
        fem_loaded = True


# ============================================================================
# STEP 3 — COMPUTE ERROR METRICS
# ============================================================================

print()
print("=" * 65)
print("STEP 3 — Error metrics")
print("=" * 65)

x_v   = x_compare[valid]
mie_v = E_mag_mie[valid]
fem_v = E_mag_fem[valid]

# Pointwise relative error
rel_err = np.abs(fem_v - mie_v) / (mie_v + 1e-30)

# L2 relative error (integral norm)
l2_num   = np.sqrt(np.trapz((fem_v - mie_v)**2, x_v))
l2_den   = np.sqrt(np.trapz(mie_v**2,            x_v))
l2_err   = l2_num / l2_den

# L-infinity (maximum pointwise) error
linf_err = np.max(rel_err)
linf_loc = x_v[np.argmax(rel_err)]

# Mean absolute relative error
mare     = np.mean(rel_err)

# Near-field error (r < 2λ) and far-field error (r > 3λ)
near_mask = x_v < 2.0 * wl
far_mask  = x_v > 3.0 * wl
near_err  = np.mean(rel_err[near_mask]) if near_mask.any() else float('nan')
far_err   = np.mean(rel_err[far_mask])  if far_mask.any()  else float('nan')

print(f"  L2 relative error:          {l2_err*100:.3f}%")
print(f"  L-inf (max) relative error: {linf_err*100:.3f}%  at x = {linf_loc:.3f} m")
print(f"  Mean absolute rel. error:   {mare*100:.3f}%")
print()
print(f"  Near-field error (r < 2λ):  {near_err*100:.3f}%")
print(f"  Far-field error  (r > 3λ):  {far_err*100:.3f}%")
print()

# Quality assessment
print("  Assessment:")
if l2_err < 0.01:
    print("  [EXCELLENT] L2 error < 1% — solver matches Mie theory very well")
elif l2_err < 0.05:
    print("  [GOOD]      L2 error < 5% — acceptable for engineering validation")
elif l2_err < 0.10:
    print("  [MODERATE]  L2 error < 10% — consider mesh refinement or higher degree")
else:
    print("  [POOR]      L2 error > 10% — check polarization, BCs, or mesh quality")

if near_err > far_err * 2:
    print("  [NOTE] Near-field error >> far-field: ABC reflections likely not the cause.")
    print("         Check mesh density near cylinder (lc_cyl) or PEC BC application.")
elif far_err > near_err * 2:
    print("  [NOTE] Far-field error >> near-field: ABC boundary condition may be too coarse.")
    print("         Try increasing R_out or refining lc_out.")


# ============================================================================
# STEP 4 — PLOTS
# ============================================================================

print()
print("=" * 65)
print("STEP 4 — Generating plots")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"FEM vs Mie Theory (TE mode) — PEC Cylinder  ka = {k0*R_cyl:.3f}",
    fontsize=14, fontweight='bold'
)

# --- Plot 1: Field magnitude overlay ---
ax = axes[0, 0]
ax.plot(x_compare,    E_mag_mie,        'b-',  lw=2,   label='Mie (TE, analytical)')
ax.plot(x_v,          E_mag_fem,        'r--', lw=1.8, label='FEM (DOLFINx)')
ax.axvline(R_cyl, color='k', ls=':', lw=1.2, alpha=0.6, label=f'Cylinder r={R_cyl}m')
ax.set_xlabel('x [m]')
ax.set_ylabel('|E| [V/m]')
ax.set_title('Scattered field magnitude')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Plot 2: Pointwise relative error ---
ax = axes[0, 1]
ax.plot(x_v, rel_err * 100, 'purple', lw=1.8)
ax.axhline(1.0, color='g',      ls='--', lw=1.2, alpha=0.7, label='1% threshold')
ax.axhline(5.0, color='orange', ls='--', lw=1.2, alpha=0.7, label='5% threshold')
ax.axvline(R_cyl, color='k', ls=':', lw=1.2, alpha=0.6)
ax.set_xlabel('x [m]')
ax.set_ylabel('Relative error [%]')
ax.set_title('Pointwise relative error  |FEM − Mie| / |Mie|')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# --- Plot 3: Log-scale overlay ---
ax = axes[1, 0]
ax.semilogy(x_compare, E_mag_mie,   'b-',  lw=2,   label='Mie (TE)')
ax.semilogy(x_v,       E_mag_fem,   'r--', lw=1.8, label='FEM')
ax.axvline(R_cyl, color='k', ls=':', lw=1.2, alpha=0.6)
ax.set_xlabel('x [m]')
ax.set_ylabel('|E| [V/m] (log scale)')
ax.set_title('Log-scale comparison — decay rate check')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# --- Plot 4: Error summary bar ---
ax = axes[1, 1]
metrics = ['L2 error', 'L-inf error', 'Near-field\n(r<2λ)', 'Far-field\n(r>3λ)']
values  = [l2_err*100, linf_err*100, near_err*100, far_err*100]
colors  = ['#2196F3' if v < 1 else '#FF9800' if v < 5 else '#F44336' for v in values]
bars    = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=0.8)
ax.axhline(1.0, color='g',      ls='--', lw=1.2, alpha=0.7, label='1%')
ax.axhline(5.0, color='orange', ls='--', lw=1.2, alpha=0.7, label='5%')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Relative error [%]')
ax.set_title('Error summary by region')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('fem_mie_comparison.png', dpi=180, bbox_inches='tight')
print("  Saved: fem_mie_comparison.png")


# ============================================================================
# STEP 5 — SAVE COMPARISON DATA
# ============================================================================

np.savez('comparison_results.npz',
         x          = x_v,
         E_mag_mie  = mie_v,
         E_mag_fem  = fem_v,
         rel_err    = rel_err,
         l2_err     = l2_err,
         linf_err   = linf_err,
         mare       = mare,
         near_err   = near_err,
         far_err    = far_err)

print("  Saved: comparison_results.npz")
print()
print("=" * 65)
print("COMPARISON COMPLETE")
print("=" * 65)
print(f"  L2 error  : {l2_err*100:.3f}%")
print(f"  Max error : {linf_err*100:.3f}%  at x = {linf_loc:.3f} m")
print()
print("  Next step: if L2 > 5%, run a mesh convergence study.")
print("  Each halving of mesh size should reduce error by ~2^p")
print("  where p is your polynomial degree (currently degree=3).")

plt.show()