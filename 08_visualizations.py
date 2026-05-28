"""
07_visual_comparison.py  —  FEM vs Mie Publication-Quality Comparison Plots
=============================================================================
Generates a four-panel comparison figure from the .npz files already on disk.
No FEniCSx required — pure NumPy/SciPy/Matplotlib.

Input files (must be in same directory)
----------------------------------------
  fem_scattered.npz          from 01_pec_cylinder.py
  analytical_reference.npz   from 02_math_analytical.py  (optional)

Output
------
  fem_vs_mie_comparison.png   high-resolution four-panel figure

Usage
-----
  python 07_visual_comparison.py

Panels
------
  A  |E_scattered| along x-axis  — FEM vs Mie overlay + error band
  B  Re(Ey) along x-axis         — FEM vs Mie overlay
  C  Pointwise relative error (%) vs x
  D  L2 error summary as function of distance from cylinder
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import jv, hankel2


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
wl    = 1.0
k0    = 2 * np.pi / wl
R_cyl = 0.5 * wl
R_out = 5.0 * wl

FIGSIZE  = (14, 10)
DPI      = 250
FONTSIZE = 12


# ─────────────────────────────────────────────────────────────────────────────
# Mie series (self-contained, corrected TE formulation)
# ─────────────────────────────────────────────────────────────────────────────

def jv_prime(n, z):
    return 0.5 * (jv(n - 1, z) - jv(n + 1, z))

def h2_prime(n, z):
    return 0.5 * (hankel2(n - 1, z) - hankel2(n + 1, z))


def compute_mie_on_line(x_pts, k0=k0, R_cyl=R_cyl, n_max=25):
    """
    Evaluate Mie scattered field along y=0 line.
    Returns Ex, Ey (complex), E_mag (real).
    """
    ka = k0 * R_cyl
    coeffs = {n: -jv_prime(n, ka) / h2_prime(n, ka)
              for n in range(-n_max, n_max + 1)}

    Ex = np.zeros(len(x_pts), dtype=np.complex128)
    Ey = np.zeros(len(x_pts), dtype=np.complex128)

    for i, x in enumerate(x_pts):
        rho = abs(x)
        phi = 0.0 if x >= 0 else np.pi   # y=0 line

        E_rho = 0j;  E_phi = 0j
        for n, a_n in coeffs.items():
            if abs(a_n) < 1e-15:
                continue
            H_n  = hankel2(n, k0 * rho)
            Hp_n = h2_prime(n, k0 * rho)
            phase = ((-1j)**n) * np.exp(1j * n * phi)
            if n != 0:
                E_rho += a_n * (1j * n / (k0 * rho)) * H_n * phase
            E_phi += a_n * Hp_n * phase

        Ex[i] = E_rho * np.cos(phi) - E_phi * np.sin(phi)
        Ey[i] = E_rho * np.sin(phi) + E_phi * np.cos(phi)

    return Ex, Ey, np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)


# ─────────────────────────────────────────────────────────────────────────────
# Load FEM data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading FEM data …")
try:
    fem = np.load("fem_scattered.npz")
    x_fem    = fem["x"]
    E_mag_fem = fem["E_mag"]
    # Try to get complex components; fall back gracefully
    Ex_fem = fem["Ex"] if "Ex" in fem else np.zeros_like(x_fem, dtype=np.complex128)
    Ey_fem = fem["Ey"] if "Ey" in fem else np.zeros_like(x_fem, dtype=np.complex128)
    print(f"  ✓ fem_scattered.npz  —  {len(x_fem)} points")
    print(f"    x range: [{x_fem[0]:.3f}, {x_fem[-1]:.3f}] m")
    print(f"    |E| range: [{E_mag_fem.min():.3e}, {E_mag_fem.max():.3e}]")
except FileNotFoundError:
    print("  ✗ fem_scattered.npz not found.")
    print("    Run 01_pec_cylinder.py first, then re-run this script.")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Compute Mie on same x-grid as FEM
# ─────────────────────────────────────────────────────────────────────────────
print("Computing Mie series on FEM x-grid …")
Ex_mie, Ey_mie, E_mag_mie = compute_mie_on_line(x_fem)
print(f"  ✓ Mie computed  —  |E| range: [{E_mag_mie.min():.3e}, {E_mag_mie.max():.3e}]")


# ─────────────────────────────────────────────────────────────────────────────
# Error metrics
# ─────────────────────────────────────────────────────────────────────────────
eps     = 1e-30
rel_err = np.abs(E_mag_fem - E_mag_mie) / (E_mag_mie + eps) * 100   # %

l2_num  = np.sqrt(np.trapz((E_mag_fem - E_mag_mie)**2, x_fem))
l2_den  = np.sqrt(np.trapz(E_mag_mie**2, x_fem))
l2_err  = l2_num / l2_den * 100   # %

mare    = np.mean(rel_err)
linf    = np.max(rel_err)
linf_x  = x_fem[np.argmax(rel_err)]

near_mask = x_fem < 2.0 * wl
far_mask  = x_fem > 3.0 * wl
near_err  = np.mean(rel_err[near_mask]) if near_mask.any() else np.nan
far_err   = np.mean(rel_err[far_mask])  if far_mask.any()  else np.nan

print()
print("─" * 50)
print("ERROR SUMMARY")
print("─" * 50)
print(f"  L2 relative error   : {l2_err:.3f}%")
print(f"  L-inf (max) error   : {linf:.3f}%  at x = {linf_x:.3f} m")
print(f"  Mean abs rel error  : {mare:.3f}%")
print(f"  Near-field (r<2λ)   : {near_err:.3f}%")
print(f"  Far-field  (r>3λ)   : {far_err:.3f}%")
print("─" * 50)

if l2_err < 1.0:
    quality = "EXCELLENT  (<1%)"
elif l2_err < 5.0:
    quality = "GOOD       (<5%)"
elif l2_err < 10.0:
    quality = "MODERATE   (<10%)"
else:
    quality = "POOR       (>10%) — check mesh or BC"
print(f"  Assessment : {quality}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=FIGSIZE, facecolor="white")
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32,
                        left=0.09, right=0.97, top=0.93, bottom=0.08)

# Colour palette
C_fem = "#1565C0"   # deep blue
C_mie = "#C62828"   # deep red
C_err = "#6A1B9A"   # purple

# ── Panel A: |E| magnitude overlay ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])

ax.plot(x_fem, E_mag_mie, color=C_mie, lw=2.2, label="Mie (analytical)", zorder=3)
ax.plot(x_fem, E_mag_fem, color=C_fem, lw=1.8, ls="--", label="FEM (DOLFINx)", zorder=4)

# ±5% error band around Mie
band = 0.05 * E_mag_mie
ax.fill_between(x_fem, E_mag_mie - band, E_mag_mie + band,
                alpha=0.15, color=C_mie, label="±5% band")

ax.axvline(R_cyl, color="k", ls=":", lw=1.2, alpha=0.6)
ax.text(R_cyl + 0.05, ax.get_ylim()[1] * 0.92, "PEC surface",
        fontsize=9, color="k", alpha=0.7)

ax.set_xlabel("x [m]", fontsize=FONTSIZE)
ax.set_ylabel("|E_scattered| [V/m]", fontsize=FONTSIZE)
ax.set_title("(A)  Scattered field magnitude", fontsize=FONTSIZE, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.25, ls=":")
ax.set_xlim([x_fem[0], x_fem[-1]])

# ── Panel B: Re(Ey) overlay ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])

ax.plot(x_fem, Ey_mie.real, color=C_mie, lw=2.2, label="Mie  Re(E_y)")
ax.plot(x_fem, Ey_fem.real, color=C_fem, lw=1.8, ls="--", label="FEM  Re(E_y)")
ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.4)
ax.axvline(R_cyl, color="k", ls=":", lw=1.2, alpha=0.6)

ax.set_xlabel("x [m]", fontsize=FONTSIZE)
ax.set_ylabel("Re(E_y) [V/m]", fontsize=FONTSIZE)
ax.set_title("(B)  Real part of E_y component", fontsize=FONTSIZE, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25, ls=":")
ax.set_xlim([x_fem[0], x_fem[-1]])

# ── Panel C: Pointwise relative error ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])

ax.plot(x_fem, rel_err, color=C_err, lw=1.8)
ax.axhline(1.0, color="#2E7D32", ls="--", lw=1.3, alpha=0.8, label="1% threshold")
ax.axhline(5.0, color="#E65100", ls="--", lw=1.3, alpha=0.8, label="5% threshold")
ax.fill_between(x_fem, 0, np.minimum(rel_err, 1.0),
                alpha=0.15, color="#2E7D32", label="<1% zone")
ax.axvline(R_cyl, color="k", ls=":", lw=1.2, alpha=0.6)

# Annotate max error
ax.annotate(f"max {linf:.2f}%\n@ x={linf_x:.2f} m",
            xy=(linf_x, linf), xytext=(linf_x + 0.4, linf * 0.85),
            fontsize=9, color=C_err,
            arrowprops=dict(arrowstyle="->", color=C_err, lw=1.2))

ax.set_xlabel("x [m]", fontsize=FONTSIZE)
ax.set_ylabel("Relative error [%]", fontsize=FONTSIZE)
ax.set_title("(C)  Pointwise relative error |FEM−Mie|/|Mie|",
             fontsize=FONTSIZE, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.25, ls=":")
ax.set_xlim([x_fem[0], x_fem[-1]])
ax.set_ylim(bottom=0)

# ── Panel D: Error summary bar chart ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])

labels = ["L2\nglobal", "L-inf\n(max)", "Near-field\n(r<2λ)", "Far-field\n(r>3λ)"]
values = [l2_err, linf, near_err, far_err]
bar_colors = ["#1565C0" if v < 1 else "#E65100" if v < 5 else "#B71C1C"
              for v in values]
bars = ax.bar(labels, values, color=bar_colors, edgecolor="white",
              linewidth=0.8, width=0.55)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f"{val:.2f}%", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold")

ax.axhline(1.0, color="#2E7D32", ls="--", lw=1.3, alpha=0.8, label="1%")
ax.axhline(5.0, color="#E65100", ls="--", lw=1.3, alpha=0.8, label="5%")
ax.set_ylabel("Relative error [%]", fontsize=FONTSIZE)
ax.set_title("(D)  Error summary by region", fontsize=FONTSIZE, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.25, ls=":", axis="y")
ax.set_ylim(bottom=0, top=max(values) * 1.35)

# ── Super-title ──────────────────────────────────────────────────────────────
fig.suptitle(
    f"FEM (DOLFINx N1curl-3) vs Mie Series  —  PEC Cylinder  "
    f"ka = {k0*R_cyl:.3f}   L2 error = {l2_err:.2f}%",
    fontsize=13, fontweight="bold", y=0.98)

outfile = "fem_vs_mie_comparison.png"
plt.savefig(outfile, dpi=DPI, bbox_inches="tight")
print(f"✓ Saved: {outfile}")
plt.show()