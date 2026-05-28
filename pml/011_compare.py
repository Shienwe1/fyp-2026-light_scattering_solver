"""
011_compare.py — Mie Reference & FEM vs Mie Comparison
========================================================
Computes the exact 2D Mie series solution for PEC cylinder scattering,
quantifies error against the FEM result from 010_main_solver.py, and
exports the Mie field onto the FEM mesh for ParaView visual comparison.

Usage
-----
    # Step 1 — run the FEM solver first
    python 010_main_solver.py

    # Step 2 — compare + generate Mie .bp for ParaView
    python 011_compare.py

Outputs
-------
    fem_mie_comparison.png   — four-panel error figure
    comparison_results.npz   — error metrics and field data
    E_mie.bp                 — Mie field on FEM mesh; open alongside
                               E_total.bp in ParaView for visual comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel2
import importlib.util, sys, pathlib

# ── Load 010_main_solver.py via importlib (filename starts with digit) ────────

_spec = importlib.util.spec_from_file_location(
    "solver",
    pathlib.Path(__file__).parent / "010_main_solver.py"
)
_solver = importlib.util.module_from_spec(_spec)
sys.modules["solver"] = _solver
_spec.loader.exec_module(_solver)

MIE_PARAMS         = _solver.MIE_PARAMS
solve_pec_cylinder = _solver.solve_pec_cylinder


# ── Bessel / Hankel derivative helpers ───────────────────────────────────────

def _jv_prime(n, z):
    return 0.5 * (jv(n - 1, z) - jv(n + 1, z))

def _h2_prime(n, z):
    return 0.5 * (hankel2(n - 1, z) - hankel2(n + 1, z))


# ── Mie series (vectorised) ───────────────────────────────────────────────────

def compute_mie(x_pts, y_pts, params=None, n_max=20):
    """
    Evaluate the 2D Mie scattered field at arbitrary (x, y) points.

    Fully vectorised — safe to call with the full mesh node array
    (tens of thousands of points) without a Python loop overhead.

    Formulation
    -----------
    Mie coefficients:  a_n = −J'_n(k₀R) / H'_n(k₀R)
    Phase factor:      (−j)^n  for e^{−ik₀x} plane wave expansion
    Radial component:  E_ρ = Σ  a_n · (jn / k₀ρ) · H_n(k₀ρ) · phase
    Azimuthal:         E_φ = Σ  a_n · H'_n(k₀ρ) · phase
    Cartesian:         E_x = E_ρ cosφ − E_φ sinφ
                       E_y = E_ρ sinφ + E_φ cosφ

    Parameters
    ----------
    x_pts, y_pts : array-like, shape (N,)
        Evaluation coordinates [m]. Can be full mesh node coordinates.
    params : dict, optional
        Override any key in MIE_PARAMS.
    n_max : int
        Series truncation order (terms n = −n_max … +n_max).

    Returns
    -------
    Ex, Ey : complex128 arrays, shape (N,)
    """
    p  = {**MIE_PARAMS, **(params or {})}
    k0 = 2 * np.pi / p["wl"]
    R  = p["R_cyl"]

    x_pts = np.asarray(x_pts, dtype=np.float64)
    y_pts = np.asarray(y_pts, dtype=np.float64)

    rho = np.sqrt(x_pts**2 + y_pts**2)
    phi = np.arctan2(y_pts, x_pts)
    rho = np.where(rho < 1e-14, 1e-14, rho)  # avoid division by zero

    E_rho = np.zeros(len(x_pts), dtype=np.complex128)
    E_phi = np.zeros(len(x_pts), dtype=np.complex128)

    for n in range(-n_max, n_max + 1):
        a_n = -_jv_prime(n, k0 * R) / _h2_prime(n, k0 * R)
        if abs(a_n) < 1e-15:
            continue

        H_n   = hankel2(n, k0 * rho)
        Hp_n  = _h2_prime(n, k0 * rho)
        phase = ((-1j) ** n) * np.exp(1j * n * phi)

        if n != 0:
            E_rho += a_n * (n / (k0 * rho)) * H_n * phase  # 1j removed — E_rho from φ-derivative
        E_phi += a_n * 1j * Hp_n * phase                   # 1j added  — curl operator contribution

    Ex = E_rho * np.cos(phi) - E_phi * np.sin(phi)
    Ey = E_rho * np.sin(phi) + E_phi * np.cos(phi)

    return Ex, Ey


# ── Write Mie field to .bp (ParaView) ─────────────────────────────────────────

def export_mie_to_bp(mesh, params=None, filename="E_mie.bp"):
    """
    Evaluate the Mie scattered field at every mesh node and write a .bp
    file readable by ParaView.

    Written as a single complex Function so DOLFINx splits it into
    E_mie_real and E_mie_imag only — matching the f_real / f_imag
    convention of E_total.bp for direct side-by-side comparison.

    Parameters
    ----------
    mesh     : dolfinx Mesh — the FEM mesh from 010_main_solver.py
    params   : dict, optional — override MIE_PARAMS
    filename : str — output path (default "E_mie.bp")

    Returns
    -------
    E_mie : dolfinx Function (complex Lagrange degree-1 vector)
    """
    from dolfinx.fem import functionspace, Function
    from dolfinx.io  import VTXWriter

    p = {**MIE_PARAMS, **(params or {})}

    x_coords = mesh.geometry.x[:, 0]
    y_coords = mesh.geometry.x[:, 1]

    print(f"  Evaluating Mie series at {len(x_coords)} mesh nodes...")
    Ex_mie, Ey_mie = compute_mie(x_coords, y_coords, params=p)

    # Single complex Function — DOLFINx splits into _real / _imag only
    # (two parts, matching E_scattered.bp — scattered field only, no incident)
    V_cg = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    E_mie = Function(V_cg, dtype=np.complex128)
    E_mie.name = "E_mie"
    E_mie.x.array[0::2] = Ex_mie
    E_mie.x.array[1::2] = Ey_mie

    with VTXWriter(mesh.comm, filename, [E_mie]) as f:
        f.write(0.0)

    print(f"  Saved: {filename}")
    print(f"  In ParaView: open alongside E_scattered.bp")
    print(f"  Calculator filter on both:  sqrt(f_0^2 + f_1^2)")

    return E_mie


# ── Error metrics ─────────────────────────────────────────────────────────────

def compute_errors(x_pts, E_mag_fem, E_mag_mie, wl):
    """
    Return a dict of error metrics between FEM and Mie field magnitudes.

    Keys
    ----
    rel_err  : pointwise relative error array
    l2       : L2 relative error (integral norm)
    linf     : maximum pointwise relative error
    linf_x   : x-location of maximum error
    mare     : mean absolute relative error
    near_err : mean relative error for r < 2λ
    far_err  : mean relative error for r > 2.5λ
    """
    rel    = np.abs(E_mag_fem - E_mag_mie) / (E_mag_mie + 1e-30)
    l2     = (np.sqrt(np.trapz((E_mag_fem - E_mag_mie)**2, x_pts))
              / np.sqrt(np.trapz(E_mag_mie**2, x_pts)))
    linf   = float(np.max(rel))
    linf_x = float(x_pts[np.argmax(rel)])
    mare   = float(np.mean(rel))

    near = x_pts < 2.0 * wl
    far  = x_pts > 2.5 * wl  # 2.5λ safely fits inside l_dom/2 = 3.0

    return dict(
        rel_err  = rel,
        l2       = float(l2),
        linf     = linf,
        linf_x   = linf_x,
        mare     = mare,
        near_err = float(np.mean(rel[near])) if near.any() else float("nan"),
        far_err  = float(np.mean(rel[far]))  if far.any()  else float("nan"),
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(x_pts, E_mag_mie, E_mag_fem, errors, R_cyl, wl, ka):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"FEM vs Mie Theory — PEC Cylinder  ka = {ka:.3f}",
        fontsize=14, fontweight="bold",
    )
    e = errors

    ax = axes[0, 0]
    ax.plot(x_pts, E_mag_mie, "b-",  lw=2,   label="Mie (analytical)")
    ax.plot(x_pts, E_mag_fem, "r--", lw=1.8, label="FEM (DOLFINx)")
    ax.axvline(R_cyl, color="k", ls=":", lw=1.2, alpha=0.6, label="r = R_cyl")
    ax.set(xlabel="x [m]", ylabel="|E| [V/m]", title="Scattered field magnitude")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x_pts, e["rel_err"] * 100, color="purple", lw=1.8)
    ax.axhline(1.0, color="g",      ls="--", lw=1.2, alpha=0.7, label="1%")
    ax.axhline(5.0, color="orange", ls="--", lw=1.2, alpha=0.7, label="5%")
    ax.axvline(R_cyl, color="k", ls=":", lw=1.2, alpha=0.6)
    ax.set(xlabel="x [m]", ylabel="Relative error [%]",
           title="|FEM − Mie| / |Mie|", ylim=(0, None))
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(x_pts, E_mag_mie, "b-",  lw=2,   label="Mie")
    ax.semilogy(x_pts, E_mag_fem, "r--", lw=1.8, label="FEM")
    ax.axvline(R_cyl, color="k", ls=":", lw=1.2, alpha=0.6)
    ax.set(xlabel="x [m]", ylabel="|E| [V/m] (log)", title="Log-scale — decay rate")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 1]
    labels = ["L2", "L-inf", "Near\n(r<2λ)", "Far\n(r>2.5λ)"]
    values = [e["l2"]*100, e["linf"]*100, e["near_err"]*100, e["far_err"]*100]
    colors = ["#2196F3" if v < 1 else "#FF9800" if v < 5 else "#F44336"
              for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    ax.axhline(1.0, color="g",      ls="--", lw=1.2, alpha=0.7, label="1%")
    ax.axhline(5.0, color="orange", ls="--", lw=1.2, alpha=0.7, label="5%")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set(ylabel="Relative error [%]", title="Error summary", ylim=(0, None))
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fem_mie_comparison.png", dpi=180, bbox_inches="tight")
    print("  Saved: fem_mie_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    p     = MIE_PARAMS
    wl    = p["wl"]
    k0    = 2 * np.pi / wl
    R_cyl = p["R_cyl"]
    l_dom = p["l_dom"]
    ka    = k0 * R_cyl

    x_pts = np.linspace(R_cyl * 1.05, (l_dom / 2) * 0.95, 300)
    y_pts = np.zeros_like(x_pts)

    # ── Mie reference ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Computing Mie reference...")
    Ex_mie, Ey_mie = compute_mie(x_pts, y_pts)
    E_mag_mie = np.sqrt(np.abs(Ex_mie)**2 + np.abs(Ey_mie)**2)
    print(f"  ka = {ka:.4f}  |  |E| range: "
          f"{E_mag_mie.min():.3e} – {E_mag_mie.max():.3e}")

    # ── FEM solution ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Loading FEM solution from fem_scattered.npz...")
    try:
        fem       = np.load("fem_scattered.npz")
        E_mag_fem = np.interp(x_pts, fem["x"], fem["E_mag"])
        print(f"  Loaded {len(fem['x'])} FEM points")
        print(f"  |E| range: {E_mag_fem.min():.3e} – {E_mag_fem.max():.3e}")
    except FileNotFoundError:
        print("  fem_scattered.npz not found — run 010_main_solver.py first.")
        raise

    # ── Error metrics ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Error metrics:")
    errors = compute_errors(x_pts, E_mag_fem, E_mag_mie, wl)

    print(f"  L2 relative error  : {errors['l2']*100:.3f}%")
    print(f"  L-inf (max) error  : {errors['linf']*100:.3f}%"
          f"  at x = {errors['linf_x']:.3f} m")
    print(f"  Near-field (r<2λ)  : {errors['near_err']*100:.3f}%")
    print(f"  Far-field  (r>2.5λ) : {errors['far_err']*100:.3f}%")

    l2 = errors["l2"]
    if l2 < 0.01:
        verdict = "EXCELLENT  — L2 < 1%, matches Mie theory very well"
    elif l2 < 0.05:
        verdict = "GOOD       — L2 < 5%, acceptable for engineering validation"
    elif l2 < 0.10:
        verdict = "MODERATE   — L2 < 10%, consider mesh refinement"
    else:
        verdict = "POOR       — L2 > 10%, check polarisation, BCs, or mesh"
    print(f"\n  Assessment: {verdict}")

    if errors["near_err"] > errors["far_err"] * 2:
        print("  Near-field error >> far-field: check lc_cyl or PEC BC.")
    elif errors["far_err"] > errors["near_err"] * 2:
        print("  Far-field error >> near-field: ABC boundary may be too coarse.")

    print()
    print("=" * 60)
    print("ERROR METRICS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'L2 relative error':<30} {errors['l2']*100:>9.3f}%")
    print(f"  {'L-inf (max) error':<30} {errors['linf']*100:>9.3f}%")
    print(f"  {'Location of max error':<30} {errors['linf_x']:>9.3f} m")
    print(f"  {'Mean abs. relative error':<30} {errors['mare']*100:>9.3f}%")
    print(f"  {'Near-field error (r<2λ)':<30} {errors['near_err']*100:>9.3f}%")
    print(f"  {'Far-field error  (r>2.5λ)':<30} {errors['far_err']*100:>9.3f}%")
    print(f"  {'-'*42}")
    print(f"  {'Overall verdict':<30} {verdict:>10}")
    print("=" * 60)

    # ── Plot & save ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    plot_comparison(x_pts, E_mag_mie, E_mag_fem, errors, R_cyl, wl, ka)

    np.savez("comparison_results.npz",
             x=x_pts,
             E_mag_mie=E_mag_mie,
             E_mag_fem=E_mag_fem,
             rel_err=errors["rel_err"],
             **{k: v for k, v in errors.items() if np.ndim(v) == 0})
    print("  Saved: comparison_results.npz")

    # ── Export Mie field onto FEM mesh for ParaView ───────────────────────────
    print()
    print("=" * 60)
    print("Exporting Mie field onto FEM mesh for ParaView...")
    _, _, mesh, _, _, _, _ = solve_pec_cylinder(wl / 30, wl / 10, wl / 8)
    export_mie_to_bp(mesh)

    print()
    print("  ParaView workflow:")
    print("  1. Load E_scattered.bp    — FEM scattered field")
    print("  2. Load E_mie.bp          — Mie scattered field (analytical)")
    print("  3. On each: Filters → Calculator →")
    print("       sqrt(f_0^2 + f_1^2)")
    print("  4. Split screen, set identical colour map range on both")
    print("  5. Screenshot for side-by-side visual validation")

    plt.show()