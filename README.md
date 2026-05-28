# Finite Element Solver for Light Scattering Simulation

**Author:** Lee Shien Wei  
**Institution:** Universiti Malaya, Department of Mechanical Engineering  
**Project:** Final Year Project (FYP) 2026  

---

## Overview

This repository contains a fully validated finite element method (FEM) solver for 2D electromagnetic light scattering simulations, implemented using [FEniCSx / DOLFINx](https://fenicsproject.org/). The project progresses from a canonical Perfect Electric Conductor (PEC) cylinder benchmark through to a plasmonic LSPR biosensor model, demonstrating the accuracy and applicability of FEM for nanophotonic sensing applications.

The solver is validated against the exact 2D Mie series analytical solution, achieving L2 errors below 0.5% on fine meshes with degree-3 Nédélec curl-conforming elements.

---

## Repository Structure

```
fyp-2026-light_scattering_solver/
│
├── abc/                          ← PEC cylinder solver with Silver-Müller ABC
│   ├── 010_main_solver.py        — FEM solver, circular domain
│   ├── 011_compare.py            — Mie reference + FEM vs Mie error metrics
│   ├── 012_compare_fem_mie.py    — Standalone FEM vs Mie comparison
│   ├── 013_convergence.py        — Mesh convergence study (ABC)
│   └── 014_solver_square.py      — FEM solver, square domain (matches PML geometry)
│
├── pml/                          ← PEC cylinder solver with rectangular PML
│   ├── 010_main_solver.py        — FEM solver with PML boundary condition
│   ├── 011_compare.py            — Mie reference + FEM vs Mie error metrics + .bp export
│   └── 012_convergence.py        — Mesh convergence study (PML, physical domain only)
│
├── plasmonic/                    ← LSPR biosensor solver (gold nanoparticle)
│   ├── 020_main_solver.py        — Two-phase LSPR biosensor solver
│   ├── 021_mesh_convergence.py   — Mesh convergence study for biosensor
│   ├── 022_validation_suite.py   — Validation 1: bulk sensitivity linearity
│   ├── 022_validation.py         — Local validation runner
│   ├── 023_decay_test.py         — Validation 2: evanescent surface decay
│   └── 024_export_field.py       — Single-wavelength E-field export for ParaView
│
├── 07_mie_to_bp.py               ← Mie analytical field → ParaView .bp export
├── 08_visualizations.py          ← Post-processing visualization utilities
├── environment.yml               ← Conda environment specification
└── README.md
```

---

## Physics

### Part 1 — PEC Cylinder Scattering

Solves the 2D time-harmonic Maxwell equations for electromagnetic scattering from a Perfect Electric Conductor (PEC) cylinder:

- **Formulation:** Scattered field formulation with homogeneous equation
- **Source:** Dirichlet BC `Es = −Ei` on PEC surface (enforces total field = 0)
- **Incident field:** y-polarised plane wave `Ei = ŷ exp(−ik₀x)`
- **Boundary conditions:**
  - `abc/`: First-order Silver-Müller absorbing boundary condition
  - `pml/`: Rectangular Perfectly Matched Layer (PML) with quadratic stretch profile
- **Validation:** Mie series analytical solution (L2 error < 0.5%, L-inf < 1%)

### Part 2 — LSPR Biosensor (Gold Nanoparticle)

Solves the inhomogeneous Maxwell scattering problem for a gold nanoparticle biosensor:

- **Formulation:** Penetrable (inhomogeneous) volume source formulation — no Dirichlet BC
- **Gold permittivity:** Johnson & Christy (1972) experimental data, cubic spline interpolation
- **Geometry:** 25 nm Au core + dielectric shell + water background + rectangular PML
- **Source term:** `k₀²(ε − ε_bkg) · Eb` drives scattering
- **Two-phase architecture:**
  - Phase 1 — `build_biosensor_mesh()`: mesh + DOF map built once per sweep
  - Phase 2 — `sweep_wavelengths()`: material update + solve per wavelength (no mesh rebuild)
- **Validation:**
  - Bulk sensitivity linearity: R² = 0.994, sensitivity = 318.9 nm/RIU
  - Evanescent surface decay: exponential saturation curve confirmed

---

## Key Results

| Test | Result |
|------|--------|
| PEC cylinder L2 error (degree-3, fine mesh) | < 0.5% |
| PEC cylinder L-inf error | < 1.0% |
| Convergence rate (degree-3 N1curl) | ~3.0 |
| LSPR bulk sensitivity | 318.9 nm/RIU |
| Bulk linearity R² | 0.994 |
| LSPR peak wavelength (25 nm Au, water) | ~358 nm |

---

## Installation

### Requirements

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Linux / WSL2 (Ubuntu 22.04 recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Shienwe1/fyp-2026-light_scattering_solver.git
cd fyp-2026-light_scattering_solver

# Create and activate the conda environment
conda env create -f environment.yml
conda activate fenicsx-complex
```

---

## Usage

### PEC Cylinder (ABC)

```bash
cd abc/

# Run FEM solver — generates E_scattered.bp, fem_scattered.npz
python 010_main_solver.py

# Compare FEM vs Mie — generates fem_mie_comparison.png
python 011_compare.py

# Mesh convergence study
python 013_convergence.py

# Export Mie field to ParaView
python 07_mie_to_bp.py
```

### PEC Cylinder (PML)

```bash
cd pml/

# Run FEM solver with PML
python 010_main_solver.py

# Compare FEM vs Mie + export E_mie.bp for ParaView
python 011_compare.py

# Mesh convergence study (L2 error restricted to physical domain)
python 012_convergence.py
```

### LSPR Biosensor

```bash
cd plasmonic/

# Full spectral sweep + biosensor_hotspot.bp ParaView export
mpirun -n 4 python 020_main_solver.py

# Validation 1 — bulk sensitivity linearity
mpirun -n 4 python 022_validation_suite.py

# Validation 2 — evanescent surface decay
mpirun -n 4 python 023_decay_test.py

# Single-wavelength E-field export for ParaView
mpirun -n 4 python 024_export_field.py
```

---

## ParaView Visualisation

After running any solver, open the output `.bp` folders in ParaView:

1. **File → Open** → select `.bp` folder → open as ADIOS2 BP4
2. **Filters → Calculator** → expression: `sqrt(f_0^2 + f_1^2)` (or `sqrt(E_total_0^2 + E_total_1^2)`)
3. Apply colour map → Screenshot for figures

For side-by-side FEM vs Mie comparison:
- Load `E_scattered.bp` and `E_mie.bp` simultaneously
- Set identical colour map range on both panels

---

## References

- Johnson & Christy, *Phys. Rev. B* **6**, 4370 (1972) — Gold optical constants
- Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*, Wiley (1983)
- Ward & Pendry, *J. Mod. Opt.* **43**, 773 (1996) — PML coordinate transformation
- Castriotta, Baratta, Dokken — FEniCSx PML demo
- Alnæs et al., *Archive of Numerical Software* **3** (2015) — FEniCS project
