"""
2D Mie Series Analytical Solution for PEC Cylinder Scattering
==============================================================

This code computes the exact analytical solution using 2D Mie theory
for electromagnetic scattering from a Perfect Electric Conductor (PEC) cylinder.

Mathematical Background:
-----------------------
For TM mode (E_z = 0, incident E_y polarized wave):
- Scattered field is expressed as infinite series of cylindrical harmonics
- Mie coefficients: a_n = -J'_n(k₀R) / H'_n(k₀R)
- Where J'_n = derivative of Bessel function
       H'_n = derivative of Hankel function (2nd kind)

Components Computed:
-------------------
1. Mie coefficients (a_n) for n = -20 to +20
2. Scattered electric field E_scattered at comparison points
3. Field magnitude |E|
4. Scattering cross-section σ_scat

HOW TO USE THIS CODE:
=====================

STEP 1: Set Parameters
-----------------------
Edit the parameters section below (lines 60-64) to match your FEM simulation:
- wl: Wavelength [m]
- R_cyl: Cylinder radius [m]
- R_out: Outer boundary radius [m]

STEP 2: Run This Code
---------------------
    python math_analytical.py

STEP 3: Outputs Generated
--------------------------
    analytical_reference.png    # Plots of analytical solution
    analytical_reference.npz    # Numerical values (x, Ex, Ey, E_mag)

STEP 4: Compare with FEM Solution
----------------------------------
Option A - Visual Comparison:
    1. Open analytical_reference.png
    2. Open ParaView with E_scattered.bp
    3. Use "Plot Over Line" filter along x-axis
    4. Visually compare field magnitudes

Option B - Quantitative Comparison:
    1. Extract FEM values using ParaView "Plot Over Line"
    2. Export to CSV
    3. Load both analytical_reference.npz and FEM CSV
    4. Compute L2 error:
       error = ||E_FEM - E_Mie|| / ||E_Mie||

Option C - Automated Comparison (if you have FEM data in .npz):
    fem = np.load('fem_solution.npz')
    mie = np.load('analytical_reference.npz')
    error = np.abs(fem['E_mag'] - mie['E_mag']) / mie['E_mag']
    print(f"Average error: {np.mean(error)*100:.2f}%")

Author: [Your Name]
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2, jv

# ============================================================================
# HELPER FUNCTIONS: Bessel and Hankel Function Derivatives
# ============================================================================

def bessel_derivative(n, z):
    """
    Compute derivative of Bessel function J_n(z) with respect to z.
    
    Formula: J'_n(z) = 0.5 * [J_{n-1}(z) - J_{n+1}(z)]
    
    Parameters:
    -----------
    n : int
        Order of Bessel function
    z : float or complex
        Argument of Bessel function
    
    Returns:
    --------
    float or complex
        Derivative J'_n(z)
    """
    return 0.5 * (jv(n - 1, z) - jv(n + 1, z))


def hankel2_derivative(n, z):
    """
    Compute derivative of Hankel function H_n^(2)(z) with respect to z.
    
    Formula: H'_n(z) = 0.5 * [H_{n-1}(z) - H_{n+1}(z)]
    
    Note: H_n^(2) is the Hankel function of the second kind.
          This represents outgoing cylindrical waves.
    
    Parameters:
    -----------
    n : int
        Order of Hankel function
    z : float or complex
        Argument of Hankel function
    
    Returns:
    --------
    complex
        Derivative H'_n^(2)(z)
    """
    return 0.5 * (hankel2(n - 1, z) - hankel2(n + 1, z))


# ============================================================================
# PARAMETERS - EDIT THESE TO MATCH YOUR FEM SIMULATION
# ============================================================================

# Physical parameters (must match your FEM solver!)
wl = 1.0                    # Wavelength [m]
R_cyl = 0.5                 # Cylinder radius [m]  
R_out = 5.0                 # Outer boundary radius [m]

# Derived quantities
k0 = 2 * np.pi / wl         # Wave number [rad/m]
ka = k0 * R_cyl             # Size parameter (dimensionless)

print("=" * 70)
print("2D MIE SERIES ANALYTICAL SOLUTION")
print("=" * 70)
print(f"\nPhysical Parameters:")
print(f"  Wavelength λ:         {wl:.4f} m")
print(f"  Wavenumber k₀:        {k0:.4f} rad/m")
print(f"  Cylinder radius R:    {R_cyl:.4f} m ({R_cyl/wl:.2f}λ)")
print(f"  Size parameter ka:    {ka:.4f}")
print(f"  Outer radius R_out:   {R_out:.4f} m ({R_out/wl:.2f}λ)")


# ============================================================================
# COMPUTE MIE COEFFICIENTS
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING MIE COEFFICIENTS")
print("=" * 70)

# Number of terms in series (±n_max)
n_max = 20
print(f"Series truncation: n = -{n_max} to +{n_max} ({2*n_max+1} terms)")

# Storage for coefficients
# Key: order n
# Value: complex coefficient a_n
coefficients = {}

# Compute each coefficient
print("\nComputing a_n = -J'_n(k₀R) / H'_n(k₀R)...")

for n in range(-n_max, n_max + 1):
    # Bessel function derivative at cylinder surface
    Jn_prime = bessel_derivative(n, k0 * R_cyl)
    
    # Hankel function derivative at cylinder surface
    Hn_prime = hankel2_derivative(n, k0 * R_cyl)
    
    # Mie coefficient for PEC boundary (TM mode)
    # This enforces tangential E = 0 on cylinder surface
    a_n = -Jn_prime / Hn_prime
    
    coefficients[n] = a_n

print(f"✓ Computed {len(coefficients)} Mie coefficients")

# Show largest coefficients
coeff_magnitudes = [(n, np.abs(a_n)) for n, a_n in coefficients.items()]
coeff_magnitudes.sort(key=lambda x: x[1], reverse=True)
print("\nLargest coefficients (|a_n|):")
for n, mag in coeff_magnitudes[:5]:
    print(f"  n = {n:3d}:  |a_n| = {mag:.6f}")


# ============================================================================
# DEFINE COMPARISON POINTS
# ============================================================================

print("\n" + "=" * 70)
print("DEFINING COMPARISON POINTS")
print("=" * 70)

# Points along x-axis from just outside cylinder to near outer boundary
# Avoid exact cylinder edge and boundary for numerical stability
x_compare = np.linspace(R_cyl * 1.1, R_out * 0.9, 200)
y_compare = np.zeros_like(x_compare)  # y = 0 (x-axis)

print(f"Number of points: {len(x_compare)}")
print(f"Range: x = {x_compare[0]:.4f} m to {x_compare[-1]:.4f} m")
print(f"       (y = 0 along x-axis)")


# ============================================================================
# COMPUTE SCATTERED FIELD (MIE SERIES)
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING ANALYTICAL SCATTERED FIELD")
print("=" * 70)

# Storage for field components
# Ex: x-component of electric field
# Ey: y-component of electric field
Ex_ana = np.zeros(len(x_compare), dtype=np.complex128)
Ey_ana = np.zeros(len(x_compare), dtype=np.complex128)

print("Evaluating Mie series at each point...")

# Loop over all comparison points
for i, (x, y) in enumerate(zip(x_compare, y_compare)):
    
    # Convert Cartesian (x, y) to cylindrical coordinates (ρ, φ)
    rho = np.sqrt(x**2 + y**2)      # Radial distance [m]
    phi = np.arctan2(y, x)           # Azimuthal angle [rad]
    
    # Initialize field components in cylindrical coordinates
    E_rho = 0.0 + 0.0j  # Radial component
    E_phi = 0.0 + 0.0j  # Azimuthal component
    
    # Sum over all Mie terms
    for n, a_n in coefficients.items():
        
        # Skip negligible terms
        if np.abs(a_n) < 1e-15:
            continue
        
        # Hankel function H_n^(2)(k₀ρ) - represents outgoing wave
        H_n = hankel2(n, k0 * rho)
        
        # Derivative H'_n^(2)(k₀ρ)
        H_n_prime = hankel2_derivative(n, k0 * rho)
        
        # Angular dependence e^(inφ)
        phase = np.exp(1j * n * phi)
        
        # Phase factor (-i)^n for TM mode
        j_power = np.exp(-1j * n * np.pi / 2)
        
        # Radial component E_ρ
        # Formula: E_ρ = (1/ik₀ρ) * Σ a_n * n * H_n * e^(inφ)
        E_rho += (1.0 / (1j * k0 * rho)) * j_power * a_n * n * H_n * phase
        
        # Azimuthal component E_φ
        # Formula: E_φ = -(1/ik₀) * Σ a_n * k₀ * H'_n * e^(inφ)
        E_phi += -(1.0 / (1j * k0)) * j_power * a_n * k0 * H_n_prime * phase
    
    # Convert from cylindrical (E_ρ, E_φ) to Cartesian (E_x, E_y)
    # Transformation matrix:
    # E_x = E_ρ cos(φ) - E_φ sin(φ)
    # E_y = E_ρ sin(φ) + E_φ cos(φ)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    Ex_ana[i] = E_rho * cos_phi - E_phi * sin_phi
    Ey_ana[i] = E_rho * sin_phi + E_phi * cos_phi

# Compute field magnitude |E| = sqrt(|E_x|² + |E_y|²)
E_mag_ana = np.sqrt(np.abs(Ex_ana)**2 + np.abs(Ey_ana)**2)

print("✓ Analytical solution computed at all points")


# ============================================================================
# FIELD PROPERTIES AND DIAGNOSTICS
# ============================================================================

print("\n" + "=" * 70)
print("FIELD PROPERTIES")
print("=" * 70)

print(f"Maximum |E|:              {np.max(E_mag_ana):.6e} V/m")
print(f"Minimum |E|:              {np.min(E_mag_ana):.6e} V/m")
print(f"At cylinder edge (r=R):   {E_mag_ana[0]:.6e} V/m")
print(f"At outer region (r≈R_out): {E_mag_ana[-1]:.6e} V/m")
print(f"Decay factor:             {E_mag_ana[0]/E_mag_ana[-1]:.2f}×")

# Sample values at key points
print("\n" + "=" * 70)
print("FIELD VALUES AT KEY POINTS")
print("=" * 70)
print(f"{'x [m]':<10} {'|E| [V/m]':<15} {'Re(Ey)':<15} {'Im(Ey)':<15}")
print("-" * 70)

sample_indices = [0, 25, 50, 75, 100, 150, 199]
for idx in sample_indices:
    if idx < len(x_compare):
        x_val = x_compare[idx]
        mag_val = E_mag_ana[idx]
        ey_real = Ey_ana[idx].real
        ey_imag = Ey_ana[idx].imag
        print(f"{x_val:<10.4f} {mag_val:<15.6e} {ey_real:<15.6e} {ey_imag:<15.6e}")


# ============================================================================
# SCATTERING CROSS-SECTION
# ============================================================================

print("\n" + "=" * 70)
print("SCATTERING CROSS-SECTION")
print("=" * 70)

# 2D scattering cross-section: σ = (2/k₀) Σ |a_n|²
# For n=0: contribute once
# For n≠0: contribute twice (±n)
sigma_sca = 0.0
for n, a_n in coefficients.items():
    if n == 0:
        sigma_sca += np.abs(a_n)**2
    else:
        sigma_sca += 2 * np.abs(a_n)**2

sigma_sca *= (2.0 / k0)

print(f"Total scattering cross-section: σ_scat = {sigma_sca:.6f} m")
print(f"Normalized by diameter (σ/2R):         {sigma_sca/(2*R_cyl):.6f}")
print(f"Normalized by wavelength (σ/λ):        {sigma_sca/wl:.6f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ----------------------------------------------------------------------------
# Plot 1: Field Magnitude |E|
# ----------------------------------------------------------------------------
ax = axes[0, 0]
ax.plot(x_compare, E_mag_ana, 'b-', linewidth=2.5, label='|E| analytical')
ax.axvline(R_cyl, color='k', linestyle='--', linewidth=1.5, alpha=0.7, 
           label='Cylinder edge (r=R)')
ax.set_xlabel('x [m]', fontsize=13, fontweight='bold')
ax.set_ylabel('|E| [V/m]', fontsize=13, fontweight='bold')
ax.set_title('Scattered Field Magnitude (Mie Theory)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

# ----------------------------------------------------------------------------
# Plot 2: Real and Imaginary Parts of E_y
# ----------------------------------------------------------------------------
ax = axes[0, 1]
ax.plot(x_compare, Ey_ana.real, 'b-', linewidth=2.5, label='Re(E_y)')
ax.plot(x_compare, Ey_ana.imag, 'r--', linewidth=2.5, label='Im(E_y)')
ax.axvline(R_cyl, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('x [m]', fontsize=13, fontweight='bold')
ax.set_ylabel('E_y [V/m]', fontsize=13, fontweight='bold')
ax.set_title('E_y Component (Real and Imaginary)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

# ----------------------------------------------------------------------------
# Plot 3: Phase Distribution
# ----------------------------------------------------------------------------
ax = axes[1, 0]
phase_y = np.angle(Ey_ana)  # Phase in radians
ax.plot(x_compare, phase_y, 'g-', linewidth=2.5)
ax.axvline(R_cyl, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xlabel('x [m]', fontsize=13, fontweight='bold')
ax.set_ylabel('Phase(E_y) [rad]', fontsize=13, fontweight='bold')
ax.set_title('Phase Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

# ----------------------------------------------------------------------------
# Plot 4: Log-Scale Magnitude (shows decay)
# ----------------------------------------------------------------------------
ax = axes[1, 1]
ax.semilogy(x_compare, E_mag_ana, 'b-', linewidth=2.5)
ax.axvline(R_cyl, color='k', linestyle='--', linewidth=1.5, alpha=0.7, 
           label='Cylinder edge')
ax.set_xlabel('x [m]', fontsize=13, fontweight='bold')
ax.set_ylabel('|E| [V/m] (log scale)', fontsize=13, fontweight='bold')
ax.set_title('Field Decay (Log Scale)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('analytical_reference.png', dpi=200, bbox_inches='tight')
print("✓ Saved: analytical_reference.png")


# ============================================================================
# SAVE NUMERICAL DATA
# ============================================================================

print("\n" + "=" * 70)
print("SAVING DATA")
print("=" * 70)

# Save analytical solution for comparison with FEM
np.savez('analytical_reference.npz',
         x=x_compare,           # x-coordinates [m]
         y=y_compare,           # y-coordinates [m] (all zeros)
         Ex=Ex_ana,             # x-component of E field [V/m]
         Ey=Ey_ana,             # y-component of E field [V/m]
         E_mag=E_mag_ana,       # Field magnitude |E| [V/m]
         k0=k0,                 # Wave number [rad/m]
         R_cyl=R_cyl,           # Cylinder radius [m]
         R_out=R_out,           # Outer radius [m]
         wl=wl)                 # Wavelength [m]

print("✓ Saved: analytical_reference.npz")
print("\nContents:")
print("  x        : x-coordinates (200 points)")
print("  Ey       : E_y component (complex)")
print("  E_mag    : Field magnitude |E|")
print("  + parameters (k0, R_cyl, R_out, wl)")


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

print("\n" + "=" * 70)
print("HOW TO USE THESE RESULTS FOR FEM VALIDATION")
print("=" * 70)

print("\n1. VISUAL COMPARISON:")
print("   - Open analytical_reference.png")
print("   - Compare field patterns with your FEM ParaView output")
print("   - Check: magnitude decay, oscillations, phase behavior")

print("\n2. QUANTITATIVE COMPARISON (Manual):")
print("   a) Open ParaView and load E_scattered.bp")
print("   b) Use 'Plot Over Line' filter:")
print(f"      - Point 1: ({R_cyl*1.1:.4f}, 0, 0)")
print(f"      - Point 2: ({R_out*0.9:.4f}, 0, 0)")
print("   c) Export data to CSV")
print("   d) Load analytical_reference.npz in Python:")
print("      >>> data = np.load('analytical_reference.npz')")
print("      >>> mie_mag = data['E_mag']")
print("   e) Compare with FEM CSV values")

print("\n3. QUANTITATIVE COMPARISON (Python Script):")
print("   # If you have FEM data in .npz format:")
print("   fem = np.load('fem_solution.npz')")
print("   mie = np.load('analytical_reference.npz')")
print("   error = np.abs(fem['E_mag'] - mie['E_mag']) / mie['E_mag']")
print("   print(f'Average error: {np.mean(error)*100:.2f}%')")

print("\n4. EXPECTED VALUES:")
print(f"   Near cylinder (x ≈ {R_cyl*1.1:.2f} m): |E| ≈ {E_mag_ana[0]:.3e} V/m")
print(f"   Far field (x ≈ {R_out*0.9:.2f} m):     |E| ≈ {E_mag_ana[-1]:.3e} V/m")

print("\n" + "=" * 70)
print("COMPUTATION COMPLETE!")
print("=" * 70)

# Display plots
plt.show()