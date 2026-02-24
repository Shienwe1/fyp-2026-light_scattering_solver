"""
Mesh Convergence Study for PEC Cylinder Scattering
===================================================
Tests solver accuracy by running on progressively refined meshes
"""

import numpy as np
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io.gmsh import read_from_msh
from dolfinx.fem import functionspace, Function, dirichletbc, locate_dofs_topological, form
from dolfinx.fem.petsc import assemble_matrix, set_bc as petsc_set_bc
import ufl
from ufl import inner, curl, dx, ds, TrialFunction, TestFunction, Measure, FacetNormal
import matplotlib.pyplot as plt


def cross_z(a, b):
    """2D cross product returning scalar"""
    return a[0]*b[1] - a[1]*b[0]


def cross_xy(a, b):
    """Cross product: (ax, ay, 0) × (0, 0, b) = (ay*b, -ax*b, 0)"""
    return ufl.as_vector((a[1]*b, -a[0]*b, 0))


def create_mesh(R_cyl, R_out, lc_cyl, lc_out, filename="mesh.msh"):
    """Create annular mesh with specified element sizes"""
    gmsh.initialize()
    gmsh.model.add("pec")
    
    cyl = gmsh.model.occ.addCircle(0, 0, 0, R_cyl)
    out = gmsh.model.occ.addCircle(0, 0, 0, R_out)
    
    cyl_loop = gmsh.model.occ.addCurveLoop([cyl])
    out_loop = gmsh.model.occ.addCurveLoop([out])
    surf = gmsh.model.occ.addPlaneSurface([out_loop, cyl_loop])
    
    gmsh.model.occ.synchronize()
    
    gmsh.model.addPhysicalGroup(1, [out], tag=1, name="ABC")
    gmsh.model.addPhysicalGroup(1, [cyl], tag=2, name="PEC")
    gmsh.model.addPhysicalGroup(2, [surf], tag=1, name="Domain")
    
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_out)
    
    # Mesh refinement near cylinder
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [cyl])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
    
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_cyl)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_out)
    gmsh.model.mesh.field.setNumber(2, "DistMin", R_cyl)
    gmsh.model.mesh.field.setNumber(2, "DistMax", R_out)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(2)
    
    gmsh.write(filename)
    gmsh.finalize()
    
    return filename


def solve_pec_cylinder(mesh_factor, degree=3):
    """
    Solve PEC cylinder problem for given mesh refinement
    
    Parameters:
    -----------
    mesh_factor : float
        Mesh size multiplier (smaller = finer mesh)
    degree : int
        Polynomial degree for N1curl elements
        
    Returns:
    --------
    Es_h : Function
        Scattered field solution
    num_cells : int
        Number of mesh cells
    num_dofs : int
        Number of degrees of freedom
    h_avg : float
        Average element size
    """
    
    # Parameters
    wl = 1.0
    k0 = 2 * np.pi / wl
    mu = 1.0
    eps = 1.0
    n_bkg = 1.0
    
    R_cyl = 0.5 * wl
    R_out = 5.0 * wl
    lc_cyl = mesh_factor * wl / 15
    lc_out = mesh_factor * wl / 5
    
    # Generate mesh
    mesh_file = create_mesh(R_cyl, R_out, lc_cyl, lc_out, f"mesh_h{mesh_factor}.msh")
    mesh, cell_tags, facet_tags, _, _, _ = read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2)
    
    num_cells = mesh.topology.index_map(2).size_local
    
    # Compute average mesh size
    import dolfinx.cpp as cpp
    h_cells = cpp.mesh.h(mesh._cpp_object, 2, np.arange(num_cells, dtype=np.int32))
    h_avg = np.mean(h_cells)
    
    # Function space
    V = functionspace(mesh, ("N1curl", degree))
    num_dofs = V.dofmap.index_map.size_global
    
    # Measures
    dx = Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    n = FacetNormal(mesh)
    
    # Trial and test functions
    Es = TrialFunction(V)
    v = TestFunction(V)
    
    Es_3d = ufl.as_vector((Es[0], Es[1], 0))
    v_3d = ufl.as_vector((v[0], v[1], 0))
    
    # Incident field
    Eb = Function(V)
    Eb.interpolate(lambda x: (np.zeros(x.shape[1], dtype=np.complex128),
                              np.exp(-1j * k0 * x[0])))
    
    # PEC boundary condition
    tdim = mesh.topology.dim
    pec_facets = facet_tags.find(2)
    pec_dofs = locate_dofs_topological(V, tdim-1, pec_facets)
    
    E_pec = Function(V)
    E_pec.interpolate(lambda x: (np.zeros(x.shape[1], dtype=np.complex128),
                                  -np.exp(-1j * k0 * x[0])))
    bc = dirichletbc(E_pec, dofs=pec_dofs)
    
    # Weak form - homogeneous system with Dirichlet BC as source
    a_form = form((
        -inner(curl(Es), curl(v)) * dx
        + k0**2 * inner(Es, v) * dx
        + 1j * k0 * inner(cross_xy(n, cross_z(n, Es_3d)), cross_xy(n, cross_z(n, v_3d))) * ds(1)
    ))
    
    # Assemble and solve
    A = assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    Es_h = Function(V)
    b = Es_h.x.petsc_vec.duplicate()
    b.set(0.0)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    petsc_set_bc(b, [bc])
    
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-9)
    ksp.solve(b, Es_h.x.petsc_vec)
    Es_h.x.scatter_forward()
    
    return Es_h, num_cells, num_dofs, h_avg


def evaluate_at_points(Es_h, points):
    """
    Evaluate solution at specific points
    
    Parameters:
    -----------
    Es_h : Function
        Solution to evaluate
    points : ndarray
        Points to evaluate at (N x 2 array)
        
    Returns:
    --------
    values : ndarray
        Complex field values at points
    """
    values = np.zeros(len(points), dtype=np.complex128)
    
    for i, (x, y) in enumerate(points):
        try:
            point = np.array([[x], [y], [0.0]])
            vals = Es_h.eval(point, Es_h.function_space.mesh.comm)
            # Compute magnitude
            values[i] = np.sqrt(vals[0, 0]**2 + vals[1, 0]**2)
        except:
            values[i] = np.nan
    
    return values


# ============================================
# MAIN CONVERGENCE STUDY
# ============================================

if __name__ == "__main__":
    
    if MPI.COMM_WORLD.rank == 0:
        print("="*70)
        print("MESH CONVERGENCE STUDY")
        print("="*70)
        print()
    
    # Mesh refinement levels - MUCH FINER to achieve kh < 0.5
    # For k0 = 2π and kh < 0.5, need h < 0.08
    # Use h_factor < 0.5 at finest level
    mesh_factors = [1.2, 0.9, 0.6, 0.45, 0.35]  # 5 levels, finest gives kh ≈ 0.3
    
    solutions = []
    num_cells_list = []
    num_dofs_list = []
    h_list = []
    
    # Reference points for evaluation (along x-axis outside cylinder)
    wl = 1.0
    R_cyl = 0.5 * wl
    R_out = 5.0 * wl
    x_ref = np.array([1.0, 2.0, 3.0, 4.0])  # 4 reference points
    points_ref = np.column_stack([x_ref, np.zeros_like(x_ref)])
    
    # Solve on each mesh
    for i, mf in enumerate(mesh_factors):
        if MPI.COMM_WORLD.rank == 0:
            print(f"\n{'='*70}")
            print(f"Mesh {i+1}/{len(mesh_factors)}: h_factor = {mf}")
            print(f"{'='*70}")
        
        Es_h, num_cells, num_dofs, h_avg = solve_pec_cylinder(mf, degree=3)
        
        solutions.append(Es_h)
        num_cells_list.append(num_cells)
        num_dofs_list.append(num_dofs)
        h_list.append(h_avg)
        
        if MPI.COMM_WORLD.rank == 0:
            Es_norm = Es_h.x.petsc_vec.norm()
            print(f"  Cells: {num_cells}")
            print(f"  DOFs:  {num_dofs}")
            print(f"  h_avg: {h_avg:.6f}")
            print(f"  ||Es||: {Es_norm:.6e}")
    
    # Convergence analysis using NORMS (not pointwise)
    if MPI.COMM_WORLD.rank == 0:
        print("\n" + "="*70)
        print("CONVERGENCE ANALYSIS")
        print("="*70)
        
        # Use finest mesh as reference
        norm_ref = solutions[-1].x.petsc_vec.norm()
        
        # Compute errors relative to finest mesh (norm-based)
        errors_norm = []
        
        for i in range(len(solutions)-1):
            norm_i = solutions[i].x.petsc_vec.norm()
            # Error estimate: |norm_i - norm_ref| / norm_ref
            error_norm = abs(norm_i - norm_ref) / norm_ref
            errors_norm.append(error_norm)
        
        # Compute convergence rates
        print("\nMesh  |  h_avg   |  Cells  |  DOFs   |  ||Es||    | Norm Error |  Rate")
        print("-"*80)
        
        for i in range(len(mesh_factors)):
            Es_norm = solutions[i].x.petsc_vec.norm()
            
            if i < len(errors_norm):
                err_str = f"{errors_norm[i]:.3e}"
                
                # Compute convergence rate
                if i > 0:
                    rate = np.log(errors_norm[i-1]/errors_norm[i]) / np.log(h_list[i-1]/h_list[i])
                    rate_str = f"{rate:.2f}"
                else:
                    rate_str = " -- "
            else:
                err_str = "  (ref)   "
                rate_str = " -- "
            
            print(f"{i+1:2d}    | {h_list[i]:.5f} | {num_cells_list[i]:6d} | "
                  f"{num_dofs_list[i]:7d} | {Es_norm:.3e} | {err_str} | {rate_str}")
        
        # Expected rate for 3rd order elements: ~3
        print("\nExpected convergence rate: ~3.0 (for 3rd order elements)")
        
        # POLLUTION EFFECT ANALYSIS (High Priority)
        print("\n" + "="*70)
        print("POLLUTION EFFECT ANALYSIS")
        print("="*70)
        print()
        print("Mesh  |  h_avg   |  kh      | DOFs    | Regime")
        print("-" * 70)
        
        wl = 1.0
        k0 = 2 * np.pi / wl
        
        for i, (h, n_dofs) in enumerate(zip(h_list, num_dofs_list)):
            kh = k0 * h
            
            if kh > 0.5:
                regime = "PRE-ASYMPTOTIC ⚠"
                warning = "Pollution risk!"
            elif kh > 0.3:
                regime = "Near boundary ⚠"
                warning = "Caution advised"
            else:
                regime = "ASYMPTOTIC ✓"
                warning = "Stable"
            
            print(f"{i+1:2d}    | {h:.5f} | {kh:.4f}  | {n_dofs:7d} | {regime}")
        
        print()
        print("Pollution criterion: kh < 0.5 for asymptotic convergence")
        print("Frelet et al. (2024): Pre-asymptotic regime shows error stagnation")
        print()
        
        # Check for error stagnation
        if len(errors_norm) >= 2:
            error_reductions = []
            for i in range(len(errors_norm)-1):
                reduction = errors_norm[i] / errors_norm[i+1]
                error_reductions.append(reduction)
            
            if error_reductions:
                avg_reduction = np.mean(error_reductions)
                print("ERROR STAGNATION CHECK:")
                print(f"  Average error reduction: {avg_reduction:.2f}×")
                
                if avg_reduction < 1.5:
                    print("  ⚠ WARNING: Error stagnation detected!")
                    print("     Possible causes:")
                    print("     - Pre-asymptotic regime (pollution)")
                    print("     - ABC reflections dominating")
                    print("     - Solver tolerance too loose")
                elif avg_reduction < 2.0:
                    print("  ⚠ CAUTION: Slow convergence")
                    print("     Consider checking kh values")
                else:
                    print("  ✓ Healthy convergence (>2× per refinement)")
                print()
        
        # Plot convergence
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Error vs h (WITH DATA!)
        ax = axes[0]
        h_plot = np.array(h_list[:-1])  # Exclude finest (reference)
        err_plot = np.array(errors_norm)
        
        if len(err_plot) > 1:
            ax.loglog(h_plot, err_plot, 'bo-', linewidth=2.5, markersize=10, 
                     label='Norm-based Error', zorder=3)
            
            # Plot reference slopes for O(h²) and O(h³)
            h_ref = np.array([h_plot[0], h_plot[-1]])
            for p, style, alpha in [(2, '--', 0.6), (3, '-.', 0.5)]:
                C = err_plot[0] / h_plot[0]**p
                ax.loglog(h_ref, C * h_ref**p, style, alpha=alpha, 
                         linewidth=2, label=f'O(h^{p})', zorder=1)
            
            # Add text annotation explaining error metric
            ax.text(0.95, 0.05, 'Error = |norm(h) - norm(h_finest)| / norm(h_finest)', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                   horizontalalignment='right', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.3))
        
        ax.set_xlabel('Average element size h [m]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Error', fontsize=12, fontweight='bold')
        ax.set_title('Convergence: Error vs Mesh Size', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, which='both', linestyle=':')
        
        # Plot 2: Solution norm vs DOFs
        ax = axes[1]
        norms = [sol.x.petsc_vec.norm() for sol in solutions]
        ax.semilogx(num_dofs_list, norms, 'rs-', linewidth=2.5, markersize=10,
                   label='||Es|| (L2 norm)', markerfacecolor='red', 
                   markeredgecolor='darkred', markeredgewidth=1.5)
        
        # Check convergence (matches console output formula)
        norm_diff = np.abs(norms[-1] - norms[-2]) / norms[-2] * 100
        ax.text(0.05, 0.95, f'Convergence: {norm_diff:.2f}%\n(finest vs 2nd finest)',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', 
                         edgecolor='orange', linewidth=2))
        
        ax.set_xlabel('Number of DOFs', fontsize=12, fontweight='bold')
        ax.set_ylabel('||Es|| (L2 norm)', fontsize=12, fontweight='bold')
        ax.set_title('Solution Norm vs DOFs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, which='both', linestyle=':')
        
        plt.tight_layout()
        plt.savefig('mesh_convergence_study.png', dpi=200, bbox_inches='tight')
        print("\nSaved: mesh_convergence_study.png")
        
        # Final assessment
        print("\n" + "="*70)
        print("CONVERGENCE ASSESSMENT")
        print("="*70)
        
        # Check finest mesh pollution
        kh_finest = k0 * h_list[-1]
        if kh_finest > 0.5:
            print("\n⚠ WARNING: Finest mesh in pre-asymptotic regime!")
            print(f"   kh = {kh_finest:.3f} > 0.5")
            print("   Pollution effect may invalidate convergence rates")
            print("   Recommendation: Refine further until kh < 0.5")
        elif kh_finest > 0.3:
            print(f"\n⚠ CAUTION: Finest mesh near pre-asymptotic boundary")
            print(f"   kh = {kh_finest:.3f}")
            print("   Results should be reliable but consider additional refinement")
        else:
            print(f"\n✓ Finest mesh in asymptotic regime")
            print(f"   kh = {kh_finest:.3f} < 0.3")
            print("   Convergence rates are reliable")
        
        if len(errors_norm) > 1:
            # Compute average rate from all pairs
            rates = []
            for i in range(len(errors_norm)-1):
                rate = np.log(errors_norm[i]/errors_norm[i+1]) / np.log(h_list[i]/h_list[i+1])
                rates.append(rate)
            
            avg_rate = np.mean(rates)
            
            print(f"\nAverage convergence rate: {avg_rate:.2f}")
            
            if avg_rate > 2.5:
                print("✓ EXCELLENT: Convergence rate matches 3rd order elements")
            elif avg_rate > 2.0:
                print("✓ GOOD: Convergence rate indicates correct implementation")
            elif avg_rate > 1.0:
                print("⚠ WARNING: Convergence rate lower than expected")
                print("  Possible causes: Pre-asymptotic regime, ABC errors")
            else:
                print("✗ ERROR: No convergence observed")
                print("  Check: formulation, BC implementation, pollution effect")
        
        if norm_diff < 1.0:
            print(f"\n✓ Solution converged (< 1% change on finest mesh)")
        elif norm_diff < 5.0:
            print(f"\n⚠ Solution nearly converged (< 5% change)")
        else:
            print(f"\n✗ Solution not fully converged (> 5% change)")
            print("  Recommendation: Add finer mesh levels")
        
        print("="*70)