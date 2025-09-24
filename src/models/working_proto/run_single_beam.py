import numpy as np
import gmsh
import dolfinx
from mpi4py import MPI
import pyvista
import os
import basix.ufl
import ufl
import dolfinx.fem.petsc

# --- Create an output directory for images ---
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)
print(f"Saving all plot outputs to '{output_folder}/'")

pyvista.set_plot_theme("document")

print(f"Using dolfinx version: {dolfinx.__version__}")

# =============================================================================
# STEP 0: DEFINE THE PARAMETERS
# =============================================================================
print("--- Step 0: Defining Beam Parameters ---")
beam_length, beam_height, mesh_size = 100.0, 20.0, 3.0
youngs_modulus_E, poissons_ratio_nu = 210e3, 0.3
force_magnitude, force_angle_deg = 500.0, -60.0
DOMAIN_TAG, FIXED_WALL_TAG, LOAD_SURFACE_TAG = 1, 2, 3

# =============================================================================
# MAIN LOGIC
# =============================================================================
gmsh.initialize() # Initialize Gmsh outside the try block
try:
    # --- STEP 1: CREATE GEOMETRY & MESH ---
    print("\n--- Step 1: Creating Geometry with the direct gmsh API ---")
    gmsh.model.add("beam_model")
    p1 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=mesh_size)
    p2 = gmsh.model.geo.addPoint(beam_length, 0, 0, meshSize=mesh_size)
    p3 = gmsh.model.geo.addPoint(beam_length, beam_height, 0, meshSize=mesh_size)
    p4 = gmsh.model.geo.addPoint(0, beam_height, 0, meshSize=mesh_size)
    l1, l2, l3, l4 = gmsh.model.geo.addLine(p1, p2), gmsh.model.geo.addLine(p2, p3), gmsh.model.geo.addLine(p3, p4), gmsh.model.geo.addLine(p4, p1)
    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface], DOMAIN_TAG)
    gmsh.model.addPhysicalGroup(1, [l4], FIXED_WALL_TAG)
    gmsh.model.addPhysicalGroup(1, [l2], LOAD_SURFACE_TAG)
    print("Generating mesh...")
    gmsh.model.mesh.generate(2)
    print("Mesh generation complete.")

    # --- VISUALIZATION 1 ---
    nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    cell_info = gmsh.model.mesh.getElementsByType(2)
    cells = cell_info[1].reshape(-1, 3) - 1
    pv_mesh = pyvista.UnstructuredGrid({pyvista.CellType.TRIANGLE: cells}, nodes)
    plotter = pyvista.Plotter(off_screen=True, window_size=[800, 300])
    plotter.add_mesh(pv_mesh, show_edges=True, style='wireframe')
    plotter.view_xy()
    plotter.set_background('white')
    plotter.add_text("Step 1: The Final Mesh", font_size=15, color='black')
    mesh_filename = os.path.join(output_folder, "01_mesh.png")
    plotter.screenshot(mesh_filename)
    print(f"Saved mesh plot to {mesh_filename}")

    # --- STEP 2: SOLVE THE FEA PROBLEM ---
    print("\n--- Step 2: Solving FEA with FEniCSx ---")
    domain, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )

    # For the vector space
    vector_element = basix.ufl.element(
        "Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,)
    )
    V = dolfinx.fem.functionspace(domain, vector_element)


    fixed_facets = facet_tags.find(FIXED_WALL_TAG)
    u_fixed = np.array([0, 0], dtype=dolfinx.default_scalar_type)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(domain, u_fixed),dolfinx.fem.locate_dofs_topological(V, 1, fixed_facets),V)
    traction = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type([0, 0]))
    force_angle_rad = np.deg2rad(force_angle_deg)
    traction.value[0] = force_magnitude * np.sin(force_angle_rad) / beam_height
    traction.value[1] = -force_magnitude * np.cos(force_angle_rad) / beam_height
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    mu = youngs_modulus_E / (2 * (1 + poissons_ratio_nu))
    lambda_ = youngs_modulus_E * poissons_ratio_nu / ((1 + poissons_ratio_nu) * (1 - 2 * poissons_ratio_nu))
    lambda_ = 2 * mu * lambda_ / (lambda_ + 2 * mu)
    def epsilon(u): return ufl.sym(ufl.grad(u))
    def sigma(u): return lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(traction, v) * ds(LOAD_SURFACE_TAG)
    print("Solving the linear system...")
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    print("Solve complete.")

    # --- STEP 3: VISUALIZE THE RESULTS ---
    print("\n--- Step 3: Saving Result Visualizations ---")
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    # Get the 2D displacement vectors
    u_2d = uh.x.array.reshape((geometry.shape[0], 2))
    # Create a new 3D array and add the 2D displacements, leaving Z as zero
    u_3d = np.zeros((u_2d.shape[0], 3))
    u_3d[:, :2] = u_2d
    # Add the 3D displacement vector to the grid
    grid["u"] = u_3d
    warped = grid.warp_by_vector("u", factor=50)
    plotter_def = pyvista.Plotter(off_screen=True, window_size=[800, 400])
    plotter_def.add_mesh(grid, style='wireframe', color='gray', opacity=0.5, label='Original')
    plotter_def.add_mesh(warped, show_edges=True, label='Deformed')
    plotter_def.add_text("Step 2: Deformation (scaled by 50x)", font_size=15, color='black')
    plotter_def.view_xy()
    plotter_def.set_background('white')
    plotter_def.add_legend()
    def_filename = os.path.join(output_folder, "02_deformation.png")
    plotter_def.screenshot(def_filename)
    print(f"Saved deformation plot to {def_filename}")

    # For the tensor space
    tensor_element = basix.ufl.element(
        "DG", domain.basix_cell(), 0, shape=(domain.geometry.dim, domain.geometry.dim)
    )
    S = dolfinx.fem.functionspace(domain, tensor_element)

    s_expr = dolfinx.fem.Expression(sigma(uh), S.element.interpolation_points())
    s_projected = dolfinx.fem.Function(S)
    s_projected.interpolate(s_expr)

    # Define Von Mises stress as a UFL expression using the solution 'uh'
    s = sigma(uh)
    von_mises = ufl.sqrt(s[0, 0]**2 - s[0, 0]*s[1, 1] + s[1, 1]**2 + 3*s[0, 1]**2)

    # Create a function space and Function for the scalar Von Mises stress
    V_von_mises = dolfinx.fem.functionspace(domain, ("DG", 0))
    von_mises_func = dolfinx.fem.Function(V_von_mises)

    # Interpolate the UFL expression into the new Function
    vm_expr = dolfinx.fem.Expression(von_mises, V_von_mises.element.interpolation_points())
    von_mises_func.interpolate(vm_expr)

    # Create the grid for plotting and add the data from the computed function
    topology_vm, cell_types_vm, geometry_vm = dolfinx.plot.vtk_mesh(domain)
    grid_vm = pyvista.UnstructuredGrid(topology_vm, cell_types_vm, geometry_vm)
    grid_vm.cell_data["Von Mises Stress (MPa)"] = von_mises_func.x.array

    plotter_stress = pyvista.Plotter(off_screen=True, window_size=[800, 300])
    plotter_stress.add_mesh(grid_vm, show_edges=False, scalar_bar_args={'title': 'Stress (MPa)'})
    plotter_stress.view_xy()
    plotter_stress.set_background('white')
    plotter_stress.add_text("Step 3: Von Mises Stress", font_size=15, color='black')
    stress_filename = os.path.join(output_folder, "03_von_mises_stress.png")
    plotter_stress.screenshot(stress_filename)
    print(f"Saved stress plot to {stress_filename}")
    print("\n--- Script finished successfully! ---")

finally:
    # This block will run whether the script succeeds or fails, ensuring cleanup.
    gmsh.finalize()
    print("Gmsh finalized.")