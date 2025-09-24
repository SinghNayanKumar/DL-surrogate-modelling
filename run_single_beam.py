import numpy as np
import matplotlib.pyplot as plt
import pygmsh
import dolfinx
from mpi4py import MPI
import pyvista
import os

# --- Create an output directory for images ---
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)
print(f"Saving all plot outputs to '{output_folder}/'")

# Set PyVista to run in "headless" or "off-screen" mode for non-interactive environments
pyvista.set_plot_theme("document")
pyvista.global_theme.off_screen = True 

print(f"Using dolfinx version: {dolfinx.__version__}")

# =============================================================================
# STEP 0: DEFINE THE PARAMETERS FOR OUR SINGLE BEAM
# =============================================================================
print("--- Step 0: Defining Beam Parameters ---")
beam_length, beam_height, mesh_size = 100.0, 20.0, 3.0
youngs_modulus_E, poissons_ratio_nu = 210e3, 0.3
force_magnitude, force_angle_deg = 500.0, -60.0
DOMAIN_TAG, FIXED_WALL_TAG, LOAD_SURFACE_TAG = 1, 2, 3


# =============================================================================
# STEP 1: CREATE GEOMETRY & MESH USING PYGMSH
# =============================================================================
print("\n--- Step 1: Creating Geometry with pygmsh ---")
with pygmsh.occ.Geometry() as geom:
    p1 = geom.add_point([0, 0, 0], mesh_size)
    p2 = geom.add_point([beam_length, 0, 0], mesh_size)
    p3 = geom.add_point([beam_length, beam_height, 0], mesh_size)
    p4 = geom.add_point([0, beam_height, 0], mesh_size)
    l1, l2, l3, l4 = geom.add_line(p1, p2), geom.add_line(p2, p3), geom.add_line(p3, p4), geom.add_line(p4, p1)
    curve_loop = geom.add_curve_loop([l1, l2, l3, l4])
    surface = geom.add_plane_surface(curve_loop)
    geom.add_physical(surface, label=DOMAIN_TAG)
    geom.add_physical(l4, label=FIXED_WALL_TAG)
    geom.add_physical(l2, label=LOAD_SURFACE_TAG)
    print("Generating mesh...")
    mesh_data = geom.generate_mesh(verbose=False)
    print("Mesh generation complete.")

    cell_block = mesh_data[0]
    cells, nodes = cell_block.data, mesh_data[1]
    pv_mesh = pyvista.UnstructuredGrid({pyvista.CellType.TRIANGLE: cells}, nodes)
    
    plotter = pyvista.Plotter(window_size=[800, 300])
    plotter.add_mesh(pv_mesh, show_edges=True, style='wireframe')
    plotter.view_xy()
    plotter.set_background('white')
    plotter.add_text("Step 1: The Final Mesh", font_size=15, color='black')
    mesh_filename = os.path.join(output_folder, "01_mesh.png")
    plotter.screenshot(mesh_filename)
    print(f"Saved mesh plot to {mesh_filename}")

# =============================================================================
# STEP 2: SOLVE THE FEA PROBLEM USING FENICSX
# =============================================================================
print("\n--- Step 2: Solving FEA with FEniCSx ---")
import gmsh
gmsh.initialize()
gmsh.model.add("temp_model")
gmsh.model.importOCCString(geom.get_code())
domain, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()
V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
fixed_facets = facet_tags.find(FIXED_WALL_TAG)
u_fixed = np.array([0, 0], dtype=dolfinx.default_scalar_type)
bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(domain, u_fixed),dolfinx.fem.locate_dofs_topological(V, 1, fixed_facets),V)
traction = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type([0, 0]))
force_angle_rad = np.deg2rad(force_angle_deg)
traction.value[0] = force_magnitude * np.sin(force_angle_rad) / beam_height
traction.value[1] = -force_magnitude * np.cos(force_angle_rad) / beam_height
u = dolfinx.fem.Function(V)
v = dolfinx.fem.TestFunction(V)
mu = youngs_modulus_E / (2 * (1 + poissons_ratio_nu))
lambda_ = youngs_modulus_E * poissons_ratio_nu / ((1 + poissons_ratio_nu) * (1 - 2 * poissons_ratio_nu))
lambda_ = 2 * mu * lambda_ / (lambda_ + 2 * mu)
def epsilon(u): return dolfinx.fem.sym(dolfinx.fem.grad(u))
def sigma(u): return lambda_ * dolfinx.fem.tr(epsilon(u)) * dolfinx.fem.Identity(len(u)) + 2 * mu * epsilon(u)
ds = dolfinx.fem.Measure("ds", domain=domain, subdomain_data=facet_tags)
a = dolfinx.fem.inner(sigma(u), epsilon(v)) * dolfinx.fem.dx
L = dolfinx.fem.inner(traction, v) * ds(LOAD_SURFACE_TAG)
print("Solving the linear system...")
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
print("Solve complete.")

# =============================================================================
# STEP 3: VISUALIZE THE RESULTS
# =============================================================================
print("\n--- Step 3: Saving Result Visualizations ---")
topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid["u"] = uh.x.array.reshape((geometry.shape[0], 2))
warped = grid.warp_by_vector("u", factor=50)

plotter_def = pyvista.Plotter(window_size=[800, 400])
plotter_def.add_mesh(grid, style='wireframe', color='gray', opacity=0.5, label='Original')
plotter_def.add_mesh(warped, show_edges=True, label='Deformed')
plotter_def.add_text("Step 2: Deformation (scaled by 50x)", font_size=15, color='black')
plotter_def.view_xy()
plotter_def.set_background('white')
plotter_def.add_legend()
def_filename = os.path.join(output_folder, "02_deformation.png")
plotter_def.screenshot(def_filename)
print(f"Saved deformation plot to {def_filename}")

S = dolfinx.fem.TensorFunctionSpace(domain, ("DG", 0))
s_expr = dolfinx.fem.Expression(sigma(uh), S.element.interpolation_points())
s_projected = dolfinx.fem.Function(S)
s_projected.interpolate(s_expr)
s = s_projected
von_mises = np.sqrt(s.sub(0).sub(0)**2 - s.sub(0).sub(0)*s.sub(1).sub(1) + s.sub(1).sub(1)**2 + 3*s.sub(0).sub(1)**2)
V_von_mises = dolfinx.fem.FunctionSpace(domain, ("DG", 0))
topology_vm, cell_types_vm, geometry_vm = dolfinx.plot.create_vtk_mesh(V_von_mises)
grid_vm = pyvista.UnstructuredGrid(topology_vm, cell_types_vm, geometry_vm)
grid_vm.cell_data["Von Mises Stress (MPa)"] = von_mises.x.array

plotter_stress = pyvista.Plotter(window_size=[800, 300])
plotter_stress.add_mesh(grid_vm, show_edges=False, scalar_bar_args={'title': 'Stress (MPa)'})
plotter_stress.view_xy()
plotter_stress.set_background('white')
plotter_stress.add_text("Step 3: Von Mises Stress", font_size=15, color='black')
stress_filename = os.path.join(output_folder, "03_von_mises_stress.png")
plotter_stress.screenshot(stress_filename)
print(f"Saved stress plot to {stress_filename}")

print("\n--- Script finished successfully! ---")
