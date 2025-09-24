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
output_folder = "output_images_3d"
os.makedirs(output_folder, exist_ok=True)
print(f"Saving all plot outputs to '{output_folder}/'")

pyvista.set_plot_theme("document")

print(f"Using dolfinx version: {dolfinx.__version__}")

# =============================================================================
# STEP 0: DEFINE THE PARAMETERS
# =============================================================================
print("--- Step 0: Defining I-Beam Parameters ---")
# --- Geometric Parameters ---
beam_length = 300.0
flange_width = 100.0
flange_thickness = 15.0
web_thickness = 10.0
beam_depth = 150.0
fillet_radius = 12.0
mesh_size = 3.0

# --- Material and Load Parameters ---
youngs_modulus_E = 210e3
poissons_ratio_nu = 0.3
force_magnitude = 1500.0  # Total downward force

# --- Gmsh Physical Group Tags ---
DOMAIN_TAG = 1
FIXED_WALL_TAG = 2
LOAD_SURFACE_TAG = 3

# =============================================================================
# MAIN LOGIC
# =============================================================================
gmsh.initialize()
try:
    # --- GEOMETRY ---
    print("\n--- Step 1: Creating 3D I-Beam Geometry with gmsh (OCC) ---")
    gmsh.model.add("i_beam_3d")

    # Define the I-beam cross-section using rectangles
    h_beam = beam_depth / 2.0
    h_flange = flange_width / 2.0

    top_flange = gmsh.model.occ.addRectangle(-h_flange, h_beam - flange_thickness, 0, flange_width, flange_thickness)
    bottom_flange = gmsh.model.occ.addRectangle(-h_flange, -h_beam, 0, flange_width, flange_thickness)
    web = gmsh.model.occ.addRectangle(-web_thickness / 2.0, -h_beam + flange_thickness, 0, web_thickness, beam_depth - 2 * flange_thickness)

    # Fuse the parts together to form the cross-section
    cross_section, _ = gmsh.model.occ.fuse([(2, top_flange), (2, bottom_flange)], [(2, web)])
    gmsh.model.occ.synchronize()

  # Extrude the 2D surface to create a 3D volume
    surface_dim_tag = cross_section[0]
    extrusion = gmsh.model.occ.extrude([surface_dim_tag], 0, 0, beam_length)
    gmsh.model.occ.synchronize()

    # Identify the volume and the four long edges to fillet
    volumes = gmsh.model.occ.getEntities(dim=3)
    if not volumes:
        raise Exception("Extrusion failed to create a volume")
    volume_tag = volumes[0][1]

    # We identify the edges to fillet by their location. They are the four
    # longitudinal edges at the intersection of the web and the flanges.
    edges_to_fillet = []
    y_fillet_coord = beam_depth / 2.0 - flange_thickness
    x_fillet_coord = web_thickness / 2.0

    all_curves = gmsh.model.occ.getEntities(dim=1)
    for curve_dim, curve_tag in all_curves:
        # Get the bounding box of the curve
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(curve_dim, curve_tag)
        # Check if it's a longitudinal edge
        if abs(zmax - zmin - beam_length) < 1e-6:
            # Check if its x,y coordinates match the fillet locations
            if abs(abs(xmin) - x_fillet_coord) < 1e-6 and abs(abs(ymin) - y_fillet_coord) < 1e-6:
                edges_to_fillet.append(curve_tag)

    # Apply the fillet to the volume's edges. The old volume is removed by default.
    filleted_entities = gmsh.model.occ.fillet([volume_tag], edges_to_fillet, [fillet_radius])
    gmsh.model.occ.synchronize()
    
    # After filleting, tags may have changed. We must find the new volume and
    # the end-cap surfaces for assigning physical groups.
    final_volumes = gmsh.model.occ.getEntities(dim=3)
    if not final_volumes:
        raise Exception("Fillet operation failed to produce a volume")
    final_volume_tag = final_volumes[0][1]

    fixed_wall_tag = -1
    load_surface_tag = -1
    final_surfaces = gmsh.model.occ.getEntities(dim=2)
    for s_dim, s_tag in final_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(s_dim, s_tag)
        # Find the surface at z=0 (fixed wall)
        if abs(zmin) < 1e-6 and abs(zmax) < 1e-6:
            fixed_wall_tag = s_tag
        # Find the surface at z=beam_length (load surface)
        elif abs(zmin - beam_length) < 1e-6 and abs(zmax - beam_length) < 1e-6:
            load_surface_tag = s_tag

    if fixed_wall_tag == -1 or load_surface_tag == -1:
        raise Exception("Could not identify fixed and load surfaces after filleting.")

    # Add Physical Groups
    gmsh.model.addPhysicalGroup(3, [final_volume_tag], DOMAIN_TAG)
    gmsh.model.addPhysicalGroup(2, [fixed_wall_tag], FIXED_WALL_TAG)
    gmsh.model.addPhysicalGroup(2, [load_surface_tag], LOAD_SURFACE_TAG)

    print("Generating 3D mesh...")
    gmsh.model.mesh.generate(3)
    #gmsh.model.mesh.optimize("Netgen")
    print("Mesh generation complete.")

    # --- VISUALIZATION 1 (3D MESH) ---
    # Gmsh does not provide a direct way to get all 3D element types at once for pyvista
    # We will visualize the mesh after loading it into DOLFINx
    
    # --- STEP 2: SOLVE THE FEA PROBLEM ---
    print("\n--- Step 2: Solving FEA with FEniCSx ---")
    domain, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=3
    )

    # For the vector space
    vector_element = basix.ufl.element(
        "Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,)
    )
    V = dolfinx.fem.functionspace(domain, vector_element)

    # --- Boundary Conditions ---
    # Fixed wall at z=0
    fixed_facets = facet_tags.find(FIXED_WALL_TAG)
    u_fixed = np.array([0, 0, 0], dtype=dolfinx.default_scalar_type)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(domain, u_fixed),
        dolfinx.fem.locate_dofs_topological(V, 2, fixed_facets), # 2 for facets in 3D
        V
    )

    # --- Traction (force) on the other end ---
    # Calculate cross-section area to apply force as traction (Force/Area)
    cross_section_area = flange_width * flange_thickness * 2 + (beam_depth - 2 * flange_thickness) * web_thickness
    traction_value = -force_magnitude / cross_section_area
    traction = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type([0, traction_value, 0]))

    # --- Weak Form ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    mu = youngs_modulus_E / (2 * (1 + poissons_ratio_nu))
    lambda_ = youngs_modulus_E * poissons_ratio_nu / ((1 + poissons_ratio_nu) * (1 - 2 * poissons_ratio_nu))
    
    def epsilon(u): return ufl.sym(ufl.grad(u))
    def sigma(u): return lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
    
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(traction, v) * ds(LOAD_SURFACE_TAG)
    
    print("Solving the linear system...")
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_solver_type": "mumps"}
    )
    uh = problem.solve()
    print("Solve complete.")

    # --- STEP 3: VISUALIZE THE RESULTS ---
    print("\n--- Step 3: Saving Result Visualizations ---")
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Add displacement vectors to the grid
    grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))

    # --- Plot the initial mesh ---
    plotter_mesh = pyvista.Plotter(off_screen=True, window_size=[800, 600])
    plotter_mesh.add_mesh(grid, show_edges=True, style='wireframe', color='gray')
    plotter_mesh.view_isometric()
    plotter_mesh.set_background('white')
    plotter_mesh.add_text("Step 1: The Final 3D Mesh", font_size=15, color='black')
    mesh_filename = os.path.join(output_folder, "01_mesh_3d.png")
    plotter_mesh.screenshot(mesh_filename)
    print(f"Saved mesh plot to {mesh_filename}")


    # --- Plot deformation ---
    warped = grid.warp_by_vector("u", factor=200) # Increased factor for visibility
    plotter_def = pyvista.Plotter(off_screen=True, window_size=[800, 600])
    plotter_def.add_mesh(grid, style='wireframe', color='gray', opacity=0.5, label='Original')
    plotter_def.add_mesh(warped, show_edges=True, label='Deformed')
    plotter_def.add_text("Step 2: Deformation (scaled by 200x)", font_size=15, color='black')
    plotter_def.view_isometric()
    plotter_def.set_background('white')
    plotter_def.add_legend()
    def_filename = os.path.join(output_folder, "02_deformation_3d.png")
    plotter_def.screenshot(def_filename)
    print(f"Saved deformation plot to {def_filename}")

    # --- Plot Von Mises Stress ---
    # Define Von Mises stress in 3D
    s = sigma(uh)
    von_mises = ufl.sqrt(0.5 * ((s[0, 0] - s[1, 1])**2 + (s[1, 1] - s[2, 2])**2 + (s[2, 2] - s[0, 0])**2) \
                       + 3 * (s[0, 1]**2 + s[1, 2]**2 + s[2, 0]**2))

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

    plotter_stress = pyvista.Plotter(off_screen=True, window_size=[800, 600])
    plotter_stress.add_mesh(grid_vm, show_edges=False, scalar_bar_args={'title': 'Stress (MPa)'})
    plotter_stress.view_isometric()
    plotter_stress.set_background('white')
    plotter_stress.add_text("Step 3: Von Mises Stress", font_size=15, color='black')
    stress_filename = os.path.join(output_folder, "03_von_mises_stress_3d.png")
    plotter_stress.screenshot(stress_filename)
    print(f"Saved stress plot to {stress_filename}")
    print("\n--- Script finished successfully! ---")

finally:
    # This block will run whether the script succeeds or fails, ensuring cleanup.
    gmsh.finalize()
    print("Gmsh finalized.")