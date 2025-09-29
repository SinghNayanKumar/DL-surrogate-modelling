import os
import argparse
import h5py
import numpy as np
import gmsh
import dolfinx
from mpi4py import MPI
import pyvista
import basix.ufl
import ufl
import dolfinx.fem.petsc

# =============================================================================
# STEP 0: DEFINE AND PARSE PARAMETERS
# =============================================================================
print("--- Step 0: Parsing Simulation Parameters ---")
parser = argparse.ArgumentParser(description="Run a parametric 3D I-Beam simulation.")
# --- Add arguments for every parameter you want to vary ---
parser.add_argument("--sim_id", type=int, default=0, help="Simulation ID number for file naming")
# Geometry
parser.add_argument("--beam_length", type=float, default=300.0)
parser.add_argument("--flange_width", type=float, default=100.0)
parser.add_argument("--flange_thickness", type=float, default=15.0)
parser.add_argument("--web_thickness", type=float, default=10.0)
parser.add_argument("--beam_depth", type=float, default=150.0)
parser.add_argument("--fillet_radius", type=float, default=12.0)
# Material
parser.add_argument("--youngs_modulus", type=float, default=210e3)
parser.add_argument("--poissons_ratio", type=float, default=0.3)
# Load
parser.add_argument("--force_magnitude", type=float, default=1500.0)
parser.add_argument("--load_type", type=str, default="bending_y", choices=["bending_y", "bending_x", "torsion"])
parser.add_argument("--load_distribution", type=str, default="uniform", choices=["uniform", "linear_y"])

args = parser.parse_args()

# --- Assign args to variables for clarity in the rest of the script ---
beam_length = args.beam_length
flange_width = args.flange_width
flange_thickness = args.flange_thickness
web_thickness = args.web_thickness
beam_depth = args.beam_depth
fillet_radius = args.fillet_radius
youngs_modulus_E = args.youngs_modulus
poissons_ratio_nu = args.poissons_ratio
force_magnitude = args.force_magnitude
sim_id = args.sim_id

# --- Create an output directory for images ---
output_folder = "output_images_3d"
os.makedirs(output_folder, exist_ok=True)
print(f"Saving all plot outputs for sim {sim_id} to '{output_folder}/'")

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
    # ... (This entire section is unchanged) ...
    print("\n--- Step 1: Creating 3D I-Beam Geometry with gmsh (OCC) ---")
    gmsh.model.add(f"i_beam_3d_{sim_id}")

    h_beam = beam_depth / 2.0
    h_flange = flange_width / 2.0

    top_flange = gmsh.model.occ.addRectangle(-h_flange, h_beam - flange_thickness, 0, flange_width, flange_thickness)
    bottom_flange = gmsh.model.occ.addRectangle(-h_flange, -h_beam, 0, flange_width, flange_thickness)
    web = gmsh.model.occ.addRectangle(-web_thickness / 2.0, -h_beam + flange_thickness, 0, web_thickness, beam_depth - 2 * flange_thickness)

    cross_section, _ = gmsh.model.occ.fuse([(2, top_flange), (2, bottom_flange)], [(2, web)])
    gmsh.model.occ.synchronize()

    surface_dim_tag = cross_section[0]
    extrusion = gmsh.model.occ.extrude([surface_dim_tag], 0, 0, beam_length)
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.occ.getEntities(dim=3)
    if not volumes: raise Exception("Extrusion failed")
    volume_tag = volumes[0][1]

    edges_to_fillet = []
    y_fillet_coord = beam_depth / 2.0 - flange_thickness
    x_fillet_coord = web_thickness / 2.0

    all_curves = gmsh.model.occ.getEntities(dim=1)
    for curve_dim, curve_tag in all_curves:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(curve_dim, curve_tag)
        if abs(zmax - zmin - beam_length) < 1e-6:
            if abs(abs(xmin) - x_fillet_coord) < 1e-6 and abs(abs(ymin) - y_fillet_coord) < 1e-6:
                edges_to_fillet.append(curve_tag)

    filleted_entities = gmsh.model.occ.fillet([volume_tag], edges_to_fillet, [fillet_radius])
    gmsh.model.occ.synchronize()
    
    final_volumes = gmsh.model.occ.getEntities(dim=3)
    if not final_volumes: raise Exception("Fillet failed")
    final_volume_tag = final_volumes[0][1]

    fixed_wall_tag = -1
    load_surface_tag = -1
    final_surfaces = gmsh.model.occ.getEntities(dim=2)
    for s_dim, s_tag in final_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(s_dim, s_tag)
        if abs(zmin) < 1e-6 and abs(zmax) < 1e-6:
            fixed_wall_tag = s_tag
        elif abs(zmin - beam_length) < 1e-6 and abs(zmax - beam_length) < 1e-6:
            load_surface_tag = s_tag

    if fixed_wall_tag == -1 or load_surface_tag == -1:
        raise Exception("Could not identify fixed and load surfaces.")

    gmsh.model.addPhysicalGroup(3, [final_volume_tag], DOMAIN_TAG)
    gmsh.model.addPhysicalGroup(2, [fixed_wall_tag], FIXED_WALL_TAG)
    gmsh.model.addPhysicalGroup(2, [load_surface_tag], LOAD_SURFACE_TAG)

    print("Generating 3D mesh...")
    gmsh.model.mesh.generate(3)
    print("Mesh generation complete.")

    # --- STEP 2: SOLVE THE FEA PROBLEM ---
    # ... (This entire section is unchanged) ...
    print("\n--- Step 2: Solving FEA with FEniCSx ---")
    domain, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=3
    )

    vector_element = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = dolfinx.fem.functionspace(domain, vector_element)

    fixed_facets = facet_tags.find(FIXED_WALL_TAG)
    u_fixed = np.array([0, 0, 0], dtype=dolfinx.default_scalar_type)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(domain, u_fixed),
        dolfinx.fem.locate_dofs_topological(V, 2, fixed_facets), V
    )

    cross_section_area = flange_width * flange_thickness * 2 + (beam_depth - 2 * flange_thickness) * web_thickness
    base_traction_magnitude = -force_magnitude / cross_section_area
    x = ufl.SpatialCoordinate(domain)
    
    if args.load_distribution == "uniform":
        traction_magnitude = base_traction_magnitude
    elif args.load_distribution == "linear_y":
        traction_magnitude = 2 * base_traction_magnitude * (x[1] / beam_depth)

    if args.load_type == "bending_y":
        traction_vector = ufl.as_vector([0, traction_magnitude, 0])
    elif args.load_type == "bending_x":
        traction_vector = ufl.as_vector([traction_magnitude, 0, 0])
    elif args.load_type == "torsion":
        torsion_magnitude = -force_magnitude / cross_section_area
        traction_vector = ufl.as_vector([-torsion_magnitude * x[1] / h_flange, torsion_magnitude * x[0] / h_flange, 0])
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    mu = youngs_modulus_E / (2 * (1 + poissons_ratio_nu))
    lambda_ = youngs_modulus_E * poissons_ratio_nu / ((1 + poissons_ratio_nu) * (1 - 2 * poissons_ratio_nu))
    def epsilon(u): return ufl.sym(ufl.grad(u))
    def sigma(u): return lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
    
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(traction_vector, v) * ds(LOAD_SURFACE_TAG)
    
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_solver_type": "mumps"})
    uh = problem.solve()

    # --- STEP 3: VISUALIZE THE RESULTS ---
    print("\n--- Step 3: Saving Result Visualizations ---")
    pyvista.set_plot_theme("document")
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))

    # ### --- CHANGE 1: Changed warp factor and added colormap --- ###
    # Use a factor of 1.0 for a true-to-scale deformation plot.
    warped = grid.warp_by_vector("u", factor=20) 
    plotter_def = pyvista.Plotter(off_screen=True, window_size=[800, 600])
    plotter_def.add_mesh(grid, style='wireframe', color='gray', opacity=0.3)
    # Added cmap="jet" for a vibrant blue-to-red color map.
    plotter_def.add_mesh(warped, show_edges=False, scalars=np.linalg.norm(grid['u'], axis=1), cmap="jet")
    
    plotter_def.add_text(f"Deformation (sim {sim_id})", font_size=15)
    plotter_def.view_isometric(); plotter_def.set_background('white')
    def_filename = os.path.join(output_folder, f"sim_{sim_id:04d}_deformation.png")
    plotter_def.screenshot(def_filename)

    s = sigma(uh)
    von_mises = ufl.sqrt(0.5 * ((s[0, 0] - s[1, 1])**2 + (s[1, 1] - s[2, 2])**2 + (s[2, 2] - s[0, 0])**2) \
                       + 3 * (s[0, 1]**2 + s[1, 2]**2 + s[2, 0]**2))
    V_von_mises = dolfinx.fem.functionspace(domain, ("DG", 0))
    von_mises_func = dolfinx.fem.Function(V_von_mises)
    vm_expr = dolfinx.fem.Expression(von_mises, V_von_mises.element.interpolation_points())
    von_mises_func.interpolate(vm_expr)

    topology_vm, cell_types_vm, geometry_vm = dolfinx.plot.vtk_mesh(domain)
    grid_vm = pyvista.UnstructuredGrid(topology_vm, cell_types_vm, geometry_vm)
    grid_vm.cell_data["Von Mises Stress (MPa)"] = von_mises_func.x.array
    
    plotter_stress = pyvista.Plotter(off_screen=True, window_size=[800, 600])
    
    # ### --- CHANGE 2: Added colormap for the stress plot --- ###
    plotter_stress.add_mesh(grid_vm, show_edges=False, scalar_bar_args={'title': 'Stress (MPa)'}, cmap="jet")

    plotter_stress.view_isometric(); plotter_stress.set_background('white')
    plotter_stress.add_text(f"Von Mises Stress (sim {sim_id})", font_size=15)
    stress_filename = os.path.join(output_folder, f"sim_{sim_id:04d}_stress.png")
    plotter_stress.screenshot(stress_filename)
    print(f"Saved plots to {def_filename} and {stress_filename}")

    # --- STEP 4: EXPORT DATA FOR MACHINE LEARNING ---
    # ... (This entire section is unchanged) ...
    print("\n--- Step 4: Exporting Data for Machine Learning ---")
    data_output_folder = "output_data_3d"
    os.makedirs(data_output_folder, exist_ok=True)
    h5_filename = os.path.join(data_output_folder, f"simulation_{sim_id:04d}.h5")

    with h5py.File(h5_filename, 'w') as f:
        f.attrs['sim_id'] = sim_id
        for key, value in vars(args).items():
            f.attrs[key] = value
        
        f.create_dataset('node_coordinates', data=geometry)
        f.create_dataset('topology', data=topology)
        f.create_dataset('displacement', data=uh.x.array.reshape((geometry.shape[0], 3)))
        f.create_dataset('von_mises_stress', data=von_mises_func.x.array)

    print(f"Successfully saved ML data to {h5_filename}")
    print("\n--- Script finished successfully! ---")

finally:
    gmsh.finalize()
    print("Gmsh finalized.")