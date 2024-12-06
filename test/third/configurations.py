import trimesh
import numpy as np

# Number of time steps we still have to get the configurations by a first piecewise interpolation ( i think)
n = ...  # Number of time steps (excluding initial and final configurations)

# Load initial and final meshes
mesh_initial = trimesh.load('path_to_initial_mesh.obj')
mesh_final = trimesh.load('path_to_final_mesh.obj')

# Ensure the meshes have the same topology
assert np.array_equal(mesh_initial.faces, mesh_final.faces), "Meshes must have the same topology"

# Create a list to store configurations
configurations = [mesh_initial]

# Initialize intermediate configurations as copies of the initial mesh
for _ in range(n - 1):
    configurations.append(mesh_initial.copy())

configurations.append(mesh_final)
