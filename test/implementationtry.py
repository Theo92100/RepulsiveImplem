import trimesh
import numpy as np
from scipy.optimize import minimize # For optimization
import trimesh.viewer
from scipy.spatial import cKDTree
import open3d as o3d


# # Charger les maillages avec Open3D
# mesh_start_o3d = o3d.io.read_triangle_mesh('./Meshes/hand/0.off')
# mesh_end_o3d = o3d.io.read_triangle_mesh('./Meshes/hand/1.off')

# # Simplifier les maillages
# target_face_count = 1000  # Ajustez selon vos besoins
# mesh_start_simplified = mesh_start_o3d.simplify_quadric_decimation(target_face_count)
# mesh_end_simplified = mesh_end_o3d.simplify_quadric_decimation(target_face_count)

# # Sauvegarder les maillages simplifiés
# o3d.io.write_triangle_mesh('./Meshes/hand/0_simplified.off', mesh_start_simplified)
# o3d.io.write_triangle_mesh('./Meshes/hand/1_simplified.off', mesh_end_simplified)

# Charger les maillages simplifiés avec Trimesh
mesh_start = trimesh.load('./Meshes/hand/0_simplified.off')
mesh_end = trimesh.load('./Meshes/hand/1_simplified.off')

# Extraire les sommets et les faces
V_start = mesh_start.vertices
F = mesh_start.faces
V_end = mesh_end.vertices
#assert np.array_equal(F, mesh_end.faces), "Les maillages de départ et d'arrivée doivent avoir les mêmes faces"

def compute_reference_metrics(V_ref, F):
    # Compute the reference edge vectors for each triangle
    v0 = V_ref[F[:, 0], :]
    v1 = V_ref[F[:, 1], :]
    v2 = V_ref[F[:, 2], :]
    e1_ref = v1 - v0  # Edge from vertex 0 to 1
    e2_ref = v2 - v0  # Edge from vertex 0 to 2
    return e1_ref, e2_ref

def compute_deformed_metrics(V_def, F):
    # Compute the deformed edge vectors for each triangle
    v0 = V_def[F[:, 0], :]
    v1 = V_def[F[:, 1], :]
    v2 = V_def[F[:, 2], :]
    e1_def = v1 - v0
    e2_def = v2 - v0
    return e1_def, e2_def

def membrane_energy(V_ref, V_def, F, mu=1.0, lam=1.0):
    """
    V_ref: Reference vertex positions (n_vertices, 3)
    V_def: Deformed vertex positions (n_vertices, 3)
    F: Faces (n_faces, 3)
    mu, lam: Material properties (Lamé parameters)
    """
    e1_ref, e2_ref = compute_reference_metrics(V_ref, F)
    e1_def, e2_def = compute_deformed_metrics(V_def, F)

    # Compute reference and deformed metric tensors
    I_ref = np.stack([
        np.einsum('ij,ij->i', e1_ref, e1_ref),
        np.einsum('ij,ij->i', e1_ref, e2_ref),
        np.einsum('ij,ij->i', e2_ref, e2_ref)
    ], axis=1)  # (n_faces, 3)

    I_def = np.stack([
        np.einsum('ij,ij->i', e1_def, e1_def),
        np.einsum('ij,ij->i', e1_def, e2_def),
        np.einsum('ij,ij->i', e2_def, e2_def)
    ], axis=1)  # (n_faces, 3)

    # Compute strain tensor C = I_ref^{-1} * I_def
    # For 2x2 matrices, inverse and multiplication can be computed efficiently
    # Define I_ref as 2x2 matrices
    I_ref_matrices = np.array([
        [I_ref[:, 0], I_ref[:, 1]],
        [I_ref[:, 1], I_ref[:, 2]]
    ])  # Shape (2, 2, n_faces)

    # Similarly for I_def
    I_def_matrices = np.array([
        [I_def[:, 0], I_def[:, 1]],
        [I_def[:, 1], I_def[:, 2]]
    ])  # Shape (2, 2, n_faces)

    # Compute inverses of I_ref
    det_I_ref = I_ref[:, 0] * I_ref[:, 2] - I_ref[:, 1] ** 2
    inv_I_ref = np.array([
        [I_ref[:, 2], -I_ref[:, 1]],
        [-I_ref[:, 1], I_ref[:, 0]]
    ]) / det_I_ref  # Shape (2, 2, n_faces)

    # Compute C = inv_I_ref * I_def
    C = np.einsum('ijk,ikl->ijl', inv_I_ref.transpose(2, 0, 1), I_def_matrices.transpose(2, 0, 1))  # (n_faces, 2, 2)

    # Compute energy density using neo-Hookean model
    # W(C) = mu * (trace(C) - 2) - mu * log(det(C)) + (lam/2) * (log(det(C)))^2
    trace_C = C[:, 0, 0] + C[:, 1, 1]
    det_C = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
    log_det_C = np.log(det_C)

    W = mu * (trace_C - 2) - mu * log_det_C + (lam / 2) * log_det_C ** 2

    # Compute area of each triangle in the reference configuration
    cross_ref = np.cross(e1_ref, e2_ref)
    area_ref = 0.5 * np.linalg.norm(cross_ref, axis=1)

    # Total membrane energy
    E_membrane = np.sum(area_ref * W)
    return E_membrane

def bending_energy(V_ref, V_def, F, delta=1.0):
    """
    V_ref: Positions des sommets dans la configuration de référence (n_vertices, 3)
    V_def: Positions des sommets dans la configuration déformée (n_vertices, 3)
    F: Faces (n_faces, 3)
    delta: Paramètre d'épaisseur (δ^3)
    """
    # Créer les maillages de référence et déformé
    mesh_ref = trimesh.Trimesh(vertices=V_ref, faces=F, process=False)
    mesh_def = trimesh.Trimesh(vertices=V_def, faces=F, process=False)

    # Obtenir les arêtes internes et les faces adjacentes
    adjacent_faces = mesh_ref.face_adjacency          # (n_edges, 2)
    interior_edges = mesh_ref.face_adjacency_edges    # (n_edges, 2)

    # Calculer les normales des faces dans la configuration de référence
    normals_ref = mesh_ref.face_normals
    n1_ref = normals_ref[adjacent_faces[:, 0]]
    n2_ref = normals_ref[adjacent_faces[:, 1]]

    # Calculer les vecteurs d'arête dans la configuration de référence
    edge_vectors_ref = V_ref[interior_edges[:, 1]] - V_ref[interior_edges[:, 0]]
    edge_vectors_ref /= np.linalg.norm(edge_vectors_ref, axis=1)[:, np.newaxis]  # Normaliser

    # Calculer les angles dièdres dans la configuration de référence
    cos_theta_ref = np.einsum('ij,ij->i', n1_ref, n2_ref)
    sin_theta_ref = np.einsum('ij,ij->i', np.cross(n1_ref, n2_ref), edge_vectors_ref)
    theta_ref = np.arctan2(sin_theta_ref, cos_theta_ref)

    # Répéter pour la configuration déformée
    normals_def = mesh_def.face_normals
    n1_def = normals_def[adjacent_faces[:, 0]]
    n2_def = normals_def[adjacent_faces[:, 1]]
    edge_vectors_def = V_def[interior_edges[:, 1]] - V_def[interior_edges[:, 0]]
    edge_vectors_def /= np.linalg.norm(edge_vectors_def, axis=1)[:, np.newaxis]

    cos_theta_def = np.einsum('ij,ij->i', n1_def, n2_def)
    sin_theta_def = np.einsum('ij,ij->i', np.cross(n1_def, n2_def), edge_vectors_def)
    theta_def = np.arctan2(sin_theta_def, cos_theta_def)

    # Calculer les longueurs des arêtes
    edge_lengths = np.linalg.norm(V_ref[interior_edges[:, 1]] - V_ref[interior_edges[:, 0]], axis=1)

    # Calculer les aires associées aux arêtes
    face_areas_ref = mesh_ref.area_faces
    areas = (face_areas_ref[adjacent_faces[:, 0]] + face_areas_ref[adjacent_faces[:, 1]]) / 2

    # Calculer l'énergie de flexion
    W_bending = delta**3 * ((theta_def - theta_ref)**2) * (edge_lengths / areas)
    E_bending = np.sum(W_bending)

    return E_bending

# def tangent_point_energy(V_def, F, alpha=2):
#     """
#     V_def: Deformed vertex positions (n_vertices, 3)
#     F: Faces (n_faces, 3)
#     alpha: Controls the strength of repulsion
#     """
#     n_faces = F.shape[0]

#     # Compute triangle properties
#     v0 = V_def[F[:, 0], :]
#     v1 = V_def[F[:, 1], :]
#     v2 = V_def[F[:, 2], :]
#     centroids = (v0 + v1 + v2) / 3  # (n_faces, 3)

#     # Compute normals
#     normals = np.cross(v1 - v0, v2 - v0)
#     normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

#     # Compute areas
#     areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

#     # Initialize energy
#     E_TPE = 0.0

#     # Compute pairwise interactions
#     for i in range(n_faces):
#         for j in range(n_faces):
#             if i == j:
#                 continue
#             # Compute vector between centroids
#             d = centroids[i] - centroids[j]
#             dist = np.linalg.norm(d)
#             if dist == 0:
#                 continue  # Avoid division by zero
#             numerator = np.abs(np.dot(normals[i], d)) ** alpha
#             denominator = dist ** (2 * alpha)
#             E_TPE += areas[i] * areas[j] * numerator / denominator

#     return E_TPE

def tangent_point_energy(V_def, F, alpha=2, cutoff_distance=0.1):
    """
    V_def: Deformed vertex positions (n_vertices, 3)
    F: Faces (n_faces, 3)
    alpha: Controls the strength of repulsion
    cutoff_distance: Only consider face pairs within this distance
    """
    # Compute triangle centroids
    v0 = V_def[F[:, 0]]
    v1 = V_def[F[:, 1]]
    v2 = V_def[F[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0  # Shape: (n_faces, 3)
    
    # Compute normals
    normals = np.cross(v1 - v0, v2 - v0)
    normals_norm = np.linalg.norm(normals, axis=1)
    normals = normals / normals_norm[:, np.newaxis]
    
    # Compute areas
    areas = 0.5 * normals_norm  # Shape: (n_faces,)
    
    # Build KD-Tree of centroids
    tree = cKDTree(centroids)
    
    # Query pairs within cutoff distance
    pairs = tree.query_pairs(cutoff_distance)
    
    if not pairs:
        return 0.0  # No interactions within cutoff distance
    
    pairs = np.array(list(pairs))
    idx_i = pairs[:, 0]
    idx_j = pairs[:, 1]
    
    # Vectorized computation
    d = centroids[idx_i] - centroids[idx_j]
    dist = np.linalg.norm(d, axis=1)
    
    # Avoid division by zero
    eps = 1e-8
    dist = np.maximum(dist, eps)
    
    dot_products = np.abs(np.einsum('ij,ij->i', normals[idx_i], d)) ** alpha
    denom = dist ** (2 * alpha)
    energy_contribs = areas[idx_i] * areas[idx_j] * dot_products / denom
    
    E_TPE = np.sum(energy_contribs)
    return E_TPE


def total_energy(V_ref, V_def, F, mu=1.0, lam=1.0, t=1.0, alpha=2, beta=1.0):
    E_membrane = membrane_energy(V_ref, V_def, F, mu, lam)
    E_bending = bending_energy(V_ref,V_def, F, t)
    E_TPE = tangent_point_energy(V_def, F, alpha)
    E_total = E_membrane + E_bending + beta * E_TPE
    return E_total

def smoothness_energy(V_prev, V_curr, V_next, w=1.0):
    """
    V_prev, V_curr, V_next: Vertex positions at consecutive time steps
    w: Weight of the smoothness term
    """
    energy = w * np.sum((V_next - 2 * V_curr + V_prev) ** 2)
    return energy

def total_path_energy(V_vars,gamma=1.0):
    # V_vars: Flattened vertex positions of intermediate meshes
    V_vars_split = np.split(V_vars, n_time_steps - 2)
    energies = 0.0

    # Include the fixed start and end meshes
    V_all = [V_start] + [V_var.reshape(-1, 3) for V_var in V_vars_split] + [V_end]

    for k in range(1, n_time_steps - 1):
        V_prev = V_all[k - 1]
        V_curr = V_all[k]
        V_next = V_all[k + 1]
        assert V_prev.ndim == 2 and V_prev.shape[1] == 3, f"V_prev a une forme incorrecte: {V_prev.shape} {k}"
        assert V_curr.ndim == 2 and V_curr.shape[1] == 3, f"V_curr a une forme incorrecte: {V_curr.shape} {k}"
        assert V_next.ndim == 2 and V_next.shape[1] == 3, f"V_next a une forme incorrecte: {V_next.shape}{k}"
        E_tot=total_energy(V_prev, V_curr, F)
        #E_smooth = smoothness_energy(V_prev, V_curr, V_next)

        energies += E_tot #+ gamma * E_smooth

    return energies

n_time_steps = 5 # Adjust as needed
tau = np.linspace(0, 1, n_time_steps)
V_path = []
for t in tau:
    V_t = (1 - t) * V_start + t * V_end
    V_path.append(V_t)
# Exclude the start and end meshes (fixed)
V_initial = []
for V_t in V_path[1:-1]:
    V_initial.append(V_t.flatten())
V_initial = np.concatenate(V_initial)

# Initial guess for the optimization
V_vars_initial = V_initial.copy()

# Optimization parameters
result = minimize(total_path_energy, V_vars_initial, method='L-BFGS-B')
V_vars_optimized = result.x
V_vars_split = np.split(V_vars_optimized, n_time_steps - 2)
V_optimized_path = [V_start] + [V_var.reshape(-1, 3) for V_var in V_vars_split] + [V_end]

# Create mesh objects for visualization
mesh_sequence = []
for V_t in V_optimized_path:
    mesh_t = trimesh.Trimesh(vertices=V_t, faces=F)
    mesh_sequence.append(mesh_t)

# Simple visualization loop
for mesh_t in mesh_sequence:
    mesh_t.show()