import scipy
import numpy as np
import helpers as h
import trimesh
def total_energy(configurations):
    elastic_energy = elastic_term(configurations)
    repulsive_energy = repulsive_term(configurations)
    return elastic_energy + repulsive_energy

def elastic_term(configurations):
    n = len(configurations) - 1
    total = 0.0
    for k in range(1, n+1):
        total += elastic_energy(configurations[k-1], configurations[k])
    return n * total

def elastic_energy(x_prev, x_curr):
    # Define the elastic energy between two configurations
    # Placeholder for the actual implementation of \hat{\mathcal{W}}
    return ...  # Compute the elastic energy
def repulsive_term(configurations,alpha=1.0):
    n = len(configurations) - 1
    total = 0.0
    for k in range(1, n+1):
        phi_diff = phi(configurations[k-1],alpha) - phi(configurations[k],alpha)
        total += phi_diff ** 2
    return n * total

#using boundary volume hierarchy
#distance threshold probably 1/4  remains to see but with bvh maybe follow the paper more explicitly
def phi(mesh, alpha=1.0, distance_threshold=np.inf):
    """
    Compute the discrete tangent-point energy of a mesh using helper functions and BVH.

    Parameters:
    - mesh: trimesh.Trimesh object
    - alpha: float, exponent parameter in the kernel
    - distance_threshold: float, maximum distance between triangle centroids to consider

    Returns:
    - phi_value: float, the tangent-point energy of the mesh
    """
    num_faces = len(mesh.faces)
    phi_value = 0.0

    # Build a k-d tree for triangle centroids
    centroids = [h.center(mesh, i) for i in range(num_faces)]
    kdtree = trimesh.kdtree.KDTree(centroids)

    for i in range(num_faces):
        # Find neighboring faces within the distance threshold
        current_centroid = centroids[i]
        neighbor_indices = kdtree.query_ball_point(current_centroid, distance_threshold)

        for j in neighbor_indices:
            if i == j:
                continue  # Skip same triangle

            # Check if triangles intersect
            if h.intersect(mesh, i, mesh, j):
                continue

            # Compute the energy contribution
            phi_ij = h.midpoint_approximation(mesh, i, mesh, j, alpha)
            phi_value += phi_ij

    return phi_value

def gradient_phi(mesh, alpha=1.0):
    """
    Compute the gradient of the discrete tangent-point energy with respect to the vertex positions.
    
    Parameters:
    - mesh: trimesh.Trimesh object
    - alpha: float, exponent parameter in the kernel
    
    Returns:
    - grad_phi: ndarray of shape (num_vertices, 3), gradient with respect to vertex positions
    """
    num_faces = len(mesh.faces)
    num_vertices = len(mesh.vertices)
    grad_phi = np.zeros((num_vertices, 3))  # Initialize gradient vector
    
    # Precompute areas, centers, normals for all faces
    areas = np.array([h.area(mesh, i) for i in range(num_faces)])
    centers = np.array([h.center(mesh, i) for i in range(num_faces)])
    normals = np.array([h.normal(mesh, i) for i in range(num_faces)])
    # Loop over all pairs of faces
    for i in range(num_faces):
        for j in range(num_faces):
            if i == j:
                continue  # Skip same triangle
            # Check if triangles intersect
            if h.intersect(mesh, i, mesh, j):
                print("intersection")
                continue  # Skip intersecting triangles
            print("no intersection")
            # Get areas, centers, normals
            a_i = areas[i]
            a_j = areas[j]
            c_i = centers[i]
            c_j = centers[j]
            n_i = normals[i]
            
            # Vector between centers
            diff = c_i - c_j
            norm_diff = np.linalg.norm(diff)
            if norm_diff == 0:
                continue  # Skip to avoid division by zero
            
            # Dot product between normal and diff
            dot_product = np.dot(n_i, diff)
            abs_dot = np.abs(dot_product)
                
            # Compute the partial derivatives
            # Compute common factors
            factor1 = alpha * (abs_dot ** (alpha - 1)) / (norm_diff ** (2 * alpha))
            factor2 = 2 * alpha * (abs_dot ** alpha) / (norm_diff ** (2 * alpha + 2))
            sign_dot = np.sign(dot_product)  # Sign of the dot product
            
            # Compute d_K w.r.t c_i
            d_K_c_i = (factor1 * sign_dot * n_i) - (factor2 * diff)
            
            # Compute the contribution to the gradient
            # Gradient w.r.t c_i
            grad_c_i = a_i * a_j * d_K_c_i
            
            # Distribute the gradient to the vertices of triangle i
            face_vertices_indices_i = mesh.faces[i]
            barycentric_weights_i = np.full(3, 1/3)  # Equal weights for centroid
            
            for k, vertex_index in enumerate(face_vertices_indices_i):
                grad_phi[vertex_index] += barycentric_weights_i[k] * grad_c_i  # Distribute equally
            
            # Since c_j depends on positions of vertices in triangle j, we need to compute the gradient w.r.t c_j
            # Compute d_K w.r.t c_j
            d_K_c_j = -d_K_c_i  # As per d_y K = -d_x K
            
            # Gradient w.r.t c_j
            grad_c_j = a_i * a_j * d_K_c_j
            
            # Distribute the gradient to the vertices of triangle j
            face_vertices_indices_j = mesh.faces[j]
            barycentric_weights_j = np.full(3, 1/3)  # Equal weights for centroid
            
            for k, vertex_index in enumerate(face_vertices_indices_j):
                grad_phi[vertex_index] += barycentric_weights_j[k] * grad_c_j  # Distribute equally
            
            # Additionally, we have to compute gradient w.r.t n_i (the normal of triangle i)
            # Compute d_K w.r.t n_i
            d_K_n_i = factor1 * sign_dot * diff
            
            # Since n_i depends on the positions of vertices in triangle i, we need to compute the derivative of n_i w.r.t vertex positions
            # Compute the gradient contribution from d_K_n_i to the vertices of triangle i
            # For simplicity, we'll assume that the normal vector changes negligibly with vertex positions (or can be approximated)
            # Alternatively, compute the derivative of n_i w.r.t vertex positions explicitly
            # For now, we may omit this term or include it if necessary with appropriate computation
            
    return grad_phi
def compute_gradient(configurations):
    # Compute the gradient of the total energy with respect to the configurations
    gradients = []
    n_steps = len(configurations) - 1

    for k in range(1, n_steps):
        #bizarre car pas besoin de trois termes pour calculer le gradient seulement les deux voisins #faux pour l'instant
        d_phi_k = gradient_phi(configurations[k])
        d_phi_k1 = gradient_phi(configurations[k + 1])
    #TODO
    return gradients

def gauss_newton_hessian(configurations):
    n_steps = len(configurations) - 1
    H_blocks = []

    for k in range(1, n_steps):
        d_phi_k = gradient_phi(configurations[k])
        d_phi_k1 = gradient_phi(configurations[k + 1])

        A_k = 4 * n_steps * np.outer(d_phi_k.flatten(), d_phi_k.flatten())
        B_k = -2 * n_steps * np.outer(d_phi_k.flatten(), d_phi_k1.flatten())

        H_blocks.append((A_k, B_k))

    H = assemble_block_tridiagonal(H_blocks)
    return H



def preconditioner(configurations):
    n_steps = len(configurations) - 1
    P_blocks = []

    for k in range(1, n_steps):
        P_k = hessian_elastic_energy(configurations[k - 1], configurations[k])
        P_blocks.append(P_k)

    # Assemble block-diagonal preconditioner
    P = scipy.sparse.block_diag(P_blocks)
    return P

def hessian_elastic_energy(mesh_prev, mesh_curr):
    # Compute the Hessian of the elastic energy with respect to mesh_curr
    # Placeholder implementation
    return ...  # Return Hessian as a sparse matrix

def assemble_block_tridiagonal(blocks):
    n = len(blocks)
    block_size = blocks[0][0].shape[0]
    total_size = block_size * n

    data = []
    rows = []
    cols = []

    for k, (A_k, B_k) in enumerate(blocks):
        idx = k * block_size

        # Diagonal block A_k
        data.append(A_k)
        rows.append(np.arange(idx, idx + block_size)[:, None])
        cols.append(np.arange(idx, idx + block_size))

        # Off-diagonal block B_k
        if k < n - 1:
            data.append(B_k)
            rows.append(np.arange(idx, idx + block_size)[:, None])
            cols.append(np.arange(idx + block_size, idx + 2 * block_size))

            # Transpose of B_k
            data.append(B_k.T)
            rows.append(np.arange(idx + block_size, idx + 2 * block_size)[:, None])
            cols.append(np.arange(idx, idx + block_size))

    # Create sparse matrix
    H = scipy.sparse.coo_matrix(
        (np.concatenate([d.flatten() for d in data]),
         (np.concatenate([r.flatten() for r in rows]),
          np.concatenate([c.flatten() for c in cols]))),
        shape=(total_size, total_size)
    ).tocsr()

    return H

import numpy as np
import trimesh

def bending_energy(mesh_undeformed, mesh_deformed):
    """
    Compute the bending energy of the mesh.

    Parameters:
    - mesh_undeformed: trimesh.Trimesh object, undeformed configuration
    - mesh_deformed: trimesh.Trimesh object, deformed configuration

    Returns:
    - total_energy: float, total bending energy
    """
    total_energy = 0.0
    num_edges = len(mesh_undeformed.edges_unique)
    
    # For each unique edge in the mesh
    for edge_index in range(num_edges):
        # Get the edge and its adjacent faces in the undeformed mesh
        edge = mesh_undeformed.edges_unique[edge_index]
        face_indices = mesh_undeformed.edges_face[edge_index]

        # Skip boundary edges (edges with only one adjacent face)
        if -1 in face_indices:
            continue

        # Get the indices of the two adjacent faces
        face1_index, face2_index = face_indices

        # Get the vertices of the adjacent faces
        face1_vertices = mesh_undeformed.faces[face1_index]
        face2_vertices = mesh_undeformed.faces[face2_index]

        # Identify the four vertices: i, j, k, l
        shared_vertices = np.intersect1d(face1_vertices, face2_vertices)
        if len(shared_vertices) != 2:
            continue  # Should not happen in a well-formed mesh

        j, k = shared_vertices
        # The unique vertices from each face
        i = list(set(face1_vertices) - set(shared_vertices))[0]
        l = list(set(face2_vertices) - set(shared_vertices))[0]

        # Compute |E| (edge length)
        E_length = np.linalg.norm(mesh_undeformed.vertices[j] - mesh_undeformed.vertices[k])

        # Compute areas of adjacent faces
        A_k = h.area(mesh_undeformed, face1_index)
        A_l = h.area(mesh_undeformed, face2_index)
        A_E = A_k + A_l

        # Dihedral angles in undeformed and deformed configurations
        theta_E = h.dihedral_angle(mesh_undeformed, edge_index)
        theta_E_tilde = h.dihedral_angle(mesh_deformed, edge_index)

        # Energy contribution from this edge
        energy_E = ((theta_E - theta_E_tilde) ** 2) * (E_length ** 2) / A_E
        total_energy += energy_E

    return total_energy
def dihedral_angle_derivative(mesh, edge_index):
    """
    Compute the gradient of the dihedral angle with respect to vertex positions.

    Parameters:
    - mesh: trimesh.Trimesh object
    - edge_index: int

    Returns:
    - grad_theta: dict, keys are vertex indices, values are ndarray gradients (3,)
    """
    grad_theta = {}
    edge = mesh.edges_unique[edge_index]
    face_indices = mesh.edges_face[edge_index]

    if -1 in face_indices:
        return grad_theta  # No gradient for boundary edges

    face1_index, face2_index = face_indices
    face1_vertices = mesh.faces[face1_index]
    face2_vertices = mesh.faces[face2_index]

    # Shared vertices are j and k
    shared_vertices = np.intersect1d(face1_vertices, face2_vertices)
    j, k = shared_vertices

    # Unique vertices
    i = list(set(face1_vertices) - set(shared_vertices))[0]
    l = list(set(face2_vertices) - set(shared_vertices))[0]

    # Positions of the vertices
    p_i = mesh.vertices[i]
    p_j = mesh.vertices[j]
    p_k = mesh.vertices[k]
    p_l = mesh.vertices[l]

    # Edge vector E
    E = p_k - p_j
    E_length = np.linalg.norm(E)
    if E_length == 0:
        return grad_theta  # Avoid division by zero

    # Normals of adjacent faces
    N_k = h.normal(mesh, face1_index)
    N_l = h.normal(mesh, face2_index)

    # Areas of adjacent faces
    A_k = h.area(mesh, face1_index)
    A_l = h.area(mesh, face2_index)

    # Compute gradients with respect to each vertex
    # Using provided equations:
    # ∇_k θ_E = (|E| / (2 A_k)) N_k
    # ∇_l θ_E = -(|E| / (2 A_l)) N_l
    # For vertices i and j, more complex expressions are required

    # Gradient with respect to vertex k
    grad_theta[k] = (E_length / (2 * A_k)) * N_k

    # Gradient with respect to vertex l
    grad_theta[l] = -(E_length / (2 * A_l)) * N_l

    # For vertices i and j, we can use approximate methods or compute as needed
    # For simplicity, we can assume their contributions are negligible or can be approximated

    return grad_theta

def bending_energy_gradient(mesh_undeformed, mesh_deformed):
    """
    Compute the gradient of the bending energy with respect to vertex positions.

    Parameters:
    - mesh_undeformed: trimesh.Trimesh object
    - mesh_deformed: trimesh.Trimesh object

    Returns:
    - grad_energy: ndarray of shape (num_vertices, 3)
    """
    num_vertices = len(mesh_deformed.vertices)
    grad_energy = np.zeros((num_vertices, 3))

    num_edges = len(mesh_undeformed.edges_unique)
    
    for edge_index in range(num_edges):
        # Get the edge and its adjacent faces in the undeformed mesh
        edge = mesh_undeformed.edges_unique[edge_index]
        face_indices = mesh_undeformed.edges_face[edge_index]

        if -1 in face_indices:
            continue  # Skip boundary edges

        face1_index, face2_index = face_indices
        face1_vertices = mesh_undeformed.faces[face1_index]
        face2_vertices = mesh_undeformed.faces[face2_index]

        # Identify the four vertices: i, j, k, l
        shared_vertices = np.intersect1d(face1_vertices, face2_vertices)
        j, k = shared_vertices
        i = list(set(face1_vertices) - set(shared_vertices))[0]
        l = list(set(face2_vertices) - set(shared_vertices))[0]

        # Compute |E| (edge length) in the undeformed mesh
        E_length = np.linalg.norm(mesh_undeformed.vertices[j] - mesh_undeformed.vertices[k])
        if E_length == 0:
            continue  # Avoid division by zero

        # Compute areas of adjacent faces in undeformed mesh
        A_k = h.area(mesh_undeformed, face1_index)
        A_l = h.area(mesh_undeformed, face2_index)
        A_E = A_k + A_l

        # Dihedral angles in undeformed and deformed configurations
        theta_E = h.dihedral_angle(mesh_undeformed, edge_index)
        theta_E_tilde = h.dihedral_angle(mesh_deformed, edge_index)

        # Difference in dihedral angles
        theta_diff = theta_E - theta_E_tilde

        # Energy contribution from this edge
        energy_E = (theta_diff ** 2) * (E_length ** 2) / A_E

        # Compute gradient of dihedral angle in deformed mesh
        grad_theta = dihedral_angle_derivative(mesh_deformed, edge_index)

        # Gradient of energy with respect to vertex positions
        # ∇_p E_b = -2 (θ_E - θ̃_E) (|E|^2 / A_E) ∇_p θ̃_E

        coef = -2 * theta_diff * (E_length ** 2) / A_E

        for vertex_index, grad_theta_p in grad_theta.items():
            grad_energy[vertex_index] += coef * grad_theta_p

    return grad_energy


