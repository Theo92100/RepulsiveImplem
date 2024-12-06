import trimesh 
import numpy as np
def area(mesh, face_index):
    """
    Calculate the area of a triangle face in the mesh.

    Parameters:
    - mesh: trimesh.Trimesh object
    - face_index: int, index of the face in the mesh.faces array

    Returns:
    - area: float, area of the triangle face
    """
    return mesh.area_faces[face_index]
def center(mesh, face_index):
    """
    Calculate the centroid of a triangle face in the mesh.

    Parameters:
    - mesh: trimesh.Trimesh object
    - face_index: int

    Returns:
    - centroid: ndarray of shape (3,), coordinates of the centroid
    """
    face_vertices = mesh.vertices[mesh.faces[face_index]]
    return np.mean(face_vertices, axis=0)
def normal(mesh, face_index):
    """
    Get the normal vector of a triangle face in the mesh.

    Parameters:
    - mesh: trimesh.Trimesh object
    - face_index: int

    Returns:
    - normal_vector: ndarray of shape (3,), unit normal vector of the face
    """
    return mesh.face_normals[face_index]

def diam(mesh, face_index):
    """
    Calculate the maximum edge length (diameter) of a triangle face.

    Parameters:
    - mesh: trimesh.Trimesh object
    - face_index: int

    Returns:
    - diameter: float, maximum edge length of the triangle
    """
    face_vertices = mesh.vertices[mesh.faces[face_index]]
    x1, x2, x3 = face_vertices
    d12 = np.linalg.norm(x2 - x1)
    d23 = np.linalg.norm(x3 - x2)
    d31 = np.linalg.norm(x1 - x3)
    return max(d12, d23, d31)

def intersect(mesh1, face_index1, mesh2, face_index2):
    """
    Check if two triangle faces intersect.

    Parameters:
    - mesh1: trimesh.Trimesh object
    - face_index1: int, index of the face in mesh1
    - mesh2: trimesh.Trimesh object
    - face_index2: int, index of the face in mesh2

    Returns:
    - intersects: bool, True if triangles intersect, False otherwise
    """
    # Extract triangles as individual meshes
    face_vertices1 = mesh1.vertices[mesh1.faces[face_index1]]
    triangle1 = trimesh.Trimesh(vertices=face_vertices1, faces=[[0, 1, 2]], process=False)

    face_vertices2 = mesh2.vertices[mesh2.faces[face_index2]]
    triangle2 = trimesh.Trimesh(vertices=face_vertices2, faces=[[0, 1, 2]], process=False)

    # Use trimesh's collision detection
    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object('triangle1', triangle1)
    collision_manager.add_object('triangle2', triangle2)
    return collision_manager.in_collision_internal()

def dist(mesh1, face_index1, mesh2, face_index2):
    """
    Calculate the minimal distance between two triangle faces.

    Parameters:
    - mesh1: trimesh.Trimesh object
    - face_index1: int
    - mesh2: trimesh.Trimesh object
    - face_index2: int

    Returns:
    - min_distance: float, minimal distance between the two triangles
    """
    # Extract triangles as individual meshes
    face_vertices1 = mesh1.vertices[mesh1.faces[face_index1]]
    triangle1 = trimesh.Trimesh(vertices=face_vertices1, faces=[[0, 1, 2]], process=False)

    face_vertices2 = mesh2.vertices[mesh2.faces[face_index2]]
    triangle2 = trimesh.Trimesh(vertices=face_vertices2, faces=[[0, 1, 2]], process=False)

    # Use ProximityQuery to find the minimal distance
    pq = trimesh.proximity.ProximityQuery(triangle1)
    distances, _ = pq.vertex(triangle2.vertices)
    min_distance = np.min(distances)

    return min_distance

def bary(tau_tilde_vertices, tau_vertices):
    """
    Calculate barycentric coordinates of tau_tilde's vertices with respect to tau.

    Parameters:
    - tau_tilde_vertices: ndarray of shape (3, 3), vertices of tau_tilde
    - tau_vertices: ndarray of shape (3, 3), vertices of tau

    Returns:
    - bary_coords: ndarray of shape (3, 3), barycentric coordinates
    """
    # Use trimesh's function to compute barycentric coordinates
    bary_coords = trimesh.triangles.points_to_barycentric(triangles=tau_vertices, points=tau_tilde_vertices)
    return bary_coords.T  # Return as a 3x3 matrix

def midpoint_approximation(mesh1, face_index1, mesh2, face_index2, alpha):
    """
    Compute the tangent-point energy approximation between two triangle faces.

    Parameters:
    - mesh1: trimesh.Trimesh object
    - face_index1: int
    - mesh2: trimesh.Trimesh object
    - face_index2: int
    - alpha: float, exponent parameter

    Returns:
    - phi: float, approximation of the tangent-point energy
    """
    # Compute areas
    a_sigma = area(mesh1, face_index1)
    a_tau = area(mesh2, face_index2)

    # Compute centers
    c_sigma = center(mesh1, face_index1)
    c_tau = center(mesh2, face_index2)

    # Compute normal of sigma
    n_sigma = normal(mesh1, face_index1)

    # Vector between centers
    diff = c_sigma - c_tau
    norm_diff = np.linalg.norm(diff)

    # Avoid division by zero
    if norm_diff == 0:
        return 0

    # Dot product between n_sigma and diff
    dot_product = np.dot(n_sigma, diff)

    # Compute the approximation
    numerator = dot_product * (norm_diff ** alpha)
    denominator = norm_diff ** (2 * alpha)
    phi = (a_sigma * a_tau * numerator) / denominator

    return phi
def subdivide(mesh, face_index):
    """
    Subdivide a triangle face into four smaller triangles.

    Parameters:
    - mesh: trimesh.Trimesh object
    - face_index: int

    Returns:
    - new_mesh: trimesh.Trimesh object containing the subdivided triangles
    """
    # Get the vertices of the face
    face = mesh.faces[face_index]
    x1_idx, x2_idx, x3_idx = face
    x1 = mesh.vertices[x1_idx]
    x2 = mesh.vertices[x2_idx]
    x3 = mesh.vertices[x3_idx]

    # Compute midpoints
    m1 = (x1 + x2) / 2
    m2 = (x2 + x3) / 2
    m3 = (x3 + x1) / 2

    # New vertices array
    new_vertices = np.array([x1, x2, x3, m1, m2, m3])

    # Indices in new_vertices
    idx_x1, idx_x2, idx_x3, idx_m1, idx_m2, idx_m3 = range(6)

    # Define new faces using indices in new_vertices
    face1 = [idx_x1, idx_m1, idx_m3]
    face2 = [idx_x2, idx_m2, idx_m1]
    face3 = [idx_x3, idx_m3, idx_m2]
    face4 = [idx_m1, idx_m2, idx_m3]

    new_faces = np.array([face1, face2, face3, face4])

    # Create new mesh
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    return new_mesh

def dihedral_angle(mesh, face_index):
    """
    Compute the dihedral angle of a triangle face in the mesh.

    Parameters:
    - mesh: trimesh.Trimesh object
    - face_index: int

    Returns:
    - angle: float, dihedral angle of the face
    """
    # Get the vertices of the face
    face = mesh.faces[face_index]
    x1_idx, x2_idx, x3_idx = face
    x1 = mesh.vertices[x1_idx]
    x2 = mesh.vertices[x2_idx]
    x3 = mesh.vertices[x3_idx]

    # Compute edge vectors
    e1 = x2 - x1
    e2 = x3 - x2
    e3 = x1 - x3

    # Compute normalized edge vectors
    e1_norm = e1 / np.linalg.norm(e1)
    e2_norm = e2 / np.linalg.norm(e2)
    e3_norm = e3 / np.linalg.norm(e3)

    # Compute dihedral angles
    angle1 = np.arccos(np.dot(-e1_norm, e3_norm))
    angle2 = np.arccos(np.dot(-e2_norm, e1_norm))
    angle3 = np.arccos(np.dot(-e3_norm, e2_norm))

    # Return the average of the three angles
    return (angle1 + angle2 + angle3) / 3