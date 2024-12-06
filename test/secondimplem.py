import trimesh
import numpy as np
from scipy.optimize import minimize # For optimization
import trimesh.viewer
from scipy.spatial import cKDTree
import open3d as o3d
from collections import deque
import numpy as np
from bounding import BVHNode

hat_delta = ...  # Valeur de \hat{\Delta}
delta = ...      # Valeur initiale de Δ₀ (entre 0 et hat_delta)
eta = ...        # Valeur de η (entre 0 et 0.25)
x_k = ...        # Valeur initiale de x₀

def compute_p_k(x_k, delta_k):
    # Approximation de la solution de (4.3)
    # Cette fonction doit être définie selon le problème spécifique
    p_k = ...
    return p_k

def compute_rho_k(x_k, p_k):
    # Calcul de ρₖ à partir de (4.4)
    rho_k = ...
    return rho_k

def convergence_criteria_not_met(k, x_k):
    # Définir les critères de convergence appropriés
    return True  # Modifier selon les critères spécifiques

k = 0
while convergence_criteria_not_met(k, x_k):
    p_k = compute_p_k(x_k, delta)
    rho_k = compute_rho_k(x_k, p_k)

    if rho_k < 0.25:
        delta = 0.25 * delta
    else:
        if rho_k > 0.75 and np.linalg.norm(p_k) == delta:
            delta = min(2 * delta, hat_delta)
        else:
            delta = delta

    if rho_k > eta:
        x_k = x_k + p_k
    else:
        x_k = x_k

    k += 1

def area(sigma):
    x1, x2, x3 = sigma
    return 0.5 * np.linalg.norm(np.cross(x2 - x1, x3 - x1))

def center(sigma):
    x1, x2, x3 = sigma
    return (x1 + x2 + x3) / 3

def normal(sigma):
    x1, x2, x3 = sigma
    n = np.cross(x2 - x1, x3 - x1)
    return n / np.linalg.norm(n)

def diam(sigma):
    x1, x2, x3 = sigma
    d12 = np.linalg.norm(x2 - x1)
    d13 = np.linalg.norm(x3 - x1)
    d23 = np.linalg.norm(x3 - x2)
    return max(d12, d13, d23)

def intersect(sigma, tau):
    # Implémentation simplifiée pour vérifier l'intersection entre deux triangles en 3D
    # Cette fonction nécessite une implémentation détaillée selon l'application
    pass

def dist(sigma, tau):
    # Calcul de la distance minimale entre deux triangles en 3D
    # Cette fonction peut utiliser des algorithmes avancés ou des bibliothèques spécialisées
    pass

def bary(tau_tilde, tau):
    # Calcul des coordonnées barycentriques des sommets de tau_tilde par rapport à tau
    x1, x2, x3 = tau
    T = np.vstack((x1, x2, x3)).T
    T = np.vstack((T, [1,1,1]))  # Ajouter une ligne pour gérer les coordonnées homogènes
    bary_coords = []
    for vertex in tau_tilde:
        v = np.append(vertex, 1)
        coords = np.linalg.solve(T, v)
        bary_coords.append(coords)
    return np.array(bary_coords).T  # Matrice 3x3

def midpoint_approximation(sigma, tau, alpha):
    # Calcul des aires des triangles
    a_sigma = area(sigma)
    a_tau = area(tau)
    
    # Calcul des centres des triangles
    c_sigma = center(sigma)
    c_tau = center(tau)
    
    # Calcul de la normale du triangle sigma
    n_sigma = normal(sigma)
    
    # Vecteur entre les centres
    diff = c_sigma - c_tau
    norm_diff = np.linalg.norm(diff)
    
    # Produit scalaire entre n_sigma et diff
    dot_product = np.dot(n_sigma, diff)
    
    # Calcul de l'approximation de l'énergie tangent-point
    numerator = dot_product * (norm_diff ** alpha)
    denominator = norm_diff ** (2 * alpha)
    phi = a_sigma * a_tau * numerator / denominator
    
    return phi

def subdivide(sigma):
    # Obtenir les sommets du triangle
    x1, x2, x3 = sigma

    # Calculer les points milieux des arêtes
    m1 = (x1 + x2) / 2
    m2 = (x2 + x3) / 2
    m3 = (x3 + x1) / 2

    # Définir les quatre triangles obtenus
    sigma_1 = (x1, m1, m3)
    sigma_2 = (x2, m2, m1)
    sigma_3 = (x3, m3, m2)
    sigma_4 = (m1, m2, m3)

    return [sigma_1, sigma_2, sigma_3, sigma_4]

import numpy as np

#theta= 1/4
def adaptive_multipole(sigma0, tau0, alpha, theta):
    # Vérifier si les triangles s'intersectent
    if intersect(sigma0, tau0):
        return float('inf'), None, None

    # Initialiser l'énergie et les dérivées
    Phi = 0.0
    d_sigma_Phi = np.zeros((3, 3))  # Pour les 3 sommets de sigma0
    d_tau_Phi = np.zeros((3, 3))    # Pour les 3 sommets de tau0

    # Initialiser la pile avec la paire initiale
    S = [(sigma0, tau0)]

    while S:
        sigma, tau = S.pop()

        # Vérifier si les triangles s'intersectent
        if intersect(sigma, tau):
            continue  # Ignorer les paires qui s'intersectent

        # Calculer Diam(σ) et Diam(τ)
        diam_sigma = diam(sigma)
        diam_tau = diam(tau)

        # Calculer Dist(σ, τ)
        dist_sigma_tau = dist(sigma, tau)

        # Critère d'acceptation multipolaire
        if max(diam_sigma, diam_tau) < theta * dist_sigma_tau:
            # Approximations par point milieu
            phi = midpoint_approximation(sigma, tau, alpha)
            Phi += phi

            # Calcul des dérivées
            d_phi_d_sigma = d_sigma_midpoint_approximation(sigma, tau, alpha)
            d_phi_d_tau = d_tau_midpoint_approximation(sigma, tau, alpha)

            # Calcul des coordonnées barycentriques
            bary_sigma = bary(sigma, sigma0)  # Matrice 3x3
            bary_tau = bary(tau, tau0)        # Matrice 3x3

            # Mise à jour des dérivées totales
            d_sigma_Phi += d_phi_d_sigma @ bary_sigma
            d_tau_Phi += d_phi_d_tau @ bary_tau
        else:
            # Subdiviser les triangles
            sigma_subdivided = subdivide(sigma)
            tau_subdivided = subdivide(tau)

            # Ajouter toutes les combinaisons à la pile
            for sigma_i in sigma_subdivided:
                for tau_j in tau_subdivided:
                    if not intersect(sigma_i, tau_j):
                        S.append((sigma_i, tau_j))

    return Phi, d_sigma_Phi, d_tau_Phi

#faux
def d_sigma_midpoint_approximation(sigma, tau, alpha):
    d_phi_d_sigma = np.zeros((3, 3))  # 3 sommets, chacun avec 3 coordonnées

    eps = 1e-6
    for i in range(3):  # Pour chaque sommet de sigma
        for j in range(3):  # Pour chaque coordonnée x, y, z
            sigma_perturbed = [vertex.copy() for vertex in sigma]
            sigma_perturbed[i] = sigma_perturbed[i].copy()
            sigma_perturbed[i][j] += eps
            phi_original = midpoint_approximation(sigma, tau, alpha)
            phi_perturbed = midpoint_approximation(sigma_perturbed, tau, alpha)
            derivative = (phi_perturbed - phi_original) / eps
            d_phi_d_sigma[i][j] = derivative

    return d_phi_d_sigma
#faux
def d_tau_midpoint_approximation(sigma, tau, alpha):
    d_phi_d_tau = np.zeros((3, 3))  # 3 sommets, chacun avec 3 coordonnées

    eps = 1e-6
    for i in range(3):  # Pour chaque sommet de tau
        for j in range(3):  # Pour chaque coordonnée x, y, z
            tau_perturbed = [vertex.copy() for vertex in tau]
            tau_perturbed[i] = tau_perturbed[i].copy()
            tau_perturbed[i][j] += eps
            phi_original = midpoint_approximation(sigma, tau, alpha)
            phi_perturbed = midpoint_approximation(sigma, tau_perturbed, alpha)
            derivative = (phi_perturbed - phi_original) / eps
            d_phi_d_tau[i][j] = derivative

    return d_phi_d_tau


def evaluate_potential_energy(bvh_U, bvh_V, alpha, theta=0.25):
    """
    Évalue l'énergie globale en traversant le BVH et en appliquant le critère MAC.
    
    :param bvh_U: Nœud racine du BVH pour l'ensemble U.
    :param bvh_V: Nœud racine du BVH pour l'ensemble V.
    :param alpha: Puissance pour l'énergie tangent-point.
    :param theta: Paramètre pour le critère d'acceptation multipolaire (MAC).
    :return: Energie totale, dérivées par rapport à U et V.
    """
    Phi_total = 0.0
    d_Phi_total_U = np.zeros((len(bvh_U.vertices), 3))  # Supposant que chaque nœud a un attribut 'vertices'
    d_Phi_total_V = np.zeros((len(bvh_V.vertices), 3))
    
    # Utiliser une file pour parcourir les paires de nœuds
    queue = deque()
    queue.append((bvh_U, bvh_V))
    
    while queue:
        node_U, node_V = queue.popleft()
        
        # Vérifier si les boîtes englobantes satisfont le MAC
        if node_U.aabb.satisfies_MAC(node_V.aabb, theta):
            # Calculer la distance minimale entre les boîtes
            min_dist = node_U.aabb.min_distance(node_V.aabb)
            
            # Obtenir les données agrégées
            a_sigma = node_U.aggregate_data['area']
            a_tau = node_V.aggregate_data['area']
            c_sigma = node_U.aggregate_data['center']
            c_tau = node_V.aggregate_data['center']
            n_sigma = node_U.aggregate_data['normal']
            
            # Calculer Phi via l'approximation multipolaire
            phi = a_sigma * a_tau * np.dot(n_sigma, (c_sigma - c_tau)) / (min_dist ** alpha)
            
            Phi_total += phi
            
            # Calculer les dérivées (approximations simplifiées)
            # Vous devrez définir comment les dérivées agrégées sont calculées
            # Ici, nous utilisons des dérivées centrales simplifiées
            d_phi_d_sigma = (a_tau / (min_dist ** alpha)) * n_sigma
            d_phi_d_tau = (-a_sigma / (min_dist ** alpha)) * n_sigma
            
            # Agréger les dérivées
            d_Phi_total_U += d_phi_d_sigma
            d_Phi_total_V += d_phi_d_tau
        
        else:
            # Si le nœud n'est pas une feuille, ajouter les enfants
            if not node_U.is_leaf() and not node_V.is_leaf():
                for child_U in node_U.children:
                    for child_V in node_V.children:
                        queue.append((child_U, child_V))
            elif node_U.is_leaf() and not node_V.is_leaf():
                for child_V in node_V.children:
                    queue.append((node_U, child_V))
            elif not node_U.is_leaf() and node_V.is_leaf():
                for child_U in node_U.children:
                    queue.append((child_U, node_V))
            else:
                # Les deux sont des feuilles, appliquer l'approximation adaptative si elles ne partagent pas de vertex ou d'arête
                if share_vertex_or_edge(node_U, node_V):
                    # Utiliser l'approximation standard midpoint
                    phi = midpoint_approximation(node_U.vertices, node_V.vertices, alpha)
                    Phi_total += phi
                    
                    # Calculer les dérivées via des approximations numériques
                    d_phi_d_sigma = d_sigma_midpoint_approximation(node_U.vertices, node_V.vertices, alpha)
                    d_phi_d_tau = d_tau_midpoint_approximation(node_U.vertices, node_V.vertices, alpha)
                    
                    d_Phi_total_U += d_phi_d_sigma
                    d_Phi_total_V += d_phi_d_tau
                else:
                    # Utiliser l'approximation adaptative multipole
                    Phi, d_sigma_Phi, d_tau_Phi = adaptive_multipole(node_U.vertices, node_V.vertices, alpha, theta)
                    Phi_total += Phi
                    if Phi < float('inf'):
                        d_Phi_total_U += d_sigma_Phi
                        d_Phi_total_V += d_tau_Phi
                    # Si Phi est infini, ignorer ou gérer autrement
                    
    return Phi_total, d_Phi_total_U, d_Phi_total_V

def share_vertex_or_edge(node_U, node_V):
    """
    Vérifie si deux nœuds partagent un vertex ou une arête.
    
    :param node_U: Nœud U.
    :param node_V: Nœud V.
    :return: True si ils partagent un vertex ou une arête, False sinon.
    """
    vertices_U = set(map(tuple, node_U.vertices))
    vertices_V = set(map(tuple, node_V.vertices))
    
    shared_vertices = vertices_U.intersection(vertices_V)
    
    if len(shared_vertices) > 0:
        return True
    
    # Vérifier les arêtes partagées
    edges_U = set()
    edges_V = set()
    
    for i in range(3):
        edge = tuple(sorted((tuple(node_U.vertices[i]), tuple(node_U.vertices[(i+1)%3]))))
        edges_U.add(edge)
    
    for i in range(3):
        edge = tuple(sorted((tuple(node_V.vertices[i]), tuple(node_V.vertices[(i+1)%3]))))
        edges_V.add(edge)
    
    shared_edges = edges_U.intersection(edges_V)
    
    if len(shared_edges) > 0:
        return True
    
    return False



