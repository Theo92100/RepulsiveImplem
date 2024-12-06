import numpy as np

class BVHNode:
    def __init__(self, triangles, depth=0, max_depth=20, min_triangles=2):
        """
        Initialise un nœud BVH.

        :param triangles: Liste des triangles, chaque triangle étant un tuple de trois vecteurs numpy (x1, x2, x3).
        :param depth: Profondeur actuelle de l'arbre.
        :param max_depth: Profondeur maximale de l'arbre.
        :param min_triangles: Nombre minimal de triangles pour continuer la subdivision.
        """
        self.triangles = triangles
        self.depth = depth
        self.max_depth = max_depth
        self.min_triangles = min_triangles
        self.left = None
        self.right = None
        self.aabb_min = np.zeros(3)
        self.aabb_max = np.zeros(3)
        self.total_area = 0.0
        self.center_of_mass = np.zeros(3)
        self.average_normal = np.zeros(3)

        self.build()

    def build(self):
        """
        Construit récursivement le BVH en subdivisant les triangles.
        """
        # Calculer la boîte englobante AABB
        self.aabb_min, self.aabb_max = self.compute_aabb(self.triangles)

        # Calculer l'aire totale, le centre de masse pondéré et la normale moyenne pondérée
        self.total_area, self.center_of_mass, self.average_normal = self.compute_properties(self.triangles)

        # Critère d'arrêt
        if self.depth >= self.max_depth or len(self.triangles) <= self.min_triangles:
            return  # Nœud feuille

        # Déterminer l'axe de division (l'axe avec la plus grande extension)
        extents = self.aabb_max - self.aabb_min
        split_axis = np.argmax(extents)

        # Trier les triangles selon le centre sur l'axe de division
        self.triangles.sort(key=lambda tri: self.center(tri)[split_axis])

        # Diviser les triangles en deux groupes
        mid = len(self.triangles) // 2
        left_triangles = self.triangles[:mid]
        right_triangles = self.triangles[mid:]

        # Créer les nœuds enfants
        self.left = BVHNode(left_triangles, depth=self.depth + 1, max_depth=self.max_depth, min_triangles=self.min_triangles)
        self.right = BVHNode(right_triangles, depth=self.depth + 1, max_depth=self.max_depth, min_triangles=self.min_triangles)

        # Libérer la liste des triangles pour les nœuds internes
        self.triangles = None

    @staticmethod
    def compute_aabb(triangles):
        """
        Calcule la boîte englobante axis-aligned (AABB) pour une liste de triangles.

        :param triangles: Liste des triangles.
        :return: Tuple (bbox_min, bbox_max) où chaque élément est un vecteur numpy de taille 3.
        """
        bbox_min = np.full(3, np.inf)
        bbox_max = np.full(3, -np.inf)

        for tri in triangles:
            for vertex in tri:
                bbox_min = np.minimum(bbox_min, vertex)
                bbox_max = np.maximum(bbox_max, vertex)

        return bbox_min, bbox_max

    @staticmethod
    def compute_properties(triangles):
        """
        Calcule l'aire totale, le centre de masse pondéré et la normale moyenne pondérée des triangles.

        :param triangles: Liste des triangles.
        :return: Tuple (total_area, center_of_mass, average_normal).
        """
        total_area = 0.0
        weighted_center = np.zeros(3)
        weighted_normal = np.zeros(3)

        for tri in triangles:
            a = BVHNode.area(tri)
            c = BVHNode.center(tri)
            n = BVHNode.normal(tri)
            total_area += a
            weighted_center += a * c
            weighted_normal += a * n

        if total_area > 0:
            center_of_mass = weighted_center / total_area
            average_normal = weighted_normal / np.linalg.norm(weighted_normal)
        else:
            center_of_mass = np.zeros(3)
            average_normal = np.zeros(3)

        return total_area, center_of_mass, average_normal

    @staticmethod
    def area(tri):
        """
        Calcule l'aire d'un triangle.

        :param tri: Tuple de trois vecteurs numpy (x1, x2, x3).
        :return: Aire du triangle.
        """
        x1, x2, x3 = tri
        return 0.5 * np.linalg.norm(np.cross(x2 - x1, x3 - x1))

    @staticmethod
    def center(tri):
        """
        Calcule le centre du triangle.

        :param tri: Tuple de trois vecteurs numpy (x1, x2, x3).
        :return: Centre du triangle.
        """
        x1, x2, x3 = tri
        return (x1 + x2 + x3) / 3.0

    @staticmethod
    def normal(tri):
        """
        Calcule la normale unitaire du triangle.

        :param tri: Tuple de trois vecteurs numpy (x1, x2, x3).
        :return: Normale unitaire du triangle.
        """
        x1, x2, x3 = tri
        n = np.cross(x2 - x1, x3 - x1)
        norm = np.linalg.norm(n)
        if norm == 0:
            return np.zeros(3)
        return n / norm

    def is_leaf(self):
        """
        Vérifie si le nœud est une feuille.

        :return: True si feuille, False sinon.
        """
        return self.left is None and self.right is None

    def __repr__(self):
        return (f"BVHNode(depth={self.depth}, total_area={self.total_area:.2f}, "
                f"center_of_mass={self.center_of_mass}, average_normal={self.average_normal}, "
                f"aabb_min={self.aabb_min}, aabb_max={self.aabb_max}, "
                f"num_triangles={'N/A' if self.is_leaf() else '0'})")

def load_off(file_path):
    """
    Charge un fichier OFF et retourne les triangles.

    :param file_path: Chemin vers le fichier OFF.
    :return: Liste des triangles, chaque triangle étant un tuple de trois vecteurs numpy.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if lines[0].strip() != 'OFF':
        raise ValueError("Le fichier n'est pas au format OFF.")

    parts = lines[1].strip().split()
    num_vertices = int(parts[0])
    num_faces = int(parts[1])

    vertices = []
    for i in range(2, 2 + num_vertices):
        vertex = np.array(list(map(float, lines[i].strip().split())))
        vertices.append(vertex)

    triangles = []
    for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
        parts = lines[i].strip().split()
        if int(parts[0]) != 3:
            continue  # Ignorer les polygones qui ne sont pas des triangles
        idx1, idx2, idx3 = map(int, parts[1:4])
        tri = (vertices[idx1], vertices[idx2], vertices[idx3])
        triangles.append(tri)

    return triangles

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les triangles depuis un fichier OFF
    file_path = './Meshes/hand/0_simplified.off'
    mesh_triangles = load_off(file_path)

    # Construire le BVH
    bvh_root = BVHNode(mesh_triangles)

    # Afficher des informations sur le BVH
    print(bvh_root)
    if not bvh_root.is_leaf():
        print(bvh_root.left)
        print(bvh_root.right)