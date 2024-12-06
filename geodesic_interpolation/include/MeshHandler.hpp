#ifndef MESH_HANDLER_HPP
#define MESH_HANDLER_HPP

#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <string>
#include <vector>
#include <goast/Core.h>

typedef OpenMesh::TriMesh_ArrayKernelT<TriTraits> Mesh;
typedef Eigen::VectorXd VectorType;


class MeshHandler {
public:
    bool load_mesh(const std::string &filename, Mesh &mesh);
    bool save_mesh(const std::string &filename, const Mesh &mesh);
    std::vector<Mesh> initialize_intermediates(const Mesh &initial, const Mesh &final, int num_intermediates);
  
    // Méthodes pour extraire les données du maillage
    std::vector<double> get_vertices(const Mesh &mesh) const;
    std::vector<int> get_simplices(const Mesh &mesh) const;
    // Méthode pour obtenir la géométrie du maillage
    VectorType get_geometry(const Mesh &mesh) const;

    // Méthode pour obtenir la topologie du maillage
    MeshTopologySaver get_topology(const Mesh &mesh) const;

    // Méthode pour obtenir la géométrie de référence
    VectorType get_reference_geometry(const Mesh &mesh) const;

    // Méthode pour charger le maillage de référence
    Mesh load_reference_mesh();
};

#endif // MESH_HANDLER_HPP