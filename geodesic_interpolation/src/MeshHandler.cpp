#include "MeshHandler.hpp"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <iostream>
#include <vector>

bool MeshHandler::load_mesh(const std::string &filename, Mesh &mesh) {
    if (!OpenMesh::IO::read_mesh(mesh, filename)) {
        std::cerr << "Erreur : Impossible de lire le maillage depuis " << filename << std::endl;
        return false;
    }
    std::cout << "Maillage chargé depuis " << filename << std::endl;
    std::cout << "  Nombre de sommets : " << mesh.n_vertices() << std::endl;
    std::cout << "  Nombre de faces : " << mesh.n_faces() << std::endl;
    return true;
}

bool MeshHandler::save_mesh(const std::string &filename, const Mesh &mesh) {
    if (!OpenMesh::IO::write_mesh(mesh, filename)) {
        std::cerr << "Erreur : Impossible d'écrire le maillage dans " << filename << std::endl;
        return false;
    }
    std::cout << "Maillage sauvegardé dans " << filename << std::endl;
    return true;
}

std::vector<Mesh> MeshHandler::initialize_intermediates(const Mesh &initial, const Mesh &final, int num_intermediates) {
    std::vector<Mesh> intermediates;
    intermediates.reserve(num_intermediates + 2);
    intermediates.push_back(initial);

    for (int k = 1; k <= num_intermediates; ++k) {
        Mesh mesh_intermediate = (k <= num_intermediates / 2) ? initial : final;
        intermediates.push_back(mesh_intermediate);
    }
    intermediates.push_back(final);
    return intermediates;
    
}
std::vector<double> MeshHandler::get_vertices(const Mesh &mesh) const {
    std::vector<double> vertices;
    for (const auto& vh : mesh.vertices()) {
        auto point = mesh.point(vh);
        vertices.push_back(point[0]);
        vertices.push_back(point[1]);
        vertices.push_back(point[2]);
    }
    return vertices;
}

std::vector<int> MeshHandler::get_simplices(const Mesh &mesh) const {
    std::vector<int> simplices;
    for (const auto& fh : mesh.faces()) {
        for (const auto& vh : mesh.fv_range(fh)) {
            simplices.push_back(vh.idx());
        }
    }
    return simplices;
}

Mesh MeshHandler::load_reference_mesh() {
    // Chargez le maillage de référence depuis un fichier ou utilisez le premier maillage du chemin
    Mesh mesh;
    load_mesh("../Meshes/hand/0.off", mesh);
    return mesh;
}

VectorType MeshHandler::get_geometry(const Mesh &mesh) const {
    VectorType geometry;
    int num_vertices = mesh.n_vertices();
    geometry.resize(num_vertices * 3); // Pour les coordonnées x, y, z

    int idx = 0;
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        Mesh::Point point = mesh.point(*v_it);
        geometry[idx++] = point[0];
        geometry[idx++] = point[1];
        geometry[idx++] = point[2];
    }

    return geometry;
}

MeshTopologySaver MeshHandler::get_topology(const Mesh &mesh) const {
    MeshTopologySaver topology(mesh);
    return topology;
}
VectorType MeshHandler::get_reference_geometry(const Mesh &mesh) const {
    // Ici, nous pouvons simplement appeler get_geometry
    return get_geometry(mesh);
}
