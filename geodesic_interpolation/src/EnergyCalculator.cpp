#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include "MeshHandler.hpp"
#include "EnergyCalculator.hpp"
#include <iostream>
#include <vector>
#include "Repulsor.hpp"
// Inclure les en-têtes nécessaires de GOAST et Repulsor

using namespace Repulsor;

using Int     = int;
using LInt    = std::size_t;
using Real    = double;


const int dom_dim = 2;
const int amb_dim = 3;
SimplicialMesh_Factory<SimplicialMeshBase<Real, Int, LInt>, dom_dim, dom_dim, amb_dim, amb_dim> mesh_factory;

EnergyCalculator::EnergyCalculator() {
    // Initialisation si nécessaire
}

double EnergyCalculator::compute_elastic_energy(const std::vector<Mesh> &path,const DeformationBase<DefaultConfigurator> &W) {
    double total_energy = 0.0;

    // Nombre de configurations dans le chemin
    size_t K = path.size() - 1;

    // Convertir les maillages en géométries (vecteurs)
    std::vector<VectorType> geometries;
    MeshHandler mesh_handler;
    for (const auto &mesh : path) {
        VectorType geometry = mesh_handler.get_geometry(mesh);
        geometries.push_back(geometry);
    }

    // Calculer l'énergie du chemin discret
    DiscretePathEnergy<DefaultConfigurator> E(W, K, geometries.front(), geometries.back());

    // Préparer le vecteur concaténé des configurations intermédiaires
    VectorType path_vector;
    concatenate_intermediate_geometries(geometries, path_vector);

    // Calculer l'énergie
    RealType energy;
    E.apply(path_vector, energy);

    total_energy = energy;

    return total_energy;
}
Eigen::VectorXd EnergyCalculator::compute_gradient_elastic(const std::vector<Mesh> &path,const DeformationBase<DefaultConfigurator> &W) {
    // Nombre de configurations dans le chemin
    size_t K = path.size() - 1;

    // Convertir les maillages en géométries (vecteurs)
    std::vector<VectorType> geometries;
    MeshHandler mesh_handler;
    for (const auto &mesh : path) {
        VectorType geometry = mesh_handler.get_geometry(mesh);
        geometries.push_back(geometry);
    }

    // Calculer le gradient de l'énergie du chemin discret
    DiscretePathEnergyGradient<DefaultConfigurator> DE(W, K, geometries.front(), geometries.back());

    // Préparer le vecteur concaténé des configurations intermédiaires
    VectorType path_vector;
    concatenate_intermediate_geometries(geometries, path_vector);

    // Initialiser le vecteur gradient
    VectorType gradient(path_vector.size());
    gradient.setZero();


    // Calculer le gradient
    DE.apply(path_vector, gradient);

    return gradient;
}
Eigen::SparseMatrix<double> EnergyCalculator::compute_hessian_elastic(const std::vector<Mesh> &path,const DeformationBase<DefaultConfigurator> &W) {
    // Nombre de configurations dans le chemin
    size_t K = path.size()-1;

    // Convertir les maillages en géométries (vecteurs)
    std::vector<VectorType> geometries;
    MeshHandler mesh_handler;
    for (const auto &mesh : path) {
        VectorType geometry = mesh_handler.get_geometry(mesh);
        geometries.push_back(geometry);
    }


    // Créer l'instance de l'énergie de déformation

    // Calculer le hessien de l'énergie du chemin discret
    DiscretePathEnergyHessian<DefaultConfigurator> D2E(W, K, geometries.front(), geometries.back());

    // Préparer le vecteur concaténé des configurations intermédiaires
    VectorType path_vector;
    concatenate_intermediate_geometries(geometries, path_vector);

    // Initialiser la matrice hessienne
    SparseMatrixType hessian;
    // Calculer le hessien
    D2E.apply(path_vector, hessian);

    return hessian;
}
double EnergyCalculator::compute_repulsive_energy(const std::vector<Mesh> &path) {
    double total_energy = 0.0;

    Real q = 6;
    Real p = 12;

    // Créer les usines pour le maillage et l'énergie
    SimplicialMesh_Factory<SimplicialMeshBase<Real, Int, LInt>, dom_dim, dom_dim, amb_dim, amb_dim> mesh_factory;
    TangentPointEnergy0_Factory<SimplicialMeshBase<Real, Int, LInt>, dom_dim, dom_dim, amb_dim, amb_dim> TPE_factory;

    MeshHandler mesh_handler;

    // Parcourir chaque configuration du chemin
    for (const auto &mesh : path) {
        // Extraire les données du maillage
        std::vector<Real> vertex_coordinates = mesh_handler.get_vertices(mesh);
        std::vector<Int> simplices = mesh_handler.get_simplices(mesh);

        LInt vertex_count = vertex_coordinates.size() / amb_dim;
        LInt simplex_count = simplices.size() / (dom_dim + 1);

        // Créer le maillage Repulsor
        std::unique_ptr<SimplicialMeshBase<Real, Int, LInt>> M_ptr = mesh_factory.Make(
            vertex_coordinates.data(), vertex_count, amb_dim, false,
            simplices.data(), simplex_count, dom_dim + 1, false,
            1 // Nombre de threads (ajustez si nécessaire)
        );

        auto &M = *M_ptr;

        // Créer l'énergie
        std::unique_ptr<EnergyBase<SimplicialMeshBase<Real, Int, LInt>>> tpe_ptr = TPE_factory.Make(dom_dim, amb_dim, q, p);
        const auto &tpe = *tpe_ptr;

        // Calculer l'énergie répulsive pour ce maillage
        Real energy = tpe.Value(M);

        // Ajouter à l'énergie totale
        total_energy += energy;
    }

    return total_energy;
}

double EnergyCalculator::compute_repulsive_energy_config(const Mesh &mesh) {
    double total_energy = 0.0;

    Real q = 6;
    Real p = 12;

    // Créer les usines pour le maillage et l'énergie
    SimplicialMesh_Factory<SimplicialMeshBase<Real, Int, LInt>, dom_dim, dom_dim, amb_dim, amb_dim> mesh_factory;
    TangentPointEnergy0_Factory<SimplicialMeshBase<Real, Int, LInt>, dom_dim, dom_dim, amb_dim, amb_dim> TPE_factory;

    MeshHandler mesh_handler;

    // Extraire les données du maillage
    std::vector<Real> vertex_coordinates = mesh_handler.get_vertices(mesh);
    std::vector<Int> simplices = mesh_handler.get_simplices(mesh);

    LInt vertex_count = vertex_coordinates.size() / amb_dim;
    LInt simplex_count = simplices.size() / (dom_dim + 1);

    // Créer le maillage Repulsor
    std::unique_ptr<SimplicialMeshBase<Real, Int, LInt>> M_ptr = mesh_factory.Make(
        vertex_coordinates.data(), vertex_count, amb_dim, false,
        simplices.data(), simplex_count, dom_dim + 1, false,
        1 // Nombre de threads (1 ici, à ajuster si nécessaire)
    );

    auto &M = *M_ptr;

    // Créer l'énergie
    std::unique_ptr<EnergyBase<SimplicialMeshBase<Real, Int, LInt>>> tpe_ptr = TPE_factory.Make(dom_dim, amb_dim, q, p);
    const auto &tpe = *tpe_ptr;

    // Calculer l'énergie répulsive pour ce maillage
    Real energy = tpe.Value(M);

    total_energy += energy;

    return total_energy;
}


Eigen::VectorXd EnergyCalculator::compute_gradient_repulsive_config(const Mesh &mesh) {
    Real q = 6;
    Real p = 12;

    // Créer les usines pour l'énergie
    TangentPointEnergy0_Factory<SimplicialMeshBase<Real, Int, LInt>, dom_dim, dom_dim, amb_dim, amb_dim> TPE_factory;

    MeshHandler mesh_handler;

    // Extraire les données du maillage
    std::vector<Real> vertex_coordinates = mesh_handler.get_vertices(mesh);
    std::vector<Int> simplices = mesh_handler.get_simplices(mesh);

    LInt vertex_count = vertex_coordinates.size() / amb_dim;
    LInt simplex_count = simplices.size() / (dom_dim + 1);

    // Créer le maillage Repulsor
    std::unique_ptr<SimplicialMeshBase<Real, Int, LInt>> M_ptr = mesh_factory.Make(
        vertex_coordinates.data(), vertex_count, amb_dim, false,
        simplices.data(), simplex_count, dom_dim + 1, false,
        1 // Nombre de threads (1 ici, à ajuster si nécessaire)
    );

    auto &M = *M_ptr;

    // Créer l'énergie TPE
    std::unique_ptr<EnergyBase<SimplicialMeshBase<Real, Int, LInt>>> tpe_ptr = TPE_factory.Make(dom_dim, amb_dim, q, p);
    const auto &tpe = *tpe_ptr;

    // Initialiser le vecteur de gradient
    Eigen::VectorXd gradient(M.VertexCount() * M.AmbDim());
    gradient.setZero();

    // Calculer le gradient de l'énergie répulsive pour ce maillage
    // La méthode Differential prend en entrée le maillage et un pointeur vers les données du gradient
    tpe.Differential(M, gradient.data());

    return gradient;
}

Eigen::VectorXd EnergyCalculator::compute_gradient_repulsive(const std::vector<Mesh> &path) {
    // Assume all meshes have the same number of vertices
    int num_vertices = path[0].n_vertices();
    int dim = 3; // Dimension of space (3D)
    int total_dofs = num_vertices * dim * (path.size() - 2); // Exclude initial and final configurations

    Eigen::VectorXd total_gradient = Eigen::VectorXd::Zero(total_dofs);

    for (int k = 1; k < path.size() - 1; ++k) {
        const Mesh &mesh = path[k];
        // Compute the repulsive gradient for the current mesh
        Eigen::VectorXd gradient = compute_gradient_repulsive_config(mesh);

        // Insert the gradient into the total vector
        int idx = (k - 1) * num_vertices * dim;
        total_gradient.segment(idx, num_vertices * dim) = gradient;
    }

    return total_gradient;
}



///HELPER FUNCTIONS

void EnergyCalculator::concatenate_intermediate_geometries(
    const std::vector<VectorType> &geometries,
    VectorType &path_vector
) {
    // Exclure les premières et dernières géométries (géométries de départ et d'arrivée)
    size_t num_intermediates = geometries.size() - 2;
    int num_dofs = geometries[0].size();

    path_vector.resize(num_intermediates * num_dofs);

    for (size_t k = 1; k <= num_intermediates; ++k) {
        path_vector.segment((k - 1) * num_dofs, num_dofs) = geometries[k];
    }
}