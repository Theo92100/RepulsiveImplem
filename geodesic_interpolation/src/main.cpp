#include "MeshHandler.hpp"
#include "EnergyCalculator.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <goast/Optimization/TrustRegionNewton.h>
#include <TotalEnergyFunctional.h>
#include <TotalEnergyGradient.h>
#include <TotalEnergyHessian.h>

// Fonction d'optimisation simple
typedef ShellDeformation<DefaultConfigurator, NonlinearMembraneDeformation<DefaultConfigurator>, SimpleBendingDeformation<DefaultConfigurator> > ShellDeformedType;

Eigen::SparseMatrix<double> construct_HGN(
    const std::vector<Mesh> &path,
    EnergyCalculator &energy_calculator
) {
    int n = path.size() - 1;  // Number of intervals between configurations
    int num_vertices = path[0].n_vertices();
    int dim = 3;  // Spatial dimension (3D)

    int dof_per_config = num_vertices * dim;
    int total_dof = (n - 1) * dof_per_config;  // Variables are x^1 to x^{n-1}

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;

    // Pre-compute gradients for each configuration (including fixed ones)
    std::vector<Eigen::VectorXd> gradients(path.size());
    for (size_t k = 0; k < path.size(); ++k) {
        gradients[k] = energy_calculator.compute_gradient_repulsive_config(path[k]);
        // Ensure gradient size matches
        if (gradients[k].size() != dof_per_config) {
            std::cerr << "Error: Gradient size mismatch at configuration " << k << std::endl;
            // Handle error appropriately
        }
    }

    // Construct the blocks of the H_GN matrix
    for (int k = 1; k <= n - 1; ++k) {  // k from 1 to n-1
        int idx_k = (k - 1) * dof_per_config;  // Index for x^k in variables
        const Eigen::VectorXd &grad_k = gradients[k];

        // Diagonal block A_k = 4n * grad_k * grad_k^T
        for (int i = 0; i < dof_per_config; ++i) {
            for (int j = 0; j < dof_per_config; ++j) {
                double value = 4 * n * grad_k(i) * grad_k(j);
                if (value != 0.0) {
                    tripletList.emplace_back(idx_k + i, idx_k + j, value);
                }
            }
        }

        // Off-diagonal block B_k = -2n * grad_k * grad_{k+1}^T
        if (k <= n - 2) {  // For k from 1 to n-2
            int idx_kp1 = idx_k + dof_per_config;  // Index for x^{k+1} in variables
            const Eigen::VectorXd &grad_kp1 = gradients[k + 1];

            for (int i = 0; i < dof_per_config; ++i) {
                for (int j = 0; j < dof_per_config; ++j) {
                    double value = -2 * n * grad_k(i) * grad_kp1(j);
                    if (value != 0.0) {
                        tripletList.emplace_back(idx_k + i, idx_kp1 + j, value);
                    }
                }
            }
        }
    }

    // Create the sparse matrix H_GN
    Eigen::SparseMatrix<double> H_GN(total_dof, total_dof);
    H_GN.setFromTriplets(tripletList.begin(), tripletList.end());

    return H_GN;
}

void update_path(std::vector<Mesh> &path, const Eigen::VectorXd &s) {
    int num_configs = path.size();
    int num_vertices = path[0].n_vertices();
    int dim = 3; // Dimension spatiale

    int dof_per_config = num_vertices * dim;

    // Exclure les configurations initiale et finale
    for (int k = 1; k < num_configs - 1; ++k) {
        int idx = (k - 1) * dof_per_config;
        Eigen::VectorXd delta = s.segment(idx, dof_per_config);

        // Mettre à jour les positions des sommets du maillage k
        auto &mesh = path[k];
        for (int i = 0; i < num_vertices; ++i) {
            Mesh::Point point = mesh.point(Mesh::VertexHandle(i));
            point[0] += delta(i * dim + 0);
            point[1] += delta(i * dim + 1);
            point[2] += delta(i * dim + 2);
            mesh.set_point(Mesh::VertexHandle(i), point);
        }
    }
}
void update_path_from_vector(std::vector<Mesh>& path, const Eigen::VectorXd& x) {
    size_t num_intermediates = path.size() - 2; // Exclude initial and final configurations if fixed
    size_t num_vertices = path[0].n_vertices();
    size_t dof_per_mesh = num_vertices * 3;

    for (size_t i = 1; i < path.size() - 1; ++i) {
        Mesh& mesh = path[i];
        size_t offset = (i - 1) * dof_per_mesh;
        size_t idx = 0;
        for (auto vh : mesh.vertices()) {
            double x_coord = x[offset + idx++];
            double y_coord = x[offset + idx++];
            double z_coord = x[offset + idx++];
            mesh.set_point(vh, OpenMesh::Vec3d(x_coord, y_coord, z_coord));
        }
    }
}
// Eigen::VectorXd compute_HGN_vector_product(
//     const std::vector<Mesh> &path,
//     const Eigen::VectorXd &v,
//     EnergyCalculator &energy_calculator
// 
//     int num_configs = path.size();
//     int num_vertices = path[0].n_vertices();
//     int dim = 3; // Dimension spatiale

//     int dof = (num_configs - 2) * num_vertices * dim; // Degrés de liberté
//     Eigen::VectorXd HGN_v = Eigen::VectorXd::Zero(dof);

//     // Calcul des contributions pour chaque configuration intermédiaire
//     for (int k = 1; k < num_configs - 1; ++k) {
//         int idx_k = (k - 1) * num_vertices * dim;
//         Eigen::VectorXd v_k = v.segment(idx_k, num_vertices * dim);

//         // Calculer la différentielle (gradient) pour la configuration k
//         Eigen::VectorXd grad_k = energy_calculator.compute_gradient_repulsive(path[k]);

//         // Calcul de A_k * v_k
//         Eigen::VectorXd A_k_v = 4 * num_configs * grad_k * (grad_k.dot(v_k));

//         Eigen::VectorXd total_contrib = A_k_v;

//         // Contribution de B_{k-1}^T * v_{k-1}
//         if (k > 1) {
//             int idx_km1 = (k - 2) * num_vertices * dim;
//             Eigen::VectorXd v_km1 = v.segment(idx_km1, num_vertices * dim);
//             Eigen::VectorXd grad_km1 = energy_calculator.compute_gradient_repulsive(path[k - 1]);
//             Eigen::VectorXd B_km1_v_km1 = -2 * num_configs * grad_k * (grad_km1.dot(v_km1));
//             total_contrib += B_km1_v_km1;
//         }

//         // Contribution de B_k * v_{k+1}
//         if (k < num_configs - 2) {
//             int idx_kp1 = k * num_vertices * dim;
//             Eigen::VectorXd v_kp1 = v.segment(idx_kp1, num_vertices * dim);
//             Eigen::VectorXd grad_kp1 = energy_calculator.compute_gradient_repulsive(path[k + 1]);
//             Eigen::VectorXd B_k_v_kp1 = -2 * num_configs * grad_k * (grad_kp1.dot(v_kp1));
//             total_contrib += B_k_v_kp1;
//         }

//         // Assignation de la contribution totale
//         HGN_v.segment(idx_k, num_vertices * dim) = total_contrib;
//     }

//     return HGN_v;
// }


Eigen::VectorXd path_to_vector(const std::vector<Mesh>& path) {
    size_t num_intermediates = path.size() - 2; // Exclude initial and final configurations if fixed
    size_t num_vertices = path[0].n_vertices(); // Assuming all meshes have the same number of vertices
    size_t dof_per_mesh = num_vertices * 3;
    size_t total_dof = num_intermediates * dof_per_mesh;

    Eigen::VectorXd x(total_dof);

    for (size_t i = 1; i < path.size() - 1; ++i) {
        const Mesh& mesh = path[i];
        size_t offset = (i - 1) * dof_per_mesh;
        size_t idx = 0;
        for (const auto& vh : mesh.vertices()) {
            OpenMesh::Vec3d point = mesh.point(vh);
            x[offset + idx++] = point[0];
            x[offset + idx++] = point[1];
            x[offset + idx++] = point[2];
        }
    }

    return x;
}



int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <config_initiale.off> <config_finale.off> <nombre_intermediates>" << std::endl;
        return 1;
    }

    std::string file_initial = argv[1];
    std::string file_final = argv[2];
    int num_intermediates = std::stoi(argv[3]);

    MeshHandler mesh_handler;
    Mesh mesh_initial, mesh_final;

    // Charger les maillages
    if (!mesh_handler.load_mesh(file_initial, mesh_initial)) {
        std::cerr << "Erreur lors du chargement du maillage initial." << std::endl;
        return 1;
    }
    if (!mesh_handler.load_mesh(file_final, mesh_final)) {
        std::cerr << "Erreur lors du chargement du maillage final." << std::endl;
        return 1;
    }

    // Initialiser les maillages intermédiaires
    std::vector<Mesh> path = mesh_handler.initialize_intermediates(mesh_initial, mesh_final, num_intermediates);

    std::cout << "Nombre total de configurations dans le chemin : " << path.size() << std::endl;

    // Initialize the energy calculator
    EnergyCalculator energy_calculator;

    // Get the topology and geometry
    MeshTopologySaver topology = mesh_handler.get_topology(mesh_initial);
    double bendWeight = 0.001;
    double max_radius = 100;
    double init_radius = 0.01;
    double max_iterations = 6;
    double tolerance = 1e-8;

    // Create the Shell Deformation instance
    ShellDeformedType W(topology, bendWeight);

    // Convert the path to the initial vector x0
    Eigen::VectorXd x0 = path_to_vector(path);

    // Instantiate the energy functional classes
    TotalEnergyFunctional<DefaultConfigurator> energyFunctional(energy_calculator, path, W);
    TotalEnergyGradient<DefaultConfigurator> energyGradient(energy_calculator, path, W);
    TotalEnergyHessian<DefaultConfigurator> energyHessian(energy_calculator, path, W);

    std::cout << "StartingTrust : " << std::endl;
    // Instantiate the optimizer
    TrustRegionNewton<DefaultConfigurator> optimizer(
        energyFunctional,
        energyGradient,
        energyHessian,
        init_radius,
        max_radius,
        tolerance,
        max_iterations
    );

    // Solve the optimization problem
    Eigen::VectorXd x_k;
    optimizer.solve(x0, x_k);
    std::cout << "Optimization completed." << std::endl;
    // Update the path with the optimized positions
    update_path_from_vector(path, x_k);

    // Save the optimized path
    std::string save_dir = "../Meshes/optimized/finger/";
    for (size_t k = 0; k < path.size(); ++k)
    {
        std::string filename = save_dir + std::to_string(k) + ".off";
        bool success = mesh_handler.save_mesh(filename, path[k]);
        if (!success)
        {
            std::cerr << "Erreur lors de la sauvegarde du maillage " << k << " dans " << filename << std::endl;
        }
    }
    std::cout << "Toutes les configurations optimisées ont été sauvegardées dans " << save_dir << std::endl;

    return 0;
}



// int main(int argc, char **argv) {
//     if (argc != 4) {
//         std::cerr << "Usage: " << argv[0] << " <config_initiale.off> <config_finale.off> <nombre_intermediates>" << std::endl;
//         return 1;
//     }
    

//     std::string file_initial = argv[1];
//     std::string file_final = argv[2];
//     int num_intermediates = std::stoi(argv[3]);

//     MeshHandler mesh_handler;
//     Mesh mesh_initial, mesh_final;

//     // Charger les maillages
//     if (!mesh_handler.load_mesh(file_initial, mesh_initial)) {
//         std::cerr << "Erreur lors du chargement du maillage initial." << std::endl;
//         return 1;
//     }
//     if (!mesh_handler.load_mesh(file_final, mesh_final)) {
//         std::cerr << "Erreur lors du chargement du maillage final." << std::endl;
//         return 1;
//     }

//     // Initialiser les maillages intermédiaires
//     std::vector<Mesh> path = mesh_handler.initialize_intermediates(mesh_initial, mesh_final, num_intermediates);

//     std::cout << "Nombre total de configurations dans le chemin : " << path.size() << std::endl;

//         // Ajouter le maillage final 
   


//     // Initialiser le calculateur d'énergie
//     EnergyCalculator energy_calculator;
//     // Calculer l'énergie répulsive pour la première configuration intermédiaire (par exemple, path[1] si path[0] est initial)
//     // Note : path[0] est initial, path[path.size()-1] est final
//     // Mesh mesh_reference = mesh_handler.load_reference_mesh();
//     VectorType referenceGeometry = mesh_handler.get_geometry(mesh_initial);
//     MeshTopologySaver topology = mesh_handler.get_topology(mesh_initial);
//     double bendWeight = 0.001;
//     // Créer l'instance de l'énergie de déformation néo-Hookéen
//     ShellDeformedType W(topology, bendWeight );
//     // Paramètres d'optimisation
//     double trust_region_radius = 1.0;
//     double tolerance = 1e-6;
//     int maxIterations = 1000;
//     int maxOuterIterations = 100; // Nombre maximum d'itérations externes


//     for (int iter = 0; iter < maxOuterIterations; ++iter) {
//         // Étape 1 : Calculer le gradient total c
//         Eigen::VectorXd gradient_elastic = energy_calculator.compute_gradient_elastic(path, W);
//         std::cout << "Taille de gradient_elastic : " << gradient_elastic.size() << std::endl;
//         Eigen::VectorXd gradient_repulsive = energy_calculator.compute_gradient_repulsive(path);
//         std::cout << "Taille de gradient_repulsive : " << gradient_repulsive.size() << std::endl;
//          // Ajouter des messages de débogage
    
//         Eigen::VectorXd c = gradient_elastic + gradient_repulsive;
//         std::cout << "Taille de c : " << c.size() << std::endl;

//         // Vérifier le critère d'arrêt
//         // if (c.norm() < tolerance) {
//         //     std::cout << "Convergence atteinte après " << iter << " itérations." << std::endl;
//         //     break;
//         // }

//         // Étape 2 : Construire le hessien total B
//         Eigen::SparseMatrix<double> hessian_elastic = energy_calculator.compute_hessian_elastic(path, W);
//         std::cout << "Taille de hessian_elastic : " << hessian_elastic.rows() << " x " << hessian_elastic.cols() << std::endl;
//         Eigen::SparseMatrix<double> H_GN = construct_HGN(path, energy_calculator);
//         std::cout << "Taille de H_GN : " << H_GN.rows() << " x " << H_GN.cols() << std::endl;
//         Eigen::SparseMatrix<double> B = hessian_elastic + H_GN;
//         B.makeCompressed();
        
//         if (!isFiniteSparseMatrix(B))
//         {
//             std::cerr << "Error: Hessian matrix contains NaN or Inf values." << std::endl;
//             return 1;
//         }

//         if (!c.allFinite())
//         {
//             std::cerr << "Error: Gradient vector contains NaN or Inf values." << std::endl;
//             return 1;
//         }

//         // Étape 3 : Créer une instance de SteihaugCGMethod
//         SteihaugCGMethod<DefaultConfigurator> steihaug_cg(
//             B,
//             c,
//             trust_region_radius,
//             tolerance,
//             maxIterations,
//             false,
         
//         );
//         std::cout <<"Initialisation de SteihaugCGMethod" << std::endl;

//         // Étape 4 : Résoudre le sous-problème
//         Eigen::VectorXd s;
//         steihaug_cg.solve(s);
//         std::cout << "Sous-problème résolu." << std::endl;
//         // Étape 5 : Mettre à jour le chemin avec la solution s
//         update_path(path, s);
//            // Optionnel : Afficher l'itération en cours
//         std::cout << "Itération " << iter + 1 << " complétée." << std::endl;
//     }
//          // **Sauvegarder les configurations optimisées**
//         std::string save_dir = "../Meshes/optimized/hand/";
//         for (size_t k = 0; k < path.size(); ++k)
//         {
//             std::string filename = save_dir + std::to_string(k) + ".off";
//             bool success = mesh_handler.save_mesh(filename, path[k]);
//             if (!success)
//             {
//                 std::cerr << "Erreur lors de la sauvegarde du maillage " << k << " dans " << filename << std::endl;
//             }
//         }
//         std::cout << "Toutes les configurations optimisées ont été sauvegardées dans " << save_dir << std::endl;

//         ////TEST
//         // double energy = energy_calculator.compute_elastic_energy(path, W);
//         // Eigen::VectorXd gradient = energy_calculator.compute_gradient_elastic(path, W);
//         // std::cout << "Gradient energie élastique : " << gradient << std::endl;
//         // Eigen::SparseMatrix<double> hessian = energy_calculator.compute_hessian_elastic(path, W);
//         // std::cout << "Hessien energie élastique : " << hessian << std::endl;

//         // // Supposons que path[0] est initial, path[1] est le premier intermédiaire
//         // Mesh first_intermediate = path[1];
//         // double repulsive_energy_first = energy_calculator.compute_repulsive_energy_config(first_intermediate);
//         // std::cout << "Énergie répulsive pour la première configuration intermédiaire : " << repulsive_energy_first << std::endl;

//         // // Calculer le gradient de l'énergie répulsive pour la première configuration intermédiaire
//         // Eigen::VectorXd gradient_first = energy_calculator.compute_gradient_repulsive(first_intermediate);
//         // std::cout << "Gradient de l'énergie répulsive pour la première configuration intermédiaire : " << gradient_first.transpose() << std::endl;

//         return 0;
// }
