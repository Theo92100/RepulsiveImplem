#ifndef ENERGY_CALCULATOR_HPP
#define ENERGY_CALCULATOR_HPP
#include <vector>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "goast/GeodesicCalculus.h"
#include <goast/DiscreteShells.h>
 // Ajouter GOAST
typedef OpenMesh::TriMesh_ArrayKernelT<TriTraits> Mesh;
typedef double RealType;
typedef Eigen::VectorXd VectorType;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> SparseMatrixType;



class EnergyCalculator {
public:
    EnergyCalculator();
    Eigen::VectorXd compute_gradient_repulsive(const std::vector<Mesh> &path);
    double compute_repulsive_energy(const std::vector<Mesh> &path);
    double compute_repulsive_energy_config(const Mesh &mesh); 
    Eigen::VectorXd compute_gradient_repulsive_config(const Mesh &mesh); // Calcule le gradient de l'énergie répulsive
    Eigen::MatrixXd compute_hessian_approx(const std::vector<Mesh> &path);
    double compute_elastic_energy(const std::vector<Mesh> &path,const DeformationBase<DefaultConfigurator> &W);
    Eigen::VectorXd compute_gradient_elastic(const std::vector<Mesh> &path,const DeformationBase<DefaultConfigurator> &W);
    Eigen::SparseMatrix<double> compute_hessian_elastic(const std::vector<Mesh> &path,const DeformationBase<DefaultConfigurator> &W);
private:
    void concatenate_intermediate_geometries(const std::vector<VectorType> &geometries, VectorType &path_vector);
};

#endif // ENERGY_CALCULATOR_HPP